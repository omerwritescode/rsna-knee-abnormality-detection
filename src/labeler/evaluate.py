"""Score labeler output against the human labels.

    python -m src.labeler.evaluate data/labels/raw_v1_dev.jsonl --errors
    python -m src.labeler.evaluate data/labels/raw_v1_heldout.jsonl

Positive = score >= 0.5 (present or uncertain). Also shows the stricter
"present only" sensitivity and the AUC of the soft score (1 / 0.5 / 0).
`--errors` prints every disagreement with the model's evidence; it refuses on
the held-out split unless you add --allow-heldout-errors (looking at those
errors while still tuning the prompt turns held-out into a second dev set).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from .prompt import LABELS
from .run import normalise
from . import postprocess


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            try:
                st.write(text)
            except UnicodeEncodeError:
                st.write(text.encode("ascii", "replace").decode())

    def flush(self):
        for st in self.streams:
            st.flush()


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok"):
                rows[rec["StudyInstanceUID"]] = rec  # last one wins
    return pd.DataFrame(rows.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--errors", action="store_true")
    ap.add_argument("--allow-heldout-errors", action="store_true")
    ap.add_argument("--labels", nargs="*", default=None, help="limit --errors to these")
    ap.add_argument("--no-post", action="store_true", help="skip postprocess rules")
    ap.add_argument("--save", type=Path, default=None,
                    help="also write the output to this file (UTF-8)")
    args = ap.parse_args()
    if args.save:
        # PowerShell's Out-File re-encodes non-Latin text badly; write it ourselves
        sys.stdout = Tee(sys.stdout, open(args.save, "w", encoding="utf-8"))

    recs = load_jsonl(args.jsonl)
    train = pd.read_csv(args.root / "data" / "raw" / "train.csv")
    split = pd.read_csv(args.root / "data" / "labels" / "split.csv")
    truth = train.merge(split, on="StudyInstanceUID").set_index("StudyInstanceUID")
    recs = recs[recs["StudyInstanceUID"].isin(truth.index)].set_index("StudyInstanceUID")
    if recs.empty:
        raise SystemExit("no labeled studies in this file")
    splits = truth.loc[recs.index, "split"].unique().tolist()
    print(f"{len(recs)} labeled studies scored · splits: {splits} · "
          f"prompt {recs['prompt'].iloc[0]}\n")

    if args.no_post:
        S = pd.DataFrame(recs["scores"].tolist(), index=recs.index)[LABELS]
        print("post-processing: OFF\n")
    else:
        post = [postprocess.apply(sc, fi) for sc, fi in zip(recs["scores"], recs["findings"])]
        S = pd.DataFrame([p[0] for p in post], index=recs.index)[LABELS]
        n_fired = sum(bool(p[1]) for p in post)
        print(f"post-processing {postprocess.POST_VERSION}: changed {n_fired} studies "
              f"(use --no-post to see raw model scores)\n")
    Y = truth.loc[recs.index, LABELS].astype(int)

    rows = []
    for lab in LABELS:
        y, s = Y[lab].to_numpy(), S[lab].to_numpy()
        p = (s >= 0.5).astype(int)
        strict = (s >= 1.0).astype(int)
        pos, neg = y == 1, y == 0
        rows.append({
            "label": lab,
            "pos": int(pos.sum()),
            "acc": (p == y).mean(),
            "sens": p[pos].mean() if pos.any() else np.nan,
            "sens_strict": strict[pos].mean() if pos.any() else np.nan,
            "spec": 1 - p[neg].mean() if neg.any() else np.nan,
            "kappa": cohen_kappa_score(y, p) if len(set(y)) > 1 or len(set(p)) > 1 else np.nan,
            "auc_soft": roc_auc_score(y, s) if 0 < pos.sum() < len(y) else np.nan,
            "n_unc": int((s == 0.5).sum()),
        })
    tab = pd.DataFrame(rows).set_index("label")
    tab.loc["MACRO"] = tab.mean(numeric_only=True)
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(tab.to_string())

    tok = recs["eval_tokens"].dropna()
    if len(tok):
        print(f"\noutput tokens: mean {tok.mean():.0f}, max {tok.max():.0f} · "
              f"seconds/report: mean {recs['seconds'].mean():.1f}")
    if "gen_s" in recs:
        t = recs[["prompt_tokens", "prompt_s", "eval_tokens", "gen_s", "load_s"]].mean()
        print(f"per report: prompt {t.prompt_tokens:.0f} tok in {t.prompt_s:.2f}s · "
              f"output {t.eval_tokens:.0f} tok in {t.gen_s:.2f}s · load {t.load_s:.2f}s")
    if recs.get("truncated", pd.Series(False)).any():
        print("WARNING: some outputs hit num_predict and were cut off")

    if not args.errors:
        return
    if "heldout" in splits and not args.allow_heldout_errors:
        raise SystemExit("\nrefusing --errors on held-out (see docstring)")
    show = args.labels or LABELS
    print("\n=== DISAGREEMENTS (FN = label 1, model missed · FP = label 0, model flagged)")
    reports = train.set_index("StudyInstanceUID")["Report"]
    for uid in recs.index:
        bad = [(lab, Y.at[uid, lab], S.at[uid, lab]) for lab in show
               if (S.at[uid, lab] >= 0.5) != bool(Y.at[uid, lab])]
        if not bad:
            continue
        ev = {lab: f"{f.get('s')}: {f.get('q', '')}"
              for lab, f in normalise(recs.at[uid, "findings"]).items()}
        print(f"\n--- {uid[-8:]}")
        for lab, y, s in bad:
            kind = "FN" if y else "FP"
            print(f"  {kind} {lab:<17} score={s:.1f}  {ev.get(lab, '(not listed)')}")
        print("  report:", " ".join(str(reports[uid]).split())[:600], "...")


if __name__ == "__main__":
    main()
