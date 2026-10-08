"""Choose the M1 / M2 blend weight from out-of-fold predictions.

    python -m src.models.blend --m1 v2 --m2 ft1

Scores M1 alone, M2 alone and weighted probability averages (M2 weight 0.0 ... 1.0) on
the studies that have OOF predictions from both, and writes the best weight to
runs/m2/<m2>/blend.json (picked up by scripts/make_model_dataset.py and the notebook).
Uses auc_llm (the CV number that matched the public LB) to choose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .folds import read_labels
from .metrics import LABELS, score


def load_oof(root: Path, m1: str, m2: str):
    o1 = pd.read_csv(root / "runs" / "m1" / m1 / "oof.csv", dtype={"StudyInstanceUID": str})
    parts = sorted((root / "runs" / "m2" / m2).glob("oof_fold*.csv"))
    if not parts:
        raise SystemExit(f"no oof_fold*.csv in runs/m2/{m2}")
    o2 = pd.concat([pd.read_csv(p, dtype={"StudyInstanceUID": str}) for p in parts])
    o2 = o2.drop_duplicates("StudyInstanceUID", keep="last")
    return o1, o2, [p.stem.replace("oof_", "") for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--m1", default="v2")
    ap.add_argument("--m2", default="ft1")
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()

    o1, o2, folds = load_oof(args.root, args.m1, args.m2)
    labels = read_labels(args.labels or args.root / "data" / "labels" / "soft_labels.parquet")
    df = labels[["StudyInstanceUID", "source", *LABELS]].merge(
        o1[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID", suffixes=("", "_m1")).merge(
        o2[["StudyInstanceUID", *LABELS]].rename(columns={l: f"{l}_m2" for l in LABELS}),
        on="StudyInstanceUID")
    y, src = df[LABELS].to_numpy(), df["source"].to_numpy()
    p1 = df[[f"{l}_m1" for l in LABELS]].to_numpy()
    p2 = df[[f"{l}_m2" for l in LABELS]].to_numpy()
    print(f"{len(df)} studies with OOF from both models (M2 {', '.join(folds)})\n")

    rows = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        sc = score((1 - w) * p1 + w * p2, y, src)
        rows.append(dict(w_m2=w, auc_llm=sc.loc["MACRO", "auc_llm"],
                         auc_human=sc.loc["MACRO", "auc_human"]))
    tab = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(tab.to_string(index=False))
    best = tab.loc[tab["auc_llm"].idxmax()]
    w = float(best["w_m2"])
    print(f"\nbest: M2 weight {w:.1f} -> auc_llm {best['auc_llm']:.4f} "
          f"(M2 alone {tab.iloc[-1]['auc_llm']:.4f}, M1 alone {tab.iloc[0]['auc_llm']:.4f})")

    per_label = pd.DataFrame({
        "M1": score(p1, y, src)["auc_llm"], "M2": score(p2, y, src)["auc_llm"],
        "blend": score((1 - w) * p1 + w * p2, y, src)["auc_llm"]})
    with pd.option_context("display.float_format", "{:.4f}".format):
        print("\nper-label auc_llm:\n" + per_label.to_string())

    out = args.root / "runs" / "m2" / args.m2 / "blend.json"
    out.write_text(json.dumps({"m1": round(1 - w, 2), "m2": round(w, 2),
                               "auc_llm": round(float(best["auc_llm"]), 4),
                               "n_studies": int(len(df)), "m2_folds": folds,
                               "m1_run": args.m1, "m2_run": args.m2}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
