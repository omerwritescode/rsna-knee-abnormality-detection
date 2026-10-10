"""Choose blend weights for M1 + any number of M2 runs from out-of-fold predictions.

    python -m src.models.blend --m2 ft1,ft2                 # M1 v2 + two M2 runs
    python -m src.models.blend --m1 none --m2 ft1,ft2       # M2 runs only

Scores every model alone, the equal-weight average, and every weighted probability
average on a grid (step 0.05 for up to 3 models, 0.1 for more) on the studies that have
OOF predictions from all models. The best weights by auc_llm (the CV number that has
matched the public LB) go to runs/blend.json, which scripts/make_model_dataset.py reads
to decide which runs go into the Kaggle dataset.

blend.json: {"weights": {"m1": w, "m2_ft1": w, ...}, "runs": {"m1": "v2", "m2_ft1": "ft1", ...},
             "auc_llm": ..., ...}   -- keys are the folder names inside the Kaggle dataset.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .folds import read_labels
from .metrics import LABELS, score


def load_m1(root: Path, run: str) -> pd.DataFrame:
    return pd.read_csv(root / "runs" / "m1" / run / "oof.csv", dtype={"StudyInstanceUID": str})


def load_m2(root: Path, run: str) -> pd.DataFrame:
    parts = sorted((root / "runs" / "m2" / run).glob("oof_fold*.csv"))
    if not parts:
        raise SystemExit(f"no oof_fold*.csv in runs/m2/{run}")
    if len(parts) < 5:
        print(f"WARNING: runs/m2/{run} has only {len(parts)} folds of OOF predictions")
    o = pd.concat([pd.read_csv(p, dtype={"StudyInstanceUID": str}) for p in parts])
    return o.drop_duplicates("StudyInstanceUID", keep="last")


def weight_grid(n: int, step: float):
    k = int(round(1 / step))
    for c in itertools.product(range(k + 1), repeat=n - 1):
        if sum(c) <= k:
            yield np.array([*c, k - sum(c)], float) / k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--m1", default="v2", help="M1 run, or 'none'")
    ap.add_argument("--m2", default="ft1", help="comma-separated M2 runs, e.g. ft1,ft2")
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default runs/blend.json")
    args = ap.parse_args()

    models: dict[str, tuple[str, pd.DataFrame]] = {}       # dataset folder -> (run, oof)
    if args.m1.lower() != "none":
        models["m1"] = (args.m1, load_m1(args.root, args.m1))
    for run in [r.strip() for r in args.m2.split(",") if r.strip()]:
        models[f"m2_{run}"] = (run, load_m2(args.root, run))
    if not models:
        raise SystemExit("no models given")

    labels = read_labels(args.labels or args.root / "data" / "labels" / "soft_labels.parquet")
    df = labels[["StudyInstanceUID", "source", *LABELS]]
    for key, (_, o) in models.items():
        df = df.merge(o[["StudyInstanceUID", *LABELS]].rename(
            columns={l: f"{l}|{key}" for l in LABELS}), on="StudyInstanceUID")
    y, src = df[LABELS].to_numpy(), df["source"].to_numpy()
    keys = list(models)
    P = np.stack([df[[f"{l}|{k}" for l in LABELS]].to_numpy() for k in keys])   # [M, n, 12]
    print(f"{len(df)} studies with OOF from all of: {', '.join(keys)}\n")

    def macro(w):
        sc = score(np.tensordot(w, P, 1), y, src)
        return sc.loc["MACRO", "auc_llm"], sc.loc["MACRO", "auc_human"]

    rows = []
    for i, k in enumerate(keys):
        w = np.eye(len(keys))[i]
        rows.append(dict(blend=f"{k} alone", w=w, **dict(zip(("auc_llm", "auc_human"), macro(w)))))
    if len(keys) > 1:
        w = np.full(len(keys), 1 / len(keys))
        rows.append(dict(blend="equal weights", w=w, **dict(zip(("auc_llm", "auc_human"), macro(w)))))
    step = 0.05 if len(keys) <= 3 else 0.1
    grid = [dict(w=w, auc_llm=macro(w)[0]) for w in weight_grid(len(keys), step)] \
        if len(keys) > 1 else []
    if grid:
        best = max(grid, key=lambda r: r["auc_llm"])
        rows.append(dict(blend="best grid", w=best["w"],
                         **dict(zip(("auc_llm", "auc_human"), macro(best["w"])))))
    tab = pd.DataFrame(rows)
    tab["weights"] = tab["w"].apply(lambda w: " ".join(f"{k}={x:.2f}" for k, x in zip(keys, w)))
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(tab[["blend", "auc_llm", "auc_human", "weights"]].to_string(index=False))

    chosen = tab.iloc[-1]
    w = chosen["w"]
    per_label = pd.DataFrame({k: score(P[i], y, src)["auc_llm"] for i, k in enumerate(keys)})
    if len(keys) > 1:
        per_label["blend"] = score(np.tensordot(w, P, 1), y, src)["auc_llm"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print("\nper-label auc_llm:\n" + per_label.to_string())

    out = args.out or args.root / "runs" / "blend.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "weights": {k: round(float(x), 3) for k, x in zip(keys, w)},
        "runs": {k: run for k, (run, _) in models.items()},
        "auc_llm": round(float(chosen["auc_llm"]), 4),
        "auc_human": round(float(chosen["auc_human"]), 4),
        "n_studies": int(len(df))}, indent=2))
    print(f"\nchosen: {chosen['weights']}  ->  auc_llm {chosen['auc_llm']:.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
