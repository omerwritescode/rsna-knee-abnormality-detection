"""Diagnostic baseline: logistic regression on pooled frozen DINOv2 features.

    python -m src.models.baseline            # mean + max over slices, CLS+mean features
    python -m src.models.baseline --grid     # also the 2x2 patch grid (slower)

Per study: for each of the 4 series kinds, the per-channel mean AND max over its 32
slices (standardised features), plus 4 "kind present" flags. One logistic regression
per label, same 5 folds as M1, binary target (soft label >= 0.5).

Purpose: tells us whether the frozen features or the M1 aggregator is the bottleneck.
If this scores close to M1, the features are the limit; if clearly higher, M1 needs work.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import KIND_SLOT, load_cache, load_stats
from .folds import read_labels
from .metrics import LABELS, score


def study_features(cache: dict, stats: dict, use_grid: bool) -> np.ndarray:
    feats, grid, present = cache["feats"], cache["grid"], cache["present"]
    n = len(feats)
    blocks = []
    for slot in KIND_SLOT:
        f = (feats[:, slot].astype(np.float32) - stats["f_mean"]) / stats["f_std"]   # [N,32,768]
        parts = [f.mean(1), f.max(1)]
        if use_grid:
            g = (grid[:, slot].astype(np.float32) - stats["g_mean"]) / stats["g_std"]  # [N,32,c,384]
            g = g.reshape(n, g.shape[1], -1)
            parts += [g.mean(1), g.max(1)]
        x = np.concatenate(parts, 1)
        x[~present[:, slot]] = 0.0
        blocks.append(x)
    blocks.append(present[:, KIND_SLOT].astype(np.float32))
    return np.concatenate(blocks, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--tag", default="dinov2s_224_g4")
    ap.add_argument("--cache", default="cache_g2")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--C", type=float, default=0.01, help="inverse L2 strength")
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()

    cache_dir = args.root / "data" / "embeddings" / args.tag / args.cache
    cache = load_cache(cache_dir, in_memory=False)
    t0 = time.time()
    stats = load_stats(cache_dir, cache)
    X = study_features(cache, stats, args.grid)
    print(f"features {X.shape} in {time.time() - t0:.0f}s")

    labels = read_labels(args.labels or args.root / "data" / "labels" / "soft_labels.parquet")
    folds = pd.read_csv(args.root / "data" / "labels" / "folds.csv", dtype={"StudyInstanceUID": str})
    row_of = {u: i for i, u in enumerate(cache["uids"])}
    df = folds.merge(labels[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID")
    df["row"] = df["StudyInstanceUID"].map(row_of)
    df = df.dropna(subset=["row"]).reset_index(drop=True)
    X = X[df["row"].astype(int).to_numpy()]
    Y = df[LABELS].to_numpy()

    oof = np.zeros_like(Y, dtype=float)
    for k in sorted(df["fold"].unique()):
        tr, va = df["fold"].to_numpy() != k, df["fold"].to_numpy() == k
        sc = StandardScaler().fit(X[tr])
        xtr, xva = sc.transform(X[tr]), sc.transform(X[va])
        for j in range(len(LABELS)):
            y = Y[tr, j] >= 0.5
            if y.min() == y.max():          # one class only in this fold
                oof[va, j] = y.mean()
                continue
            clf = LogisticRegression(C=args.C, max_iter=3000)
            clf.fit(xtr, y)
            oof[va, j] = clf.predict_proba(xva)[:, 1]
        print(f"  fold {k} done ({time.time() - t0:.0f}s)")

    res = score(oof, Y, df["source"].to_numpy())
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(f"\nlogistic-regression baseline (grid={args.grid}, C={args.C}):")
        print(res.to_string())


if __name__ == "__main__":
    main()
