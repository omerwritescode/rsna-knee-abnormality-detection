"""Split the 58 human-labeled studies into dev (20) / held-out (38).

Usage (from the project root):
    python -m src.labeler.split
    python -m src.labeler.split --n-dev 20 --seed 2026

Why not iterative-stratification: with 58 studies a direct search is better.
We draw many random splits from a fixed seed and keep the one whose per-label
positive counts in dev are closest to the target proportion (20/58), with a
hard requirement that every label has >= MIN_POS positives in BOTH splits.
If study_meta_chunk*.csv files exist, studies from the same PatientID are kept
on the same side of the split.

Output: data/labels/split.csv  (StudyInstanceUID, split, PatientID)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
MIN_POS = 2  # minimum positives per label in each split (MCL has 9 in total)
N_TRIALS = 200_000


def load_patient_ids(raw: Path) -> pd.Series | None:
    files = sorted(raw.glob("study_meta_chunk*.csv"))
    if not files:
        return None
    meta = pd.concat([pd.read_csv(f, usecols=["StudyInstanceUID", "PatientID"],
                                  dtype=str) for f in files])
    return meta.set_index("StudyInstanceUID")["PatientID"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--n-dev", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    raw = args.root / "data" / "raw"
    out = args.root / "data" / "labels" / "split.csv"
    df = pd.read_csv(raw / "train.csv")
    lab = df.dropna(subset=LABELS).reset_index(drop=True)
    Y = lab[LABELS].to_numpy().astype(int)
    n = len(lab)
    print(f"labeled studies: {n}")

    # --- patient grouping -------------------------------------------------
    pids = load_patient_ids(raw)
    if pids is None:
        print("WARNING: no study_meta_chunk*.csv found -> splitting without "
              "patient grouping")
        groups = np.arange(n)
        lab["PatientID"] = ""
    else:
        lab["PatientID"] = lab["StudyInstanceUID"].map(pids).fillna("")
        missing = (lab["PatientID"] == "").sum()
        key = lab["PatientID"].where(lab["PatientID"] != "",
                                     lab["StudyInstanceUID"])
        groups = pd.factorize(key)[0]
        multi = pd.Series(groups).value_counts()
        print(f"patients: {groups.max() + 1} for {n} studies "
              f"({(multi > 1).sum()} with >1 study, {missing} missing PatientID)")

    # Group-level label sums, so a patient's studies move together
    G = groups.max() + 1
    gY = np.zeros((G, len(LABELS)), dtype=int)
    np.add.at(gY, groups, Y)
    gsize = np.bincount(groups, minlength=G)

    totals = Y.sum(0)
    frac = args.n_dev / n
    target = totals * frac

    rng = np.random.default_rng(args.seed)
    best, best_score = None, np.inf
    for _ in range(N_TRIALS):
        order = rng.permutation(G)
        take = order[np.cumsum(gsize[order]) <= args.n_dev]
        if gsize[take].sum() != args.n_dev:
            continue
        dev_pos = gY[take].sum(0)
        if (dev_pos < MIN_POS).any() or ((totals - dev_pos) < MIN_POS).any():
            continue
        # relative squared deviation, so rare labels weigh as much as common
        score = (((dev_pos - target) / np.sqrt(totals)) ** 2).sum()
        if score < best_score:
            best, best_score = np.sort(take), score
    if best is None:
        raise SystemExit("no valid split found - lower MIN_POS")

    is_dev = np.isin(groups, best)
    lab["split"] = np.where(is_dev, "dev", "heldout")
    out.parent.mkdir(parents=True, exist_ok=True)
    lab[["StudyInstanceUID", "split", "PatientID"]].to_csv(out, index=False)

    summary = pd.DataFrame({
        "total": totals,
        "dev": Y[is_dev].sum(0),
        "heldout": Y[~is_dev].sum(0),
        "dev_target": np.round(target, 1),
    }, index=LABELS)
    print(summary.to_string())
    print(f"dev={is_dev.sum()} heldout={(~is_dev).sum()}  score={best_score:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
