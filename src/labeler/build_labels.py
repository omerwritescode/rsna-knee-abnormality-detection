"""Build data/labels/soft_labels.parquet from a full labeling run.

    python -m src.labeler.build_labels data/labels/raw_v1_all.jsonl

The 58 human-labeled studies keep their true labels (source="human");
everything else gets the LLM soft labels 1 / 0.5 / 0 (source="llm").
Studies with no successful LLM output get NaN and source="missing".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluate import load_jsonl
from .prompt import LABELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    train = pd.read_csv(args.root / "data" / "raw" / "train.csv")
    recs = load_jsonl(args.jsonl).set_index("StudyInstanceUID")
    llm = pd.DataFrame(recs["scores"].tolist(), index=recs.index)[LABELS]

    out = train[["StudyInstanceUID"]].set_index("StudyInstanceUID")
    out = out.join(llm)
    out["source"] = "llm"
    out.loc[out[LABELS].isna().all(axis=1), "source"] = "missing"

    human = train.dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS]
    out.loc[human.index, LABELS] = human.astype(float)
    out.loc[human.index, "source"] = "human"
    out["prompt"] = recs["prompt"].reindex(out.index)

    path = args.root / "data" / "labels" / "soft_labels.parquet"
    out.reset_index().to_parquet(path, index=False)
    print(out["source"].value_counts().to_string())
    print("\nprevalence (LLM rows, uncertain counted as 0.5):")
    print(out.loc[out["source"] == "llm", LABELS].mean().round(3).to_string())
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
