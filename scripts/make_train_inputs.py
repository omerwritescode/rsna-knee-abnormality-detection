"""Collect the small files Kaggle M2 training needs into one private Kaggle dataset.

    python scripts/make_train_inputs.py

Creates kaggle/train_inputs/ with every data/labels/*.parquet and *.csv
(soft_labels.parquet, folds.csv, split.csv, and later any pseudo-label files) plus
dataset-metadata.json, and prints the kaggle commands to upload it.

Everything else Kaggle training needs is already on Kaggle:
  shards      outputs of the notebooks rsna-knee-prep-chunk0 / rsna-knee-prep-chunk1
  M1 + DINOv2 the rsna-knee-models dataset (m1/fold*.pt, dinov2/model.safetensors)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "train_inputs")
    ap.add_argument("--user", default="omerraza2202")
    ap.add_argument("--slug", default="rsna-knee-train-inputs")
    args = ap.parse_args()

    lab_dir = ROOT / "data" / "labels"
    files = sorted([*lab_dir.glob("*.parquet"), *lab_dir.glob("*.csv")])
    names = {f.name for f in files}
    for need in ("soft_labels.parquet", "folds.csv"):
        if need not in names:
            raise SystemExit(f"missing data/labels/{need}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        if old.is_file():
            old.unlink()
    for f in files:
        shutil.copy2(f, out / f.name)
        print(f"  {f.name:28s} {f.stat().st_size / 1e3:8.0f} kB")

    meta = {"title": args.slug, "id": f"{args.user}/{args.slug}",
            "licenses": [{"name": "CC0-1.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndataset folder ready: {out} ({len(files)} files)")
    print("first upload (creates a PRIVATE dataset):")
    print(f'  kaggle datasets create -p "{out}"')
    print("later updates (e.g. new label files):")
    print(f'  kaggle datasets version -p "{out}" -m "<what changed>"')


if __name__ == "__main__":
    main()
