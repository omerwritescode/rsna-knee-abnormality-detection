"""Collect everything Kaggle M2 training (and scoring) needs into one private Kaggle dataset.

    python scripts/make_train_inputs.py

Creates kaggle/train_inputs/ with
    every data/labels/*.parquet and *.csv   soft_labels.parquet, folds.csv, split.csv, later
                                            pseudo-label files
    m1/fold*.pt                             M1 run the M2 head starts from (--m1-run, default v2)
    dinov2/model.safetensors                DINOv2-S weights (local HF cache)
    priors.json                             per-label prevalence (submission fallback)
    wheels/                                 copied from kaggle/model_dataset/wheels if present
    dataset-metadata.json
and prints the kaggle commands (with --dir-mode zip, otherwise subfolders are skipped).

The shards come from the outputs of the notebooks rsna-knee-prep-chunk0 / chunk1.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
DINO_REPO = "timm/vit_small_patch14_dinov2.lvd142m"


def dinov2_weights() -> Path:
    old = ROOT / "kaggle" / "model_dataset" / "dinov2" / "model.safetensors"
    try:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(DINO_REPO, "model.safetensors", local_files_only=True))
    except Exception:  # noqa: BLE001
        if old.exists():
            return old
        raise SystemExit("DINOv2 weights not found in the HF cache or kaggle/model_dataset/dinov2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "train_inputs")
    ap.add_argument("--m1-run", default="v2")
    ap.add_argument("--user", default="omerraza2202")
    ap.add_argument("--slug", default="rsna-knee-train-inputs")
    args = ap.parse_args()

    lab_dir = ROOT / "data" / "labels"
    files = sorted([*lab_dir.glob("*.parquet"), *lab_dir.glob("*.csv")])
    names = {f.name for f in files}
    for need in ("soft_labels.parquet", "folds.csv"):
        if need not in names:
            raise SystemExit(f"missing data/labels/{need}")
    m1 = sorted((ROOT / "runs" / "m1" / args.m1_run).glob("fold*.pt"))
    if len(m1) != 5:
        raise SystemExit(f"expected 5 fold*.pt in runs/m1/{args.m1_run}, found {len(m1)}")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "m1").mkdir(parents=True)
    for f in files:
        shutil.copy2(f, out / f.name)
        print(f"  {f.name:28s} {f.stat().st_size / 1e3:8.0f} kB")
    for f in m1:
        shutil.copy2(f, out / "m1" / f.name)
    print(f"  m1/  {len(m1)} folds from runs/m1/{args.m1_run}")

    (out / "dinov2").mkdir()
    src = dinov2_weights()
    shutil.copy2(src, out / "dinov2" / "model.safetensors")
    print(f"  dinov2/model.safetensors   from {src}")

    import pandas as pd
    lab = pd.read_parquet(lab_dir / "soft_labels.parquet")
    lab = lab[lab["source"] == "llm"]
    priors = {k: round(float(lab[k].mean()), 4) for k in LABELS}
    (out / "priors.json").write_text(json.dumps(priors, indent=2))
    print("  priors.json")

    wheels = ROOT / "kaggle" / "model_dataset" / "wheels"
    if wheels.exists() and any(wheels.glob("*.whl")):
        shutil.copytree(wheels, out / "wheels")
        print(f"  wheels/  {len(list(wheels.glob('*.whl')))} files")
    else:
        print("  (no kaggle/model_dataset/wheels - the scoring notebook will need rsna-knee-models for them)")

    meta = {"title": args.slug, "id": f"{args.user}/{args.slug}",
            "licenses": [{"name": "CC0-1.0"}]}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\ndataset folder ready: {out} ({size:.0f} MB)")
    print("first upload (creates a PRIVATE dataset):")
    print(f'  kaggle datasets create -p "{out}" --dir-mode zip')
    print("later updates:")
    print(f'  kaggle datasets version -p "{out}" -m "<what changed>" --dir-mode zip')


if __name__ == "__main__":
    main()
