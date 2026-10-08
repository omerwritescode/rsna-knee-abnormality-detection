"""Collect everything the internet-off submission notebook needs into one Kaggle dataset.

    python scripts/make_model_dataset.py --run v2                 # M1 only
    python scripts/make_model_dataset.py --run v2 --m2-run ft1    # M1 + M2 (+ blend.json)

Creates kaggle/model_dataset/ with
    m1/fold*.pt (+ config.json, metrics.csv)   the M1 run's fold checkpoints
    m2/fold*.pt (+ blend.json)                 the M2 run's fold checkpoints (optional)
    dinov2/model.safetensors                   DINOv2-S weights (from the local HF cache)
    priors.json                                per-label prevalence, used if a study fails
    wheels/                                    pydicom + JPEG/JPEG2000 decoders for Linux
    dataset-metadata.json                      for the Kaggle CLI
and prints the kaggle commands to upload it (private dataset).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
DINO_REPO = "timm/vit_small_patch14_dinov2.lvd142m"
WHEEL_PKGS = ["pydicom>=3", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"]


def dinov2_weights() -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(DINO_REPO, "model.safetensors", local_files_only=True))


def download_wheels(dest: Path, py: str) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-q", "-d", str(dest),
           "--only-binary=:all:", "--implementation", "cp", "--python-version", py,
           "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_17_x86_64",
           "--platform", "manylinux2014_x86_64", "--platform", "any", *WHEEL_PKGS]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("WARNING: wheel download failed:\n", r.stderr[-1500:])
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2", help="runs/m1/<run>")
    ap.add_argument("--m2-run", default=None, help="runs/m2/<run> (optional)")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "model_dataset")
    ap.add_argument("--user", default="omerraza2202")
    ap.add_argument("--slug", default="rsna-knee-models")
    ap.add_argument("--python", default="3.13", help="Kaggle's Python version (see notebook log)")
    ap.add_argument("--skip-wheels", action="store_true")
    args = ap.parse_args()

    run_dir = ROOT / "runs" / "m1" / args.run
    folds = sorted(run_dir.glob("fold*.pt"))
    if not folds:
        raise SystemExit(f"no fold*.pt in {run_dir}")
    out = args.out
    (out / "m1").mkdir(parents=True, exist_ok=True)
    for old in (out / "m1").glob("*"):
        old.unlink()
    for f in [*folds, run_dir / "config.json", run_dir / "metrics.csv"]:
        if f.exists():
            shutil.copy2(f, out / "m1" / f.name)
    print(f"M1 run '{args.run}': {len(folds)} folds copied")

    m2_dir = out / "m2"
    if m2_dir.exists():
        for old in m2_dir.glob("*"):
            old.unlink()
    if args.m2_run:
        run2 = ROOT / "runs" / "m2" / args.m2_run
        folds2 = sorted(run2.glob("fold*.pt"))
        if not folds2:
            raise SystemExit(f"no fold*.pt in {run2}")
        m2_dir.mkdir(exist_ok=True)
        for f in [*folds2, run2 / "config.json", run2 / "blend.json"]:
            if f.exists():
                shutil.copy2(f, m2_dir / f.name)
        blend = json.loads((run2 / "blend.json").read_text()) if (run2 / "blend.json").exists() \
            else {"m1": 0.3, "m2": 0.7}
        if not (run2 / "blend.json").exists():
            (m2_dir / "blend.json").write_text(json.dumps(blend, indent=2))
            print("WARNING: no blend.json from src.models.blend - using M1 0.3 / M2 0.7")
        print(f"M2 run '{args.m2_run}': {len(folds2)} folds copied, blend {blend}")
        if len(folds2) < 5:
            print(f"WARNING: only {len(folds2)} M2 folds")

    (out / "dinov2").mkdir(exist_ok=True)
    src = dinov2_weights()
    shutil.copy2(src, out / "dinov2" / "model.safetensors")
    print(f"DINOv2 weights copied from {src}")

    import pandas as pd
    lab = pd.read_parquet(ROOT / "data" / "labels" / "soft_labels.parquet")
    lab = lab[lab["source"] == "llm"]
    priors = {k: round(float(lab[k].mean()), 4) for k in LABELS}
    (out / "priors.json").write_text(json.dumps(priors, indent=2))
    print("priors:", priors)

    if not args.skip_wheels:
        ok = download_wheels(out / "wheels", args.python)
        n = len(list((out / "wheels").glob("*.whl")))
        print(f"wheels for Python {args.python}: {n} files" + ("" if ok else " (INCOMPLETE)"))

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
