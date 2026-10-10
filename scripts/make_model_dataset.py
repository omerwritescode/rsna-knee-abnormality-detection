"""Collect everything the internet-off submission notebook needs into one Kaggle dataset.

    python -m src.models.blend --m2 ft1,ft2        # first: writes runs/blend.json
    python scripts/make_model_dataset.py           # packs the runs blend.json gives weight to
    python scripts/make_model_dataset.py --run v2  # M1 only (no blend file needed)

Creates kaggle/model_dataset/ with
    blend.json                                  the weights (keys = folder names below)
    m1/fold*.pt (+ config.json, metrics.csv)    the M1 run's folds, if its weight > 0
    m2_<run>/fold*.pt (+ config.json)           each M2 run with weight > 0
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


def copy_run(src_dir: Path, dst: Path, extra=("config.json", "metrics.csv")) -> int:
    folds = sorted(src_dir.glob("fold*.pt"))
    if not folds:
        raise SystemExit(f"no fold*.pt in {src_dir}")
    dst.mkdir(parents=True)
    for f in [*folds, *(src_dir / e for e in extra)]:
        if f.exists():
            shutil.copy2(f, dst / f.name)
    if len(folds) < 5:
        print(f"WARNING: only {len(folds)} folds in {src_dir}")
    return len(folds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", type=Path, default=ROOT / "runs" / "blend.json",
                    help="from src.models.blend; decides which runs are included")
    ap.add_argument("--run", default=None, help="M1 only: runs/m1/<run>, ignores --blend")
    ap.add_argument("--out", type=Path, default=ROOT / "kaggle" / "model_dataset")
    ap.add_argument("--user", default="omerraza2202")
    ap.add_argument("--slug", default="rsna-knee-models")
    ap.add_argument("--python", default="3.13", help="Kaggle's Python version (see notebook log)")
    ap.add_argument("--skip-wheels", action="store_true")
    args = ap.parse_args()

    if args.run:
        blend = {"weights": {"m1": 1.0}, "runs": {"m1": args.run}}
    else:
        if not args.blend.exists():
            raise SystemExit(f"{args.blend} not found - run python -m src.models.blend first")
        blend = json.loads(args.blend.read_text())
        if "weights" not in blend:
            raise SystemExit(f"{args.blend} is in the old format - rerun python -m src.models.blend")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for old in [*out.glob("m1"), *out.glob("m2*")]:          # models from the previous version
        shutil.rmtree(old)
    for old in out.glob("blend.json"):
        old.unlink()

    used = {k: w for k, w in blend["weights"].items() if w > 0}
    for key, w in used.items():
        run = blend["runs"][key]
        src_dir = ROOT / "runs" / ("m1" if key == "m1" else "m2") / run
        n = copy_run(src_dir, out / key)
        print(f"{key:10s} <- {src_dir.relative_to(ROOT)}  ({n} folds, weight {w})")
    dropped = [k for k, w in blend["weights"].items() if w <= 0]
    if dropped:
        print(f"weight 0, left out: {', '.join(dropped)}")
    (out / "blend.json").write_text(json.dumps({**blend, "weights": used}, indent=2))

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
