"""Inference: study DICOMs -> 12 probabilities. Shared by the Kaggle submission notebook
and a local parity check.

Local check (laptop, before uploading anything):
    python -m src.models.predict --check --run v2
  re-embeds a few training shards with the same Embedder, compares the features with the
  training cache, and checks the fold ensemble gives the same predictions from both.

Kaggle (called from the submission notebook):
    run_test(data_root, model_dir, "/kaggle/working/submission.csv")
  DICOM -> kneeprep (CPU worker pool) -> DINOv2 features (GPU) -> M1 fold ensemble with
  mirror TTA -> submission.csv in sample_submission's row and column order. A study that
  fails at any step gets the training prevalence instead, so the run never crashes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .data import SLOTS, pool_grid
from .embed import Embedder, plan_images, split_outputs
from .metrics import LABELS

N_SLOTS, N_SLICES, FEAT_DIM, PATCH_DIM = len(SLOTS), 32, 768, 384
SERIES_COLS = ["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]


# ----------------------------------------------------------------------------- features
def study_tensors(arrays: dict, embedder: Embedder, g_out: int = 2):
    """kneeprep arrays -> (feats [6,32,768] f16, grid [6,32,g*g,384] f16, present [6] bool)."""
    feats = np.zeros((N_SLOTS, N_SLICES, FEAT_DIM), np.float16)
    grid = np.zeros((N_SLOTS, N_SLICES, g_out * g_out, PATCH_DIM), np.float16)
    present = np.zeros(N_SLOTS, bool)
    images, segments = plan_images(arrays)
    if not segments:
        return feats, grid, present
    f, g = embedder(images)
    out = split_outputs(f, g, segments)
    for j, slot in enumerate(SLOTS):
        if slot in out:
            feats[j] = out[slot]
            grid[j] = pool_grid(out[f"{slot}_grid"], g_out)
            present[j] = True
    return feats, grid, present


# ----------------------------------------------------------------------------- model ensemble
class Ensemble:
    """All fold checkpoints of one M1 run; averages their (mirror-TTA) probabilities."""

    def __init__(self, ckpt_paths, device=None):
        import torch

        from .m1 import M1
        self.torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models, self.cfg, self.stats = [], None, None
        for p in sorted(ckpt_paths):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            m = M1(cfg["in_dim"], cfg["d"], cfg["layers"], cfg["heads"], cfg["drop"])
            m.load_state_dict(ck["state_dict"])
            self.models.append(m.eval().to(self.device))
            if self.cfg is None:
                self.cfg = cfg
                self.stats = {k: v.numpy() for k, v in ck["stats"].items()}
        if not self.models:
            raise FileNotFoundError("no fold*.pt checkpoints given")
        self.use_grid = not self.cfg.get("no_grid", False)

    def predict(self, feats: np.ndarray, grid: np.ndarray, present: np.ndarray) -> np.ndarray:
        from .m1 import Features, predict
        torch = self.torch
        fs = Features(dict(feats=feats, grid=grid, present=present), self.use_grid, self.stats)
        rows = torch.arange(len(feats))
        preds = [predict(m, fs, rows, self.device, tta=True) for m in self.models]
        return np.mean(preds, 0)


# ----------------------------------------------------------------------------- Kaggle run
def prep_task(task):
    """Pool worker: (index, study_dir, series rows) -> (index, arrays | None, message)."""
    i, study_dir, rows = task
    try:
        from src.kneeprep import process_study
        if not rows:
            return i, None, "no rows in test_series.csv"
        if not os.path.isdir(study_dir):
            return i, None, "study folder missing"
        arrays, series_meta, _ = process_study(study_dir, rows)
        failed = sum(int(m.get("n_failed", 0) or 0) for m in series_meta)
        errs = [m["error"] for m in series_meta if m.get("error")]
        msg = f"failed slice reads: {failed}" if failed else ""
        if errs:
            msg += f" series errors: {errs[:2]}"
        return i, (arrays or None), msg
    except Exception as e:  # noqa: BLE001
        return i, None, f"{type(e).__name__}: {e}"[:300]


def run_test(data_root, model_dir, out_csv, workers: int | None = None,
             time_limit_h: float = 8.0, split: str = "test", log=print) -> "pd.DataFrame":
    import multiprocessing as mp

    import pandas as pd

    t0 = time.time()
    data_root, model_dir = Path(data_root), Path(model_dir)
    series = pd.read_csv(data_root / f"{split}_series.csv", dtype={"StudyInstanceUID": str,
                                                                   "SeriesInstanceUID": str})
    sample = pd.read_csv(data_root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    groups = {s: g[SERIES_COLS].to_dict("records") for s, g in series.groupby("StudyInstanceUID")}
    ids = sample["StudyInstanceUID"].tolist()
    priors = json.loads(next(model_dir.rglob("priors.json")).read_text())
    log(f"{len(ids)} studies to predict | {len(groups)} studies in {split}_series.csv")

    # Start the CPU pool BEFORE touching CUDA (fork after CUDA init is unsafe).
    workers = workers or os.cpu_count() or 2
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    pool = ctx.Pool(workers)
    tasks = [(i, str(data_root / f"{split}_series" / s), groups.get(s, [])) for i, s in enumerate(ids)]
    results = pool.imap_unordered(prep_task, tasks)

    weights = next(model_dir.rglob("model.safetensors"), None)
    if weights is None:
        log("WARNING: no model.safetensors in the model dataset - timm will try to download")
    embedder = Embedder(weights=str(weights) if weights else None)
    ens = Ensemble(list(model_dir.rglob("fold*.pt")), embedder.device)
    g_out = int(ens.cfg.get("cache", "cache_g2").split("_g")[-1])
    log(f"models ready: {len(ens.models)} folds, grid {g_out}, device {embedder.device} "
        f"({time.time() - t0:.0f}s)")

    n = len(ids)
    feats = np.zeros((n, N_SLOTS, N_SLICES, FEAT_DIM), np.float16)
    grid = np.zeros((n, N_SLOTS, N_SLICES, g_out * g_out, PATCH_DIM), np.float16)
    present = np.zeros((n, N_SLOTS), bool)
    problems, timed_out = {}, False
    for k, (i, arrays, msg) in enumerate(results, 1):
        if msg:
            problems[ids[i]] = msg
        if arrays:
            try:
                feats[i], grid[i], present[i] = study_tensors(arrays, embedder, g_out)
            except Exception as e:  # noqa: BLE001
                problems[ids[i]] = f"embed {type(e).__name__}: {e}"[:300]
        el = time.time() - t0
        if k % 50 == 0 or k == n:
            log(f"  {k}/{n} studies  {el / k:.2f} s/study  elapsed {el / 60:.1f} min  "
                f"ETA {el / k * (n - k) / 60:.1f} min")
        if el > time_limit_h * 3600:
            log(f"TIME LIMIT reached after {k} studies - the rest get prior values")
            timed_out = True
            break
    pool.terminate()

    ok = present.any(1)
    pred = np.tile(np.array([priors[lab] for lab in LABELS], float), (n, 1))
    if ok.any():
        pred[ok] = ens.predict(feats[ok], grid[ok], present[ok])
    sub = pd.DataFrame(pred, columns=LABELS)
    sub.insert(0, "StudyInstanceUID", ids)
    sub = sub[sample.columns.tolist()]
    sub.to_csv(out_csv, index=False)

    log(f"\npredicted {ok.sum()}/{n} studies from images, {n - ok.sum()} got priors"
        f"{' (time limit)' if timed_out else ''} | total {(time.time() - t0) / 60:.1f} min")
    log("series kinds present: " + str(dict(zip(SLOTS, present.sum(0).tolist()))))
    for sid, msg in list(problems.items())[:20]:
        log(f"  note {sid[-12:]}: {msg}")
    if len(problems) > 20:
        log(f"  ... {len(problems) - 20} more notes")
    log(f"saved -> {out_csv}")
    return sub


# ----------------------------------------------------------------------------- local check
def local_check(root: Path, run: str, tag: str, cache_name: str, n: int) -> None:
    from .data import load_cache
    run_dir = root / "runs" / "m1" / run
    cache = load_cache(root / "data" / "embeddings" / tag / cache_name, in_memory=False)
    uids = cache["uids"][:n]
    embedder = Embedder()
    ens = Ensemble(list(run_dir.glob("fold*.pt")), embedder.device)
    g_out = int(cache_name.split("_g")[-1])

    t0 = time.time()
    F = np.zeros((n, N_SLOTS, N_SLICES, FEAT_DIM), np.float16)
    G = np.zeros((n, N_SLOTS, N_SLICES, g_out * g_out, PATCH_DIM), np.float16)
    P = np.zeros((n, N_SLOTS), bool)
    for i, u in enumerate(uids):
        with np.load(root / "data" / "shards" / f"{u}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        F[i], G[i], P[i] = study_tensors(arrays, embedder, g_out)
    print(f"re-embedded {n} studies in {time.time() - t0:.1f}s")

    cF = np.asarray(cache["feats"][:n]).astype(np.float32)
    cP = np.asarray(cache["present"][:n])
    a, b = F.astype(np.float32)[P], cF[cP]
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    print(f"present slots identical: {np.array_equal(P, cP)}")
    print(f"feature cosine vs training cache: min {cos.min():.5f}, mean {cos.mean():.5f}   "
          f"[expect > 0.999]")
    p_new = ens.predict(F, G, P)
    p_old = ens.predict(np.asarray(cache["feats"][:n]), np.asarray(cache["grid"][:n]), cP)
    diff = np.abs(p_new - p_old).max()
    print(f"ensemble ({len(ens.models)} folds) max |prediction difference|: {diff:.5f}   "
          f"[expect < 0.01]")
    ok = np.array_equal(P, cP) and cos.min() > 0.999 and diff < 0.01
    print("PARITY CHECK", "PASSED" if ok else "FAILED - send this output")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--tag", default="dinov2s_224_g4")
    ap.add_argument("--cache", default="cache_g2")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    if not args.check:
        raise SystemExit("use --check locally; run_test() is called from the Kaggle notebook")
    local_check(args.root, args.run, args.tag, args.cache, args.n)


if __name__ == "__main__":
    main()
