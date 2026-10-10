"""Inference: study DICOMs -> 12 probabilities. Shared by the Kaggle submission notebook
and a local parity check.

Local checks (laptop, before uploading anything):
    python -m src.models.predict --check --run v2        # M1 feature parity
    python -m src.models.predict --check-m2 ft1          # M2 inference vs training OOF
  re-embeds a few training shards with the same Embedder, compares the features with the
  training cache, and checks the fold ensemble gives the same predictions from both.

Kaggle (called from the submission notebook):
    run_test(data_root, model_dir, "/kaggle/working/submission.csv")
  DICOM -> kneeprep (CPU worker pool) -> M1 (frozen DINOv2 features + fold ensemble) and
  one fold ensemble per M2 run (folders m2/ or m2_<run>/), all with mirror TTA -> weighted
  average by blend.json -> submission.csv in sample_submission's row and column order.
  A study that fails at any step gets the training prevalence, so the run never crashes.
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


class M2Ensemble:
    """All fold checkpoints of one M2 run. The frozen DINOv2 blocks are identical in every
    fold, so they run once per image and only the trained blocks + head run per fold."""

    def __init__(self, ckpt_paths, weights: str | None = None, device=None):
        import torch

        from .m1 import M1
        from .m2 import M2
        self.torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        for p in sorted(ckpt_paths):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            cfg, cfg1 = ck["config"], ck["m1_config"]
            head = M1(cfg1["in_dim"], cfg1["d"], cfg1["layers"], cfg1["heads"], cfg["drop"])
            stats = {k: v.numpy() for k, v in ck["stats"].items()}
            m = M2(head, stats, cfg["backbone"], cfg["img_size"], 2, cfg["frozen_blocks"], weights)
            m.load_state_dict(ck["state_dict"])
            self.models.append(m.eval().to(self.device))
        if not self.models:
            raise FileNotFoundError("no M2 fold*.pt checkpoints given")
        self.shared = all(self._same_frozen(self.models[0], m) for m in self.models[1:])

    def _same_frozen(self, a, b) -> bool:
        if a.frozen_blocks != b.frozen_blocks:
            return False
        trained = tuple(f"backbone.blocks.{i}." for i in range(a.frozen_blocks, len(a.backbone.blocks)))
        sa, sb = a.state_dict(), b.state_dict()
        keys = [k for k in sa if k.startswith("backbone.") and not k.startswith(trained)
                and not k.startswith("backbone.norm.")]
        return all(self.torch.equal(sa[k], sb[k]) for k in keys)

    def predict_arrays(self, arrays: dict, share: bool | None = None) -> np.ndarray | None:
        """kneeprep arrays -> [12] probabilities (folds x {normal, mirrored} averaged)."""
        from .data import sample_study
        torch, dev = self.torch, self.device
        share = self.shared if share is None else share
        out = []
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16,
                                             enabled=dev.type == "cuda"):
            for mirror in (False, True):
                vol, present, idx = sample_study(arrays, False, mirror=mirror)
                if not present.any():
                    return None
                pr = torch.from_numpy(present).to(dev)
                imgs = torch.from_numpy(vol).to(dev)[pr].reshape(-1, vol.shape[-2], vol.shape[-1])
                pos = torch.from_numpy(idx)[None].to(dev)
                x = self.models[0].stem(imgs) if share else None
                for m in self.models:
                    feats = m.tail(x if share else m.stem(imgs))
                    logit = m.head_from_features(feats, pr[None], pos)
                    out.append(torch.sigmoid(logit.float())[0].cpu().numpy())
        return np.mean(out, 0)


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


def blend_weights(model_dir: Path, groups: list[str], log=print) -> dict[str, float]:
    """Weight per model folder ('m1', 'm2', 'm2_ft1', ...) from blend.json.
    New format: {"weights": {folder: w}}; old format: {"m1": w, "m2": w}."""
    f = next(model_dir.rglob("blend.json"), None)
    raw = json.loads(f.read_text()) if f else {}
    w = raw.get("weights", {k: raw[k] for k in ("m1", "m2") if k in raw})
    if not w:
        log("WARNING: no blend.json - equal weights for every model folder")
        w = {g: 1.0 for g in groups}
    missing = [k for k, v in w.items() if v > 0 and k not in groups]
    if missing:
        log(f"WARNING: blend.json gives weight to {missing}, which are not in the dataset")
    return {g: float(w.get(g, 0.0)) for g in groups}


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
    weights = str(weights) if weights else None
    paths: dict[str, list] = {}                     # model folder -> fold checkpoints
    for p in model_dir.rglob("fold*.pt"):
        if p.parent.name == "m1" or p.parent.name.startswith("m2"):
            paths.setdefault(p.parent.name, []).append(p)
    w = blend_weights(model_dir, sorted(paths), log)
    use = {g: ps for g, ps in sorted(paths.items()) if w[g] > 0}
    if not use:
        raise FileNotFoundError(f"no usable checkpoints: {({g: len(v) for g, v in paths.items()})}, "
                                f"weights {w}")
    use_m1 = "m1" in use
    m2_groups = [g for g in use if g != "m1"]
    embedder = ens = None
    g_out = 2
    if use_m1:
        embedder = Embedder(weights=weights)
        ens = Ensemble(use["m1"], embedder.device)
        g_out = int(ens.cfg.get("cache", "cache_g2").split("_g")[-1])
    m2ens = {g: M2Ensemble(use[g], weights) for g in m2_groups}
    for g in use:
        extra = f", frozen stem shared by its folds: {m2ens[g].shared}" if g in m2ens else ""
        log(f"  {g}: {len(use[g])} folds, weight {w[g]}{extra}")
    log(f"models ready ({time.time() - t0:.0f}s)")

    n = len(ids)
    feats = np.zeros((n, N_SLOTS, N_SLICES, FEAT_DIM), np.float16)
    grid = np.zeros((n, N_SLOTS, N_SLICES, g_out * g_out, PATCH_DIM), np.float16)
    present = np.zeros((n, N_SLOTS), bool)
    pred = {g: np.full((n, len(LABELS)), np.nan) for g in use}
    problems, timed_out = {}, False
    for k, (i, arrays, msg) in enumerate(results, 1):
        if msg:
            problems[ids[i]] = msg
        if arrays and use_m1:
            try:
                feats[i], grid[i], present[i] = study_tensors(arrays, embedder, g_out)
            except Exception as e:  # noqa: BLE001
                problems[ids[i]] = f"M1 {type(e).__name__}: {e}"[:300]
        for g in (m2_groups if arrays else []):
            try:
                p2 = m2ens[g].predict_arrays(arrays)
                if p2 is not None:
                    pred[g][i] = p2
            except Exception as e:  # noqa: BLE001
                problems[ids[i]] = f"{g} {type(e).__name__}: {e}"[:300]
        el = time.time() - t0
        if k % 50 == 0 or k == n:
            log(f"  {k}/{n} studies  {el / k:.2f} s/study  elapsed {el / 60:.1f} min  "
                f"ETA {el / k * (n - k) / 60:.1f} min")
        if el > time_limit_h * 3600:
            log(f"TIME LIMIT reached after {k} studies - the rest get prior values")
            timed_out = True
            break
    pool.terminate()

    ok1 = present.any(1)
    if use_m1 and ok1.any():
        pred["m1"][ok1] = ens.predict(feats[ok1], grid[ok1], present[ok1])
    # weighted average over the models that produced a prediction for each study
    num = np.zeros((n, len(LABELS)))
    den = np.zeros(n)
    for g, p_ in pred.items():
        valid = ~np.isnan(p_).any(1)
        num[valid] += w[g] * p_[valid]
        den[valid] += w[g]
    ok = den > 0
    final = np.tile(np.array([priors[lab] for lab in LABELS], float), (n, 1))
    final[ok] = num[ok] / den[ok, None]
    sub = pd.DataFrame(final, columns=LABELS)
    sub.insert(0, "StudyInstanceUID", ids)
    sub = sub[sample.columns.tolist()]
    sub.to_csv(out_csv, index=False)

    log(f"\npredicted {ok.sum()}/{n} studies from images, {n - ok.sum()} got priors"
        f"{' (time limit)' if timed_out else ''} | total {(time.time() - t0) / 60:.1f} min")
    if use_m1:
        log("M1 series kinds present: " + str(dict(zip(SLOTS, present.sum(0).tolist()))))
    for g in m2_groups:
        log(f"{g} predicted {int((~np.isnan(pred[g]).any(1)).sum())} studies")
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


def local_check_m2(root: Path, run: str, n: int) -> None:
    """Re-predict the first n validation studies of fold 0 with the inference path and
    compare with the training-time out-of-fold predictions."""
    import pandas as pd
    run_dir = root / "runs" / "m2" / run
    oof = pd.read_csv(run_dir / "oof_fold0.csv", dtype={"StudyInstanceUID": str}).head(n)
    t0 = time.time()
    ens = M2Ensemble([run_dir / "fold0.pt"])
    print(f"loaded fold 0 in {time.time() - t0:.1f}s on {ens.device}")
    t0 = time.time()
    preds = []
    for u in oof["StudyInstanceUID"]:
        with np.load(root / "data" / "shards" / f"{u}.npz") as z:
            preds.append(ens.predict_arrays({k: z[k] for k in z.files}))
    preds = np.array(preds)
    diff = np.abs(preds - oof[LABELS].to_numpy()).max()
    print(f"{n} studies in {time.time() - t0:.1f}s ({(time.time() - t0) / n:.2f} s/study, 1 fold)")
    print(f"max |inference - training OOF| = {diff:.4f}   [expect < 0.02: fp16 vs bf16]")
    folds = sorted(run_dir.glob("fold*.pt"))
    shared_ok = True
    if len(folds) > 1:
        ens2 = M2Ensemble(folds[:2])
        with np.load(root / "data" / "shards" / f"{oof['StudyInstanceUID'][0]}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        a, b = ens2.predict_arrays(arrays, share=True), ens2.predict_arrays(arrays, share=False)
        shared_ok = ens2.shared and np.abs(a - b).max() < 1e-3
        print(f"frozen stem identical across folds: {ens2.shared} | shared vs separate "
              f"max diff {np.abs(a - b).max():.6f}")
    print("M2 CHECK", "PASSED" if diff < 0.02 and shared_ok else "FAILED - send this output")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", default="v2")
    ap.add_argument("--tag", default="dinov2s_224_g4")
    ap.add_argument("--cache", default="cache_g2")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--check-m2", default=None, metavar="RUN", help="check an M2 run, e.g. ft1")
    args = ap.parse_args()
    if args.check_m2:
        local_check_m2(args.root, args.check_m2, args.n)
    elif args.check:
        local_check(args.root, args.run, args.tag, args.cache, args.n)
    else:
        raise SystemExit("use --check or --check-m2 locally; run_test() runs in the Kaggle notebook")


if __name__ == "__main__":
    main()
