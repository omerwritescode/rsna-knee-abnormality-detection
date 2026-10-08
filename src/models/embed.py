"""DINOv2-S slice embeddings for every study shard (frozen backbone, computed once).

Run from the project root (venv active):
    python -m src.models.embed --check      # 3 studies: shapes, sanity, speed, ETA
    python -m src.models.embed              # all studies; resumable (skips finished ones)

Input : data/shards/<StudyInstanceUID>.npz   {kind: uint8 [32, 256, 256]} (kneeprep)
Output: data/embeddings/<tag>/<StudyInstanceUID>.npz, for each kind present
        (sag_fl, cor_fl, ax_fl, sag_t1):
            <kind>        float16 [32, 768]          CLS (384) + mean patch token (384)
            <kind>_grid   float16 [32, G*G, 384]     patch tokens average-pooled to G x G
        and for the coronal/axial kinds also the LEFT-RIGHT MIRRORED images:
            <kind>_m, <kind>_m_grid
        plus data/embeddings/<tag>/config.json and index.csv.

Mirroring a knee (left <-> right) in kneeprep's standard orientation:
    coronal, axial : flip the image horizontally  -> needs the extra *_m features
    sagittal       : reverse the slice order       -> free at training time
So M1 can use "mirror the whole knee" as augmentation / test-time augmentation.

Inference uses fp16 autocast (the Kaggle T4 has no bf16), images are resized
256 -> 224 (whole field of view kept, no crop), grey -> 3 channels, ImageNet norm.
The same `Embedder` class must be used in the Kaggle submission notebook.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

KINDS = ("sag_fl", "cor_fl", "ax_fl", "sag_t1")
MIRROR_KINDS = ("cor_fl", "ax_fl")       # mirrored by a horizontal image flip
N_SLICES, SIZE = 32, 256
DEFAULT_MODEL = "vit_small_patch14_dinov2.lvd142m"


# ----------------------------------------------------------------------------- data side (numpy only)
def load_shard(path: Path) -> dict[str, np.ndarray]:
    """{kind: uint8 [32, 256, 256]}; raises ValueError on a malformed shard."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for k, a in arrays.items():
        if k not in KINDS:
            raise ValueError(f"unexpected key {k!r}")
        if a.dtype != np.uint8 or a.shape != (N_SLICES, SIZE, SIZE):
            raise ValueError(f"{k}: {a.dtype} {a.shape}")
    return arrays


def plan_images(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Stack every image to embed into one uint8 [N, 256, 256] batch.

    Returns (images, segments) where segments = [(output_key, start, stop)].
    Mirrored kinds are appended as horizontally flipped copies.
    """
    blocks, segments, pos = [], [], 0
    for kind in KINDS:
        if kind not in arrays:
            continue
        variants = [(kind, arrays[kind])]
        if kind in MIRROR_KINDS:
            variants.append((f"{kind}_m", arrays[kind][:, :, ::-1]))
        for key, vol in variants:
            blocks.append(vol)
            segments.append((key, pos, pos + len(vol)))
            pos += len(vol)
    if not blocks:
        return np.zeros((0, SIZE, SIZE), np.uint8), []
    return np.ascontiguousarray(np.concatenate(blocks)), segments


def split_outputs(feat: np.ndarray, grid: np.ndarray,
                  segments: list[tuple[str, int, int]]) -> dict[str, np.ndarray]:
    out = {}
    for key, a, b in segments:
        out[key] = feat[a:b].astype(np.float16)
        out[f"{key}_grid"] = grid[a:b].astype(np.float16)
    return out


def save_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def sanity_report(outputs: dict[str, dict[str, np.ndarray]], log: logging.Logger) -> bool:
    """Checks on a few embedded studies. Returns True if everything looks right."""
    ok = True
    uids = list(outputs)
    for uid in uids:
        o = outputs[uid]
        shapes = ", ".join(f"{k}{tuple(v.shape)}" for k, v in o.items() if not k.endswith("_grid"))
        log.info("  %s: %s", uid[-8:], shapes)
        for k, v in o.items():
            if not np.isfinite(v.astype(np.float32)).all():
                log.error("  NaN/inf in %s %s", uid[-8:], k)
                ok = False
    first = outputs[uids[0]]
    kind = next(k for k in KINDS if k in first)
    v = first[kind]
    adj = cosine_rows(v[:-1], v[1:]).mean()
    log.info("  cosine, neighbouring slices (%s): %.3f   [expect high, ~0.7-0.98]", kind, adj)
    for other_uid in uids[1:]:
        common = [k for k in KINDS if k in first and k in outputs[other_uid]]
        if common:
            k = common[0]
            cross = cosine_rows(first[k], outputs[other_uid][k]).mean()
            log.info("  cosine, %s same slice index, different study: %.3f   "
                     "[expect lower than neighbouring slices]", k, cross)
            break
    for k in MIRROR_KINDS:
        if k in first:
            c = cosine_rows(first[k], first[f"{k}_m"]).mean()
            log.info("  cosine, %s vs its mirror: %.3f   [expect < 0.999]", k, c)
            if c > 0.9995:
                log.error("  mirrored features identical to originals - flip not applied?")
                ok = False
    spread = float(np.std(v.astype(np.float32)))
    if spread < 1e-3:
        log.error("  features are (almost) constant - model not loaded correctly?")
        ok = False
    return ok


# ----------------------------------------------------------------------------- model side (torch)
class Embedder:
    """Frozen DINOv2 feature extractor. Same code for training features and Kaggle inference."""

    def __init__(self, model_name: str = DEFAULT_MODEL, img_size: int = 224, grid: int = 4,
                 weights: str | None = None, device: str = "cuda", batch_size: int = 256):
        import timm
        import torch

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        kw = dict(pretrained=True, num_classes=0, img_size=img_size)
        if weights:  # local weights file (Kaggle, internet off)
            kw["pretrained_cfg_overlay"] = dict(file=weights)
        self.model = timm.create_model(model_name, **kw).eval().to(self.device)
        cfg = self.model.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.device).view(1, 3, 1, 1)
        self.img_size, self.grid, self.batch_size = img_size, grid, batch_size
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        self.config = dict(model=model_name, img_size=img_size, grid=grid,
                           mean=list(cfg["mean"]), std=list(cfg["std"]),
                           n_prefix=self.n_prefix, embed_dim=self.model.num_features,
                           timm=timm.__version__, torch=torch.__version__, amp="fp16")

    def __call__(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """uint8 [N, H, W] -> (float32 [N, 2D], float32 [N, G*G, D])."""
        torch = self.torch
        F = torch.nn.functional
        feats, grids = [], []
        with torch.inference_mode():
            for s in range(0, len(images), self.batch_size):
                x = torch.from_numpy(images[s:s + self.batch_size]).to(self.device, non_blocking=True)
                x = x.float().div_(255.0).unsqueeze(1)                       # [B,1,H,W]
                if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
                    x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                                      align_corners=False, antialias=True)
                x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.device.type == "cuda"):
                    tok = self.model.forward_features(x)                     # [B, P+L, D]
                tok = tok.float()
                cls = tok[:, 0]
                patches = tok[:, self.n_prefix:]
                b, n, d = patches.shape
                side = int(round(n ** 0.5))
                assert side * side == n, f"patch tokens {n} not a square grid"
                g = patches.transpose(1, 2).reshape(b, d, side, side)
                g = F.adaptive_avg_pool2d(g, self.grid).flatten(2).transpose(1, 2)  # [B,G*G,D]
                feats.append(torch.cat([cls, patches.mean(1)], 1).cpu().numpy())
                grids.append(g.cpu().numpy())
        return np.concatenate(feats), np.concatenate(grids)


# ----------------------------------------------------------------------------- loader (top-level for Windows spawn)
class ShardDataset:
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            images, segments = plan_images(load_shard(p))
            return p.stem, images, segments, ""
        except Exception as e:  # noqa: BLE001 - reported per study, run continues
            return p.stem, None, None, f"{type(e).__name__}: {e}"[:300]


def first_item(batch):
    return batch[0]


def iterate(dataset: ShardDataset, workers: int):
    if workers <= 0:
        for i in range(len(dataset)):
            yield dataset[i]
        return
    from torch.utils.data import DataLoader
    yield from DataLoader(dataset, batch_size=1, num_workers=workers, collate_fn=first_item,
                          prefetch_factor=4, persistent_workers=False)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--weights", default=None, help="local weights file instead of downloading")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="3 studies + sanity checks + ETA")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"dinov2s_{args.img_size}_g{args.grid}"
    shard_dir = args.root / "data" / "shards"
    out_dir = args.root / "data" / "embeddings" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "embed.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    log = logging.getLogger("embed")

    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {shard_dir}")
    todo = [p for p in shards if not (out_dir / p.name).exists()]
    if args.check:
        todo = todo[:3] if len(todo) >= 3 else shards[:3]
    elif args.limit:
        todo = todo[:args.limit]
    n_done = sum((out_dir / p.name).exists() for p in shards)
    log.info("%d shards, %d already embedded, %d to do -> %s", len(shards), n_done, len(todo), out_dir)

    emb = Embedder(args.model, args.img_size, args.grid, args.weights, batch_size=args.batch_size)
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        keys = ("model", "img_size", "grid", "mean", "std")
        if any(old.get(k) != emb.config.get(k) for k in keys):
            raise SystemExit(f"{cfg_path} was made with different settings; use another --tag")
    else:
        cfg_path.write_text(json.dumps(emb.config, indent=2))
    log.info("model %s on %s, img %d, grid %d, timm %s, torch %s", args.model, emb.device,
             args.img_size, args.grid, emb.config["timm"], emb.config["torch"])

    index_path = out_dir / "index.csv"
    new_index = not index_path.exists()
    t0, n_img, n_ok, n_err = time.time(), 0, 0, 0
    kept = {}
    with open(index_path, "a", encoding="utf-8", newline="") as idx:
        writer = csv.writer(idx)
        if new_index:
            writer.writerow(["StudyInstanceUID", "keys", "n_images", "error"])
        for i, (uid, images, segments, err) in enumerate(iterate(ShardDataset(todo), args.workers), 1):
            if not err:
                try:
                    feat, grid = emb(images)
                    out = split_outputs(feat, grid, segments)
                    save_atomic(out_dir / f"{uid}.npz", out)
                    if args.check:
                        kept[uid] = out
                    n_img += len(images)
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:300]
            if err:
                n_err += 1
                log.warning("%s: %s", uid, err)
            keys = "|".join(k for k, *_ in segments) if not err else ""
            writer.writerow([uid, keys, 0 if err else len(images), err])
            idx.flush()
            if i % 100 == 0 or i == len(todo):
                el = time.time() - t0
                log.info("%d/%d  ok=%d err=%d  %.0f img/s  %.2f s/study  ETA %.1f min",
                         i, len(todo), n_ok, n_err, n_img / el, el / i, el / i * (len(todo) - i) / 60)

    if args.check and kept:
        log.info("sanity checks:")
        good = sanity_report(kept, log)
        el = time.time() - t0
        remaining = len([p for p in shards if not (out_dir / p.name).exists()])
        log.info("throughput %.0f images/s (includes model warm-up); %d studies left, "
                 "rough full-run time %.0f min", n_img / el, remaining,
                 el / max(n_ok, 1) * remaining / 60)
        log.info("CHECK %s", "PASSED" if good else "FAILED - send the log above")
    else:
        log.info("finished: ok=%d err=%d (failed studies are retried on the next run)", n_ok, n_err)


if __name__ == "__main__":
    main()
