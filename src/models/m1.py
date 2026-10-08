"""M1: frozen DINOv2-S slice features -> small transformer aggregator -> 12 labels.

    python -m src.models.m1 --smoke          # 1 fold, 2 epochs, 400 studies: pipeline check
    python -m src.models.m1 --name base      # 5-fold CV, OOF scores, checkpoints

Needs: data/labels/soft_labels.parquet, data/labels/folds.csv (src.models.folds) and
the feature cache (src.models.data). Outputs go to runs/m1/<name>/:
    fold{k}.pt, oof.csv, metrics.csv, config.json, train.log

Per study the input is 4 series kinds x 32 slices = 128 tokens; each token is the
slice's CLS + mean-patch feature (768) plus, by default, its 2x2 patch grid (1536).
token = MLP(feature) + kind embedding + slice-position embedding
-> 2-layer transformer -> 12 learned label queries attend over the tokens -> 12 logits.

Inputs are standardised per channel (stats.npz in the cache dir, also saved in every
checkpoint). A masked mean of the tokens is added to each label's attention output, so
the model learns from the first epoch instead of waiting for the attention to focus.

Training: BCE on soft labels (1 / 0.5 / 0), AdamW + cosine, and augmentation:
  * mirror the knee left<->right (p=0.5): coronal/axial use the flipped-image
    features, sagittal kinds get their slice order reversed;
  * series dropout (drop whole kinds, keep >= 1) and slice dropout.
Validation/prediction average the normal and mirrored knee (TTA).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import KIND_SLOT, KIND_SLOT_MIRROR, N_SLICES, SAGITTAL, load_cache, load_stats
from .folds import read_labels
from .metrics import LABELS, score

N_KINDS = 4


# ----------------------------------------------------------------------------- model
class M1(nn.Module):
    def __init__(self, in_dim: int, d: int = 256, layers: int = 2, heads: int = 4,
                 drop: float = 0.1, n_labels: int = len(LABELS)):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(d, d))
        self.kind = nn.Parameter(torch.randn(N_KINDS, 1, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, N_SLICES, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head_w = nn.Parameter(torch.zeros(n_labels, d))
        self.head_b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """x [B, K, S, F] float; keep [B, K, S] bool (True = real token)."""
        b, k, s, _ = x.shape
        h = self.inp(x) + self.kind[None] + self.pos[None]          # [B,K,S,d]
        h = h.reshape(b, k * s, -1)
        pad = ~keep.reshape(b, k * s)
        h = self.enc(h, src_key_padding_mask=pad)
        w = keep.reshape(b, k * s, 1).to(h.dtype)
        h_mean = (h * w).sum(1) / w.sum(1).clamp(min=1.0)                # [B,d]
        q = self.queries[None].expand(b, -1, -1)
        o, _ = self.pool(q, h, h, key_padding_mask=pad)
        o = self.norm(o + h_mean[:, None])     # mean path: learns fast; attention refines
        return (o * self.head_w).sum(-1) + self.head_b                 # [B,12]


# ----------------------------------------------------------------------------- batches
class Features:
    """Holds the cache on the CPU and assembles batches (with optional mirroring)."""

    def __init__(self, cache: dict, use_grid: bool, stats: dict):
        self.stats = stats
        self.f_mean = torch.from_numpy(stats["f_mean"])
        self.f_std = torch.from_numpy(stats["f_std"])
        self.g_mean = torch.from_numpy(stats["g_mean"])
        self.g_std = torch.from_numpy(stats["g_std"])
        self.feats = torch.from_numpy(np.ascontiguousarray(cache["feats"]))
        self.grid = torch.from_numpy(np.ascontiguousarray(cache["grid"])) if use_grid else None
        self.present = torch.from_numpy(cache["present"])
        self.slot = torch.from_numpy(KIND_SLOT)
        self.slot_m = torch.from_numpy(KIND_SLOT_MIRROR)
        self.sag = torch.from_numpy(SAGITTAL)
        self.in_dim = self.feats.shape[-1] + (self.grid.shape[-2] * self.grid.shape[-1]
                                              if use_grid else 0)

    def batch(self, rows: torch.Tensor, mirror: torch.Tensor):
        """rows [B] cache rows, mirror [B] bool -> x [B,4,32,F] float32, present [B,4] bool."""
        slots = torch.where(mirror[:, None], self.slot_m[None], self.slot[None])   # [B,4]
        r = rows[:, None].expand(-1, N_KINDS)
        x = (self.feats[r, slots].float() - self.f_mean) / self.f_std
        if self.grid is not None:
            g = (self.grid[r, slots].float() - self.g_mean) / self.g_std     # [B,4,32,c,384]
            x = torch.cat([x, g.flatten(-2)], -1)
        present = self.present[r, slots]
        flip = (mirror[:, None] & self.sag[None])[:, :, None, None]
        x = torch.where(flip, x.flip(2), x)
        return x, present


def token_mask(present: torch.Tensor, train: bool, p_series: float, p_slice: float) -> torch.Tensor:
    """[B,4] kinds present -> [B,4,32] tokens kept (dropout only when training)."""
    kinds = present.clone()
    if train and p_series > 0:
        drop = (torch.rand(kinds.shape) < p_series) & kinds
        survivors = kinds & ~drop
        drop[~survivors.any(1)] = False            # never drop every series
        kinds = kinds & ~drop
    keep = kinds[:, :, None].expand(-1, -1, N_SLICES).clone()
    keep[~kinds.any(1), 2, 0] = True               # no series at all: one dummy token
    if train and p_slice > 0:
        keep &= torch.rand(keep.shape) >= p_slice
        empty = ~keep.flatten(1).any(1)
        keep[empty] = kinds[empty][:, :, None].expand(-1, -1, N_SLICES)
    return keep


@torch.no_grad()
def predict(model: M1, feats: Features, rows: torch.Tensor, device, tta: bool = True,
            bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        acc = 0
        views = (False, True) if tta else (False,)
        for m in views:
            x, present = feats.batch(r, torch.full((len(r),), m))
            keep = token_mask(present, False, 0, 0)
            acc = acc + torch.sigmoid(model(x.to(device), keep.to(device))).cpu()
        out.append(acc / len(views))
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- training
def lr_lambda(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def train_fold(k: int, df: pd.DataFrame, Y: torch.Tensor, rows: torch.Tensor,
               feats: Features, args, device, log) -> tuple[np.ndarray, M1]:
    torch.manual_seed(args.seed + k)
    tr = torch.from_numpy(np.flatnonzero(df["fold"].to_numpy() != k))
    va = torch.from_numpy(np.flatnonzero(df["fold"].to_numpy() == k))
    model = M1(feats.in_dim, args.d, args.layers, args.heads, args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: lr_lambda(st, total, steps_per_epoch))
    src = df["source"].to_numpy()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        perm = tr[torch.randperm(len(tr))]
        for s in range(0, len(perm), args.bs):
            idx = perm[s:s + args.bs]
            mirror = torch.rand(len(idx)) < args.p_mirror
            x, present = feats.batch(rows[idx], mirror)
            keep = token_mask(present, True, args.p_series, args.p_slice)
            logits = model(x.to(device), keep.to(device))
            loss = F.binary_cross_entropy_with_logits(logits, Y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
        if ep % args.eval_every == 0 or ep == args.epochs:
            p = predict(model, feats, rows[va], device, tta=True)
            sc = score(p, Y[va].numpy(), src[va.numpy()])
            log.info("fold %d ep %2d  loss %.4f  val auc_llm %.4f  auc_human %.3f  (%.1fs)",
                     k, ep, run_loss / len(tr), sc.loc["MACRO", "auc_llm"],
                     sc.loc["MACRO", "auc_human"], time.time() - t0)
    return predict(model, feats, rows[va], device, tta=True), model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--tag", default="dinov2s_224_g4")
    ap.add_argument("--cache", default="cache_g2")
    ap.add_argument("--name", default=None)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-grid", action="store_true", help="use only CLS+mean features")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--p-mirror", type=float, default=0.5)
    ap.add_argument("--p-series", type=float, default=0.2)
    ap.add_argument("--p-slice", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.folds, args.epochs, args.eval_every = "0", 2, 1
        args.name = args.name or "smoke"

    name = args.name or time.strftime("%Y%m%d_%H%M")
    out = args.root / "runs" / "m1" / name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)
    log = logging.getLogger("m1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = read_labels(args.root / "data" / "labels" / "soft_labels.parquet")
    folds = pd.read_csv(args.root / "data" / "labels" / "folds.csv", dtype={"StudyInstanceUID": str})
    cache_dir = args.root / "data" / "embeddings" / args.tag / args.cache
    cache = load_cache(cache_dir)
    stats = load_stats(cache_dir, cache)
    row_of = {u: i for i, u in enumerate(cache["uids"])}
    df = folds.merge(labels[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID")
    df["row"] = df["StudyInstanceUID"].map(row_of)
    missing = df["row"].isna().sum()
    if missing:
        log.warning("%d studies have no embeddings and are skipped", missing)
    df = df.dropna(subset=["row"]).reset_index(drop=True)
    if args.smoke:
        df = df.groupby("fold").head(80).reset_index(drop=True)

    feats = Features(cache, use_grid=not args.no_grid, stats=stats)
    Y = torch.tensor(df[LABELS].to_numpy(), dtype=torch.float32)
    rows = torch.tensor(df["row"].astype(int).to_numpy())
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(in_dim=feats.in_dim, n_studies=len(df), device=str(device),
               torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log.info("M1 %s: %d studies, in_dim %d, device %s", name, len(df), feats.in_dim, device)

    oof = pd.DataFrame(index=df.index, columns=LABELS, dtype=float)
    for k in [int(f) for f in args.folds.split(",")]:
        pred, model = train_fold(k, df, Y, rows, feats, args, device, log)
        oof.loc[df.index[df["fold"] == k], LABELS] = pred
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "stats": {k2: torch.from_numpy(v) for k2, v in stats.items()}},
                   out / f"fold{k}.pt")

    done = oof.notna().all(axis=1).to_numpy()
    res = df.loc[done, ["StudyInstanceUID", "fold", "source"]].join(oof.loc[done])
    res.to_csv(out / "oof.csv", index=False)
    sc = score(oof.loc[done].to_numpy(), Y.numpy()[done], df["source"].to_numpy()[done])
    sc.to_csv(out / "metrics.csv")
    with pd.option_context("display.float_format", "{:.4f}".format):
        log.info("\nOOF scores (%d studies):\n%s", done.sum(), sc.to_string())
    log.info("saved -> %s", out)


if __name__ == "__main__":
    main()
