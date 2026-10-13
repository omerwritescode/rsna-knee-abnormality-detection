"""M2: fine-tune DINOv2-S end to end, with the M1 aggregator on top.

    python -m src.models.m2 --smoke                      # tiny pipeline check (~2 min)
    python -m src.models.m2 --folds 0 --name ft1         # one fold first: compare with M1 v2
    python -m src.models.m2 --name ft1                   # all 5 folds (overnight)
    python -m src.models.m2 --name ft2 --collect         # score all oof_fold*.csv of a run
    python -m src.models.m2 --name pl --train-labels pseudo_pl1.parquet   # other TRAINING targets

`--train-labels` (a file in data/labels, same columns as soft_labels.parquet, e.g. pseudo-labels)
replaces only the training targets; validation, OOF scores and metrics are always against
soft_labels.parquet, so the numbers stay comparable with every other run.

Starts from the M1 v2 checkpoint of the SAME fold (head + feature standardisation), so
fold k begins at M1's validation score and fine-tuning should only add to it.

Per training step, each study contributes 4 series x 16 slices (every 2nd slice, random
offset) read straight from data/shards; the first `--frozen-blocks` ViT blocks run
without gradients, the rest are trained (autocast: bf16 where the GPU supports it, else
fp16 with gradient scaling, e.g. on a Kaggle T4; `--amp` overrides). Augmentation per series:
small random scale/shift/rotation, brightness/contrast, and the left<->right knee mirror
(coronal/axial images flipped, sagittal slice order reversed). Whole-series dropout.
Validation: all 32 slices, normal + mirrored view averaged (same TTA as M1).

Outputs: runs/m2/<name>/fold{k}.pt (backbone + head + stats), oof_fold{k}.csv,
metrics.csv (when folds finish), config.json, train.log. When only some folds run (e.g. one
process per GPU on Kaggle), the log and metrics get a suffix (train_f3.log, metrics_f3.csv)
so parallel runs don't overwrite each other; `--collect` then scores all folds together.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import sample_study
from .folds import read_labels
from .m1 import M1, lr_lambda
from .metrics import LABELS, score

# ----------------------------------------------------------------------------- data
class StudyDataset(Dataset):
    """Reads one study shard -> uint8 [4, S, 256, 256], present [4], slice positions [S]."""

    def __init__(self, shard_dir: Path, uids, targets: np.ndarray, train: bool,
                 n_slices: int = 16, mirror: bool = False, p_mirror: float = 0.5,
                 p_series: float = 0.2, seed: int = 0):
        self.shard_dir, self.uids, self.targets = Path(shard_dir), list(uids), targets
        self.train, self.n_slices, self.mirror = train, n_slices, mirror
        self.p_mirror, self.p_series, self.seed = p_mirror, p_series, seed
        self.epoch = 0

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        with np.load(self.shard_dir / f"{self.uids[i]}.npz") as z:
            arrays = {k: z[k] for k in z.files}
        vol, present, idx = sample_study(arrays, self.train, rng, self.n_slices, self.mirror,
                                         self.p_mirror, self.p_series)
        return (torch.from_numpy(vol), torch.from_numpy(present), torch.from_numpy(idx),
                torch.from_numpy(self.targets[i].astype(np.float32)))


# ----------------------------------------------------------------------------- model
class M2(nn.Module):
    def __init__(self, head: M1, stats: dict, backbone: str, img_size: int = 224,
                 grid: int = 2, frozen_blocks: int = 8, weights: str | None = None):
        super().__init__()
        import timm
        kw = dict(pretrained=True, num_classes=0, img_size=img_size)
        if weights:
            kw["pretrained_cfg_overlay"] = dict(file=weights)
        self.backbone = timm.create_model(backbone, **kw)
        self.head = head
        self.img_size, self.grid, self.frozen_blocks = img_size, grid, frozen_blocks
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("img_mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        for k in ("f_mean", "f_std", "g_mean", "g_std"):
            self.register_buffer(k, torch.as_tensor(np.asarray(stats[k]), dtype=torch.float32))
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[frozen_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

    def stem(self, img: torch.Tensor) -> torch.Tensor:
        """uint8 [N,256,256] -> tokens after the FROZEN blocks (no gradients).
        Identical for every fold, so inference runs it once and shares it."""
        m = self.backbone
        x = img.float().div_(255.0).unsqueeze(1)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x.expand(-1, 3, -1, -1) - self.img_mean) / self.img_std
        with torch.no_grad():
            x = m.patch_embed(x)
            x = m._pos_embed(x)
            x = m.patch_drop(x)
            x = m.norm_pre(x)
            for blk in m.blocks[:self.frozen_blocks]:
                x = blk(x)
        return x

    def tail(self, x: torch.Tensor) -> torch.Tensor:
        """stem tokens -> trained blocks -> standardised [N, 768 + g*g*384] (same as M1)."""
        m = self.backbone
        for blk in m.blocks[self.frozen_blocks:]:
            x = blk(x)
        tok = m.norm(x).float()
        cls, patches = tok[:, 0], tok[:, self.n_prefix:]
        n, p, d = patches.shape
        side = int(round(p ** 0.5))
        g = patches.transpose(1, 2).reshape(n, d, side, side)
        g = F.adaptive_avg_pool2d(g, self.grid).flatten(2).transpose(1, 2)      # [N,g*g,d]
        f = (torch.cat([cls, patches.mean(1)], 1) - self.f_mean) / self.f_std
        g = (g - self.g_mean) / self.g_std
        return torch.cat([f, g.flatten(1)], 1)

    def slice_features(self, img: torch.Tensor) -> torch.Tensor:
        return self.tail(self.stem(img))

    def head_from_features(self, feats: torch.Tensor, present: torch.Tensor,
                           pos: torch.Tensor) -> torch.Tensor:
        """feats [n_present_series * S, F] in vol[present] order -> logits [B, 12]."""
        b, k = present.shape
        s = pos.shape[1]
        x = feats.new_zeros(b, k, s, feats.shape[-1])
        x[present] = feats.reshape(-1, s, feats.shape[-1])
        keep = present[:, :, None].expand(-1, -1, s)
        return self.head(x, keep, pos)

    def forward(self, vol: torch.Tensor, present: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, k, s, h, w = vol.shape
        feats = self.slice_features(vol[present].reshape(-1, h, w))        # only real series
        return self.head_from_features(feats, present, pos)


# ----------------------------------------------------------------------------- loops
def amp_dtype(choice: str, device) -> torch.dtype:
    """auto: bf16 if the GPU has it natively (RTX 30xx+), else fp16 (T4, P100)."""
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16" or device.type != "cuda":
        return torch.float16
    try:
        ok = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:                                   # older torch: no emulation flag
        ok = torch.cuda.get_device_capability()[0] >= 8
    return torch.bfloat16 if ok else torch.float16


def run_eval(model: M2, ds_plain: StudyDataset, ds_mirror: StudyDataset, device,
             workers: int, dtype: torch.dtype = torch.bfloat16) -> np.ndarray:
    model.eval()
    preds = []
    for ds in (ds_plain, ds_mirror):
        out = []
        dl = DataLoader(ds, batch_size=1, num_workers=workers, pin_memory=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype,
                                             enabled=device.type == "cuda"):
            for vol, present, pos, _ in dl:
                logit = model(vol.to(device, non_blocking=True), present.to(device),
                              pos.to(device))
                out.append(torch.sigmoid(logit.float()).cpu())
        preds.append(torch.cat(out).numpy())
    return np.mean(preds, 0)


def train_fold(k, df, Y, args, device, log, out_dir, Yt=None):
    tr = np.flatnonzero(df["fold"].to_numpy() != k)
    va = np.flatnonzero(df["fold"].to_numpy() == k)
    uids = df["StudyInstanceUID"].to_numpy()
    shard_dir = args.root / "data" / "shards"

    ck = torch.load(args.root / "runs" / "m1" / args.init / f"fold{k}.pt",
                    map_location="cpu", weights_only=False)
    cfg1 = ck["config"]
    if cfg1.get("no_grid") or cfg1.get("cache") != "cache_g2":
        raise SystemExit("M2 expects an M1 run trained with the 2x2 grid (cache_g2)")
    head = M1(cfg1["in_dim"], cfg1["d"], cfg1["layers"], cfg1["heads"], args.drop)
    head.load_state_dict(ck["state_dict"])
    stats = {kk: v.numpy() for kk, v in ck["stats"].items()}
    torch.manual_seed(args.seed + k)
    model = M2(head, stats, args.backbone, args.img_size, 2, args.frozen_blocks,
               args.weights).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("fold %d: train %d / val %d studies, %.1fM trainable params (head init from "
             "m1/%s)", k, len(tr), len(va), n_train / 1e6, args.init)

    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone")]
    hd = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone},
                             {"params": hd, "lr": args.lr_head}], weight_decay=args.wd)
    Yt = Y if Yt is None else Yt                       # training targets (pseudo-labels or Y)
    ds_tr = StudyDataset(shard_dir, uids[tr], Yt[tr], True, args.slices, seed=args.seed + k,
                         p_series=args.p_series)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True)
    # not persistent: workers are re-created each epoch so they see ds_tr.epoch (new augmentations)
    ds_va = StudyDataset(shard_dir, uids[va], Y[va], False)
    ds_vm = StudyDataset(shard_dir, uids[va], Y[va], False, mirror=True)
    steps_per_epoch = len(dl_tr) // args.accum
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: lr_lambda(st, total, max(1, steps_per_epoch // 2)))
    src = df["source"].to_numpy()
    dtype = amp_dtype(args.amp, device)
    # fp16 needs loss scaling (small gradients underflow); bf16 does not
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16 and device.type == "cuda")
    log.info("fold %d: autocast %s%s", k, str(dtype).replace("torch.", ""),
             " + GradScaler" if scaler.is_enabled() else "")

    if args.eval_start:
        p = run_eval(model, ds_va, ds_vm, device, args.workers, dtype)
        sc = score(p, Y[va], src[va])
        log.info("fold %d start (= M1 head)  val auc_llm %.4f  auc_human %.3f", k,
                 sc.loc["MACRO", "auc_llm"], sc.loc["MACRO", "auc_human"])

    for ep in range(1, args.epochs + 1):
        ds_tr.epoch = ep
        model.train()
        model.backbone.blocks[:args.frozen_blocks].eval()      # frozen part: no dropout etc.
        t0, run_loss, n_img, n_seen = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for it, (vol, present, pos, y) in enumerate(dl_tr, 1):
            vol, present = vol.to(device, non_blocking=True), present.to(device)
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                logit = model(vol, present, pos.to(device))
            loss = F.binary_cross_entropy_with_logits(logit.float(), y.to(device))
            scaler.scale(loss / args.accum).backward()
            run_loss += loss.item() * len(y)
            n_seen += len(y)
            n_img += int(present.sum()) * vol.shape[2]
            if it % args.accum == 0:
                scaler.unscale_(opt)                    # no-op when the scaler is disabled
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)                        # skipped if fp16 gradients overflowed
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if it % args.log_every == 0:
                el = time.time() - t0
                log.info("  fold %d ep %d  %d/%d  loss %.4f  %.0f img/s  epoch ETA %.1f min",
                         k, ep, it, len(dl_tr), run_loss / n_seen, n_img / el,
                         el / it * (len(dl_tr) - it) / 60)
        msg = f"fold {k} ep {ep}  loss {run_loss / max(n_seen, 1):.4f}  ({(time.time() - t0) / 60:.1f} min)"
        if ep % args.eval_every == 0 or ep == args.epochs:
            p = run_eval(model, ds_va, ds_vm, device, args.workers, dtype)
            sc = score(p, Y[va], src[va])
            msg += (f"  val auc_llm {sc.loc['MACRO', 'auc_llm']:.4f}  "
                    f"auc_human {sc.loc['MACRO', 'auc_human']:.3f}")
        log.info(msg)

    torch.save({"state_dict": model.state_dict(), "m1_config": cfg1, "config": vars_str(args),
                "stats": ck["stats"]}, out_dir / f"fold{k}.pt")
    oof = pd.DataFrame(p, columns=LABELS)
    oof.insert(0, "StudyInstanceUID", uids[va])
    oof.to_csv(out_dir / f"oof_fold{k}.csv", index=False)
    return va, p


def vars_str(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def collect(out_dir: Path, labels: pd.DataFrame, log) -> None:
    """Score every oof_fold*.csv of a run together (after folds ran in separate processes)."""
    parts = sorted(out_dir.glob("oof_fold*.csv"))
    if not parts:
        raise SystemExit(f"no oof_fold*.csv in {out_dir}")
    oof = pd.concat([pd.read_csv(p, dtype={"StudyInstanceUID": str}) for p in parts])
    df = labels[["StudyInstanceUID", "source", *LABELS]].merge(
        oof, on="StudyInstanceUID", suffixes=("", "_p"))
    sc = score(df[[f"{l}_p" for l in LABELS]].to_numpy(), df[LABELS].to_numpy(),
               df["source"].to_numpy())
    sc.to_csv(out_dir / "metrics.csv")
    folds = ",".join(p.stem.replace("oof_fold", "") for p in parts)
    with pd.option_context("display.float_format", "{:.4f}".format):
        log.info("\nOOF scores (%d studies, folds %s):\n%s", len(df), folds, sc.to_string())
    log.info("saved -> %s", out_dir / "metrics.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--name", default=None)
    ap.add_argument("--init", default="v2", help="M1 run to initialise the head from")
    ap.add_argument("--train-labels", default=None,
                    help="training targets from this file in data/labels (validation still "
                         "uses soft_labels.parquet)")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--weights", default=None, help="local DINOv2 weights file (optional)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--frozen-blocks", type=int, default=8, help="of 12; the rest are trained")
    ap.add_argument("--slices", type=int, default=16, help="slices per series in training")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=2, help="studies per forward pass")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--p-series", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-start", action="store_true",
                    help="score the untouched start (should match M1 v2 on this fold)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16"],
                    help="autocast precision; auto = bf16 if native, else fp16 + GradScaler")
    ap.add_argument("--collect", action="store_true",
                    help="only score all oof_fold*.csv in runs/m2/<name> -> metrics.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.name, args.folds, args.epochs, args.slices = args.name or "smoke", "0", 1, 8
        args.eval_every, args.log_every, args.accum = 1, 5, 2

    if args.collect and not args.name:
        raise SystemExit("--collect needs --name")
    name = args.name or time.strftime("%Y%m%d_%H%M")
    out_dir = args.root / "runs" / "m2" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_list = [int(f) for f in args.folds.split(",")]
    # a subset of folds (e.g. one process per GPU) gets its own log / metrics file
    sfx = "" if args.collect or sorted(fold_list) == [0, 1, 2, 3, 4] else \
        "_f" + "".join(str(f) for f in fold_list)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / f"train{sfx}.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)
    log = logging.getLogger("m2")
    labels = read_labels(args.root / "data" / "labels" / "soft_labels.parquet")
    if args.collect:
        collect(out_dir, labels, log)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (out_dir / f"config{sfx}.json").write_text(json.dumps(vars_str(args), indent=2))

    folds = pd.read_csv(args.root / "data" / "labels" / "folds.csv",
                        dtype={"StudyInstanceUID": str})
    df = folds.merge(labels[["StudyInstanceUID", *LABELS]], on="StudyInstanceUID")
    have = {p.stem for p in (args.root / "data" / "shards").glob("*.npz")}
    df = df[df["StudyInstanceUID"].isin(have)].reset_index(drop=True)
    if args.smoke:
        df = pd.concat([df[df["fold"] != 0].head(40), df[df["fold"] == 0].head(24)],
                       ignore_index=True)
    Y = df[LABELS].to_numpy().astype(np.float32)
    log.info("M2 %s: %d studies, device %s, torch %s", name, len(df), device, torch.__version__)
    Yt = None
    if args.train_labels:
        tl = Path(args.train_labels)
        tl = tl if tl.exists() else args.root / "data" / "labels" / tl
        pl = read_labels(tl).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        miss = ~df["StudyInstanceUID"].isin(pl.index)
        if miss.any():
            raise SystemExit(f"{tl.name}: {int(miss.sum())} training studies have no targets")
        Yt = pl.loc[df["StudyInstanceUID"], LABELS].to_numpy().astype(np.float32)
        if not np.isfinite(Yt).all() or Yt.min() < 0 or Yt.max() > 1:
            raise SystemExit(f"{tl.name}: targets must be finite and within [0, 1]")
        log.info("training targets from %s (validation still vs soft_labels.parquet): "
                 "mean |change| %.3f, per label %s", tl.name, float(np.abs(Yt - Y).mean()),
                 ", ".join(f"{l} {d:.3f}" for l, d in zip(LABELS, np.abs(Yt - Y).mean(0))))

    oof = np.full(Y.shape, np.nan)
    for k in fold_list:
        va, p = train_fold(k, df, Y, args, device, log, out_dir, Yt)
        oof[va] = p
    done = ~np.isnan(oof).any(1)
    sc = score(oof[done], Y[done], df["source"].to_numpy()[done])
    sc.to_csv(out_dir / f"metrics{sfx}.csv")
    with pd.option_context("display.float_format", "{:.4f}".format):
        log.info("\nOOF scores (%d studies, folds %s):\n%s", done.sum(), args.folds, sc.to_string())
    log.info("saved -> %s", out_dir)


if __name__ == "__main__":
    main()
