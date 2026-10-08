"""Pack the per-study DINOv2 embeddings into a few arrays for fast training.

    python -m src.models.data            # builds data/embeddings/<tag>/cache_g2/ once

Cache layout (all rows in the order of uids.txt):
    feats.npy    float16 [N, 6, 32, 768]        CLS + mean patch per slice
    grid.npy     float16 [N, 6, 32, g*g, 384]   patch grid pooled 4x4 -> g x g (g=2 default)
    present.npy  bool    [N, 6]                 which slots exist for the study
Slots: SLOTS below. Mirrored slots (cor_fl_m, ax_fl_m) hold the left-right flipped
images; sagittal kinds are mirrored by reversing the slice order instead.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

SLOTS = ("sag_fl", "cor_fl", "cor_fl_m", "ax_fl", "ax_fl_m", "sag_t1")
KINDS = ("sag_fl", "cor_fl", "ax_fl", "sag_t1")
# slot index of each kind, normal and mirrored
KIND_SLOT = np.array([0, 1, 3, 5])
KIND_SLOT_MIRROR = np.array([0, 2, 4, 5])
SAGITTAL = np.array([True, False, False, True])      # mirrored by reversing slice order
N_SLICES, FEAT_DIM, PATCH_DIM = 32, 768, 384


def pool_grid(grid: np.ndarray, g_out: int) -> np.ndarray:
    """[S, G*G, D] (row-major G x G) -> [S, g_out*g_out, D] by block averaging."""
    s, n, d = grid.shape
    g_in = int(round(n ** 0.5))
    if g_out == g_in:
        return grid
    f = g_in // g_out
    assert f * g_out == g_in, f"cannot pool {g_in}x{g_in} to {g_out}x{g_out}"
    x = grid.astype(np.float32).reshape(s, g_out, f, g_out, f, d).mean((2, 4))
    return x.reshape(s, g_out * g_out, d)


def build_cache(emb_dir: Path, g_out: int = 2) -> Path:
    files = sorted(p for p in emb_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no embeddings in {emb_dir}")
    out = emb_dir / f"cache_g{g_out}"
    out.mkdir(exist_ok=True)
    n = len(files)
    feats = np.lib.format.open_memmap(out / "feats.tmp.npy", "w+", np.float16,
                                      (n, len(SLOTS), N_SLICES, FEAT_DIM))
    grid = np.lib.format.open_memmap(out / "grid.tmp.npy", "w+", np.float16,
                                     (n, len(SLOTS), N_SLICES, g_out * g_out, PATCH_DIM))
    present = np.zeros((n, len(SLOTS)), bool)
    t0 = time.time()
    for i, f in enumerate(files):
        with np.load(f) as z:
            for j, slot in enumerate(SLOTS):
                if slot in z.files:
                    feats[i, j] = z[slot]
                    grid[i, j] = pool_grid(z[f"{slot}_grid"], g_out)
                    present[i, j] = True
                else:
                    feats[i, j] = 0
                    grid[i, j] = 0
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}  {time.time() - t0:.0f}s")
    feats.flush()
    grid.flush()
    del feats, grid
    (out / "feats.tmp.npy").replace(out / "feats.npy")
    (out / "grid.tmp.npy").replace(out / "grid.npy")
    np.save(out / "present.npy", present)
    (out / "uids.txt").write_text("\n".join(f.stem for f in files))
    print(f"cache: {n} studies -> {out}")
    print("slots present:", dict(zip(SLOTS, present.sum(0).tolist())))
    return out


def channel_stats(cache: dict, chunk: int = 64) -> dict[str, np.ndarray]:
    """Per-channel mean / std over every present slice (no labels involved).

    feats: 768 channels; grid: 384 channels shared by all grid cells. DINOv2 has a few
    channels with very large values; standardising each channel stops them dominating.
    """
    feats, grid, present = cache["feats"], cache["grid"], cache["present"]
    sums = {k: np.zeros(d, np.float64) for k, d in (("f", feats.shape[-1]), ("g", grid.shape[-1]))}
    sq = {k: np.zeros_like(v) for k, v in sums.items()}
    n = {"f": 0, "g": 0}
    for s in range(0, len(feats), chunk):
        m = present[s:s + chunk]
        f = feats[s:s + chunk][m].astype(np.float64).reshape(-1, feats.shape[-1])
        g = grid[s:s + chunk][m].astype(np.float64).reshape(-1, grid.shape[-1])
        for key, a in (("f", f), ("g", g)):
            sums[key] += a.sum(0)
            sq[key] += (a * a).sum(0)
            n[key] += len(a)
    out = {}
    for key in ("f", "g"):
        mu = sums[key] / n[key]
        sd = np.sqrt(np.maximum(sq[key] / n[key] - mu * mu, 1e-12))
        out[f"{key}_mean"] = mu.astype(np.float32)
        out[f"{key}_std"] = np.maximum(sd, 1e-3).astype(np.float32)
    return out


def load_stats(cache_dir: Path, cache: dict) -> dict[str, np.ndarray]:
    path = cache_dir / "stats.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    stats = channel_stats(cache)
    np.savez(path, **stats)
    return stats


def load_cache(cache_dir: Path, in_memory: bool = True) -> dict:
    mode = None if in_memory else "r"
    return dict(
        uids=cache_dir.joinpath("uids.txt").read_text().split("\n"),
        feats=np.load(cache_dir / "feats.npy", mmap_mode=mode),
        grid=np.load(cache_dir / "grid.npy", mmap_mode=mode),
        present=np.load(cache_dir / "present.npy"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--tag", default="dinov2s_224_g4")
    ap.add_argument("--grid", type=int, default=2)
    args = ap.parse_args()
    build_cache(args.root / "data" / "embeddings" / args.tag, args.grid)


if __name__ == "__main__":
    main()
