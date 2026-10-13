"""Occlusion sensitivity map for any registered model (causal attribution).

Each 16^3 block of the 96^3 input is zeroed in turn (6x6x6 block grid = 216
occlusions) and the change in predicted CL_WC is recorded. Unlike attention
weights, this is a causal measure: blocks whose removal moves the prediction
are blocks the model actually uses.

Signed maps are reported per amyloid-burden group:
  delta = pred(occluded) - pred(original)
Interpretation for a model that implicitly computes a cortex/reference ratio:
occluding cortex should lower the prediction (negative delta), occluding the
reference region (cerebellum/pons, inferior) should raise or destabilize it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.data import load_target_stats, unstandardize  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import seed_everything  # noqa: E402

GROUPS = [("negative (CL_WC <= 0)", -np.inf, 0.0),
          ("gray zone (0 < CL_WC <= 40)", 0.0, 40.0),
          ("positive (> 40)", 40.0, np.inf)]
GRID, BLOCK = 6, 16


@torch.no_grad()
def occlusion_deltas(model, vol: torch.Tensor, stats, targets, device,
                     wc_idx: int, batch_size: int = 27) -> np.ndarray:
    """Signed Delta CL_WC (original units) for each occluded block, [6, 6, 6]."""
    base = model(vol[None].to(device)).float().cpu().numpy()
    base_wc = unstandardize(base, stats, targets)[0, wc_idx]
    coords = [(i, j, k) for i in range(GRID) for j in range(GRID) for k in range(GRID)]
    deltas = np.zeros(len(coords), dtype=np.float32)
    for s in range(0, len(coords), batch_size):
        chunk = coords[s:s + batch_size]
        batch = vol[None].repeat(len(chunk), 1, 1, 1, 1).clone()
        for b, (i, j, k) in enumerate(chunk):
            batch[b, :, i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK,
                  k * BLOCK:(k + 1) * BLOCK] = 0.0
        pred = model(batch.to(device)).float().cpu().numpy()
        wc = unstandardize(pred, stats, targets)[:, wc_idx]
        deltas[s:s + len(chunk)] = wc - base_wc
    return deltas.reshape(GRID, GRID, GRID)


def upsample(a: np.ndarray, size: int = 96) -> np.ndarray:
    t = torch.from_numpy(a)[None, None].float()
    return F.interpolate(t, size=(size,) * 3, mode="trilinear",
                         align_corners=False).squeeze().numpy()


def overlay_signed(pet: np.ndarray, dmap: np.ndarray, title: str, out_png: Path,
                   vmax: float | None = None) -> None:
    """Signed overlay: red = occlusion raises prediction, blue = lowers.
    Pass a shared `vmax` (in CL) so group figures are directly comparable."""
    if vmax is None:
        vmax = np.percentile(np.abs(dmap), 99)
    show_thr = 0.25 * vmax
    mids = [s // 2 for s in pet.shape]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    views = [(pet[mids[0]], dmap[mids[0]], "sagittal"),
             (pet[:, mids[1]], dmap[:, mids[1]], "coronal"),
             (pet[:, :, mids[2]], dmap[:, :, mids[2]], "axial")]
    for ax, (p, d, name) in zip(axes, views):
        ax.imshow(np.rot90(p), cmap="gray", vmin=0, vmax=np.percentile(pet, 99.5))
        d = np.rot90(d).copy()
        d[np.abs(d) < show_thr] = np.nan
        im = ax.imshow(d, cmap="seismic", alpha=0.6, vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01,
                 label="Delta pred CL_WC (occluded - original)")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_per_group", type=int, default=12)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    targets = cfg.columns.targets
    seed_everything(cfg.train.seed)

    manifest = pd.read_csv(cfg.resolve_path(cfg.paths.manifest_csv))
    df = manifest[(manifest["split"] == args.split) & (manifest["qc_fov_flag"] == 0)]
    if "outlier_flag" in df.columns:
        df = df[df["outlier_flag"] == 0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model(cfg.train.model, dropout=cfg.train.dropout,
                        n_targets=len(cfg.columns.targets),
                        **cfg.train.model_kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = load_target_stats(cfg.resolve_path(cfg.paths.target_stats))
    wc_idx = targets.index(cfg.columns.primary_target)

    cache_dir = cfg.resolve_path(cfg.paths.cache_dir)
    out_dir = run_dir / "eval" / "occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.split.seed)
    summary = {}
    groups_out = []
    for name, lo, hi in GROUPS:
        grp = df[(df["CL_WC"] > lo) & (df["CL_WC"] <= hi)]
        take = grp.sample(min(args.n_per_group, len(grp)),
                          random_state=int(rng.integers(1 << 31)))
        dmaps, pets = [], []
        for _, row in take.iterrows():
            vol = np.load(cache_dir / f"{row['scan_id']}.npy").astype(np.float32)
            v = torch.from_numpy(vol)[None]
            dmaps.append(occlusion_deltas(model, v, stats, targets, device, wc_idx))
            pets.append(vol)
        dmean = np.mean(dmaps, axis=0)
        d96 = upsample(dmean)
        pet_mean = np.mean(pets, axis=0)
        tag = name.split(" ")[0]
        groups_out.append((name, tag, len(take), pet_mean, d96))
        np.save(out_dir / f"occlusion_{tag}_6.npy", dmean)
        z = np.abs(dmean).sum(axis=(0, 1))
        total = np.abs(dmean).sum()
        summary[name] = {
            "n": int(len(take)),
            "max_abs_delta_CL": float(np.abs(dmean).max()),
            "z_absmass_inferior_middle_superior": [
                round(float(z[:2].sum() / total), 3), round(float(z[2:4].sum() / total), 3),
                round(float(z[4:].sum() / total), 3)],
            "most_positive_block_ijk": [int(x) for x in
                                        np.unravel_index(dmean.argmax(), dmean.shape)],
            "most_negative_block_ijk": [int(x) for x in
                                        np.unravel_index(dmean.argmin(), dmean.shape)],
        }
    # render with one shared colorbar scale (in CL) so groups are comparable,
    # plus an equal-weight pooled "all" figure. Note: the pooled map dilutes
    # burden-dependent effects (cortical blue exists mainly in positives) —
    # the grouped figures are the primary presentation.
    vmax = max(np.percentile(np.abs(d), 99) for _, _, _, _, d in groups_out)
    for name, tag, n, pet_mean, d96 in groups_out:
        overlay_signed(pet_mean, d96,
                       f"{cfg.train.model} occlusion sensitivity — {name}, n={n}",
                       out_dir / f"occlusion_{tag}.png", vmax=vmax)
    overlay_signed(np.mean([p for _, _, _, p, _ in groups_out], axis=0),
                   np.mean([d for _, _, _, _, d in groups_out], axis=0),
                   f"{cfg.train.model} occlusion sensitivity — all groups pooled",
                   out_dir / "occlusion_all.png", vmax=vmax)

    (out_dir / "occlusion_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
