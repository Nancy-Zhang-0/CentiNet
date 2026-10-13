"""Grad-CAM maps from the CNN feature maps (not transformer attention).

Standard Grad-CAM w.r.t. the predicted CL_WC output, computed on the last
convolutional feature map:
  - cnn_transformer_hybrid: CNN stem output [384, 6, 6, 6]
  - resnet3d_10/18/34:      layer4 output   [512, 3, 3, 3]
Channel weights = spatial average of d(pred CL_WC)/d(activation); map =
ReLU(sum_k w_k * A_k), normalized per scan, group-averaged over amyloid-burden
groups, upsampled to 96^3 and overlaid on the mean PET.

A signed variant (no ReLU) is saved alongside, since for a ratio-like task the
negative direction (reference region) is also informative.
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
from src.models import build_model  # noqa: E402
from src.utils import seed_everything  # noqa: E402

GROUPS = [("negative (CL_WC <= 0)", -np.inf, 0.0),
          ("gray zone (0 < CL_WC <= 40)", 0.0, 40.0),
          ("positive (> 40)", 40.0, np.inf)]

TARGET_LAYER = {
    "cnn_transformer_hybrid": lambda m: m.stem,
    "resnet_transformer_hybrid": lambda m: m.proj,
    "resnet3d_10": lambda m: m.layer4,
    "resnet3d_18": lambda m: m.layer4,
    "resnet3d_34": lambda m: m.layer4,
}


def gradcam(model, layer, vol: torch.Tensor, device, wc_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (relu_map, signed_map) on the feature-map grid for one scan."""
    acts, grads = {}, {}

    def fwd_hook(_m, _i, out):
        acts["a"] = out

    def bwd_hook(_m, _gi, gout):
        grads["g"] = gout[0]

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        pred = model(vol[None].to(device))
        pred[0, wc_idx].backward()
    finally:
        h1.remove()
        h2.remove()
    a = acts["a"][0]                      # [C, d, d, d]
    g = grads["g"][0]
    w = g.mean(dim=(1, 2, 3), keepdim=True)
    signed = (w * a).sum(dim=0)
    relu = torch.relu(signed)
    return relu.detach().cpu().numpy(), signed.detach().cpu().numpy()


def upsample(a: np.ndarray, size: int = 96) -> np.ndarray:
    t = torch.from_numpy(a)[None, None].float()
    return F.interpolate(t, size=(size,) * 3, mode="trilinear",
                         align_corners=False).squeeze().numpy()


def overlay(pet, amap, title, out_png, signed=False, vmax=None):
    mids = [s // 2 for s in pet.shape]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    views = [(pet[mids[0]], amap[mids[0]], "sagittal"),
             (pet[:, mids[1]], amap[:, mids[1]], "coronal"),
             (pet[:, :, mids[2]], amap[:, :, mids[2]], "axial")]
    # vmax may be passed in so all group figures share one colorbar scale
    if signed:
        if vmax is None:
            vmax = np.percentile(np.abs(amap), 99)
        kw = dict(cmap="seismic", vmin=-vmax, vmax=vmax)
        thr = 0.25 * vmax
    else:
        if vmax is None:
            vmax = amap.max()
        kw = dict(cmap="jet", vmin=0.3 * vmax, vmax=vmax)
        thr = 0.3 * vmax
    for ax, (p, m, name) in zip(axes, views):
        ax.imshow(np.rot90(p), cmap="gray", vmin=0, vmax=np.percentile(pet, 99.5))
        m = np.rot90(m).copy()
        m[np.abs(m) < thr] = np.nan
        im = ax.imshow(m, alpha=0.55, **kw)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="Grad-CAM")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_per_group", type=int, default=16)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if cfg.train.model not in TARGET_LAYER:
        raise SystemExit(f"Grad-CAM target layer not defined for {cfg.train.model}; "
                         f"supported: {sorted(TARGET_LAYER)}")
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
    layer = TARGET_LAYER[cfg.train.model](model)
    wc_idx = cfg.columns.targets.index(cfg.columns.primary_target)

    cache_dir = cfg.resolve_path(cfg.paths.cache_dir)
    out_dir = run_dir / "eval" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.split.seed)
    groups_out = []
    for name, lo, hi in GROUPS:
        grp = df[(df["CL_WC"] > lo) & (df["CL_WC"] <= hi)]
        take = grp.sample(min(args.n_per_group, len(grp)),
                          random_state=int(rng.integers(1 << 31)))
        relus, signeds, pets = [], [], []
        for _, row in take.iterrows():
            vol = np.load(cache_dir / f"{row['scan_id']}.npy").astype(np.float32)
            v = torch.from_numpy(vol)[None]
            r, s = gradcam(model, layer, v, device, wc_idx)
            if r.max() > 0:
                r = r / r.max()
            smax = np.abs(s).max()
            if smax > 0:
                s = s / smax
            relus.append(r)
            signeds.append(s)
            pets.append(vol)
        pet_mean = np.mean(pets, axis=0)
        tag = name.split(" ")[0]
        r96 = upsample(np.mean(relus, axis=0))
        s96 = upsample(np.mean(signeds, axis=0))
        groups_out.append((name, tag, len(take), pet_mean, r96, s96))
        np.save(out_dir / f"gradcam_{tag}_grid.npy", np.mean(signeds, axis=0))

    # shared colorbar scale across the three group figures
    vmax_relu = max(r.max() for _, _, _, _, r, _ in groups_out)
    vmax_signed = max(np.percentile(np.abs(s), 99) for _, _, _, _, _, s in groups_out)
    for name, tag, n, pet_mean, r96, s96 in groups_out:
        overlay(pet_mean, r96, f"{cfg.train.model} Grad-CAM (CNN features) — {name}, "
                f"n={n}", out_dir / f"gradcam_{tag}.png", vmax=vmax_relu)
        overlay(pet_mean, s96, f"{cfg.train.model} signed Grad-CAM — {name}, "
                f"n={n}", out_dir / f"gradcam_signed_{tag}.png", signed=True,
                vmax=vmax_signed)
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
