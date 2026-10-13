"""CLS-attention interpretability map for cnn_transformer_hybrid.

Extracts the last encoder layer's CLS -> patch attention (averaged over heads),
reshapes it to the 6x6x6 token grid, upsamples to 96^3, and overlays it on the
preprocessed PET. Scans are grouped by amyloid burden (CL_WC < 0, 0-40, > 40);
per-group mean maps are saved, plus a quantitative attention center-of-mass /
mass-distribution summary.

Sanity-check logic: if high attention sits on the cerebellum + cortical
composite, the model has rediscovered the SUVr definition; attention outside
the head or on FOV edges would suggest shortcut learning.
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


@torch.no_grad()
def cls_attention(model, vol: torch.Tensor) -> np.ndarray:
    """CLS -> patch attention of the LAST encoder layer, head-averaged.

    vol: [B, 1, 96, 96, 96]. Returns [B, 6, 6, 6].
    Reimplements the (norm_first) layer forward to expose attention weights,
    which PyTorch's fused TransformerEncoderLayer path does not return.
    """
    f = model.stem(vol)
    tokens = f.flatten(2).transpose(1, 2) + model.pos_embed
    x = torch.cat([model.cls_token.expand(vol.shape[0], -1, -1), tokens], dim=1)
    # nn.TransformerEncoder (drop_path == 0) vs nn.Sequential of PreNormBlock
    # (drop_path > 0): both are pre-norm, but attribute names differ.
    layers = list(getattr(model.encoder, "layers", model.encoder))
    for layer in layers[:-1]:
        x = layer(x)
    last = layers[-1]
    attn_mod = getattr(last, "self_attn", None) or last.attn
    sa_in = last.norm1(x)  # norm_first architecture
    _, attn = attn_mod(sa_in, sa_in, sa_in, need_weights=True,
                       average_attn_weights=True)  # [B, L, L]
    w = attn[:, 0, 1:]                       # CLS query -> 216 patch keys
    w = w / w.sum(dim=1, keepdim=True)       # renormalize after dropping CLS->CLS
    return w.reshape(-1, 6, 6, 6).cpu().numpy()


def upsample(att: np.ndarray, size: int = 96) -> np.ndarray:
    t = torch.from_numpy(att).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)
    return up.squeeze().numpy()


def overlay_figure(pet: np.ndarray, att: np.ndarray, title: str, out_png: Path) -> None:
    """3 orthogonal mid-slices; PET in gray, attention as thresholded heat overlay."""
    thr = np.percentile(att, 70)
    mids = [s // 2 for s in pet.shape]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    views = [(pet[mids[0]], att[mids[0]], "sagittal"),
             (pet[:, mids[1]], att[:, mids[1]], "coronal"),
             (pet[:, :, mids[2]], att[:, :, mids[2]], "axial")]
    for ax, (p, a, name) in zip(axes, views):
        ax.imshow(np.rot90(p), cmap="gray", vmin=0, vmax=np.percentile(pet, 99.5))
        a = np.rot90(a).copy()
        a[a < thr] = np.nan
        im = ax.imshow(a, cmap="jet", alpha=0.55,
                       vmin=thr, vmax=att.max())
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="CLS attention")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def mass_summary(att96: np.ndarray) -> dict:
    """Center of mass (voxel) and attention mass by z-thirds (z: inferior->superior)."""
    total = att96.sum()
    grids = np.meshgrid(*[np.arange(s) for s in att96.shape], indexing="ij")
    com = [float((g * att96).sum() / total) for g in grids]
    z = att96.sum(axis=(0, 1))
    thirds = [float(z[:32].sum() / total), float(z[32:64].sum() / total),
              float(z[64:].sum() / total)]
    return {"center_of_mass_xyz_vox": [round(c, 1) for c in com],
            "z_mass_inferior_middle_superior": [round(t, 3) for t in thirds]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_per_group", type=int, default=16)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    assert cfg.train.model in ("cnn_transformer_hybrid", "resnet_transformer_hybrid"), \
        "CLS attention maps are defined for the CNN+transformer hybrids only"
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

    cache_dir = cfg.resolve_path(cfg.paths.cache_dir)
    out_dir = run_dir / "eval" / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.split.seed)
    summary = {}
    all_att, all_pet = [], []
    for name, lo, hi in GROUPS:
        grp = df[(df["CL_WC"] > lo) & (df["CL_WC"] <= hi)]
        take = grp.sample(min(args.n_per_group, len(grp)),
                          random_state=int(rng.integers(1 << 31)))
        atts, pets = [], []
        for _, row in take.iterrows():
            vol = np.load(cache_dir / f"{row['scan_id']}.npy").astype(np.float32)
            v = torch.from_numpy(vol)[None, None].to(device)
            atts.append(cls_attention(model, v)[0])
            pets.append(vol)
        att_mean = upsample(np.mean(atts, axis=0))
        pet_mean = np.mean(pets, axis=0)
        all_att.append(att_mean)
        all_pet.append(pet_mean)
        tag = name.split(" ")[0]
        overlay_figure(pet_mean, att_mean,
                       f"{cfg.train.model} CLS attention (head-avg, last layer) — "
                       f"{name}, n={len(take)}", out_dir / f"attention_{tag}.png")
        summary[name] = {"n": int(len(take)), **mass_summary(att_mean)}
        np.save(out_dir / f"attention_{tag}_96.npy", att_mean.astype(np.float32))

    overlay_figure(np.mean(all_pet, axis=0), np.mean(all_att, axis=0),
                   f"{cfg.train.model} CLS attention — all groups pooled",
                   out_dir / "attention_all.png")
    (out_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
