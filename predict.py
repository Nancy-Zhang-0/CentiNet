"""Predict Centiloid values for native-space amyloid PET NIfTI volumes.

    python predict.py --ckpt weights/resnet_transformer_hybrid_seed0.pt \
        scan1.nii.gz scan2.nii ... [--out predictions.csv]

Input: one or more native-space PET volumes (no MRI, no spatial normalization,
no reference-region masking). Each volume is reoriented to RAS, resampled to
the training voxel size, cropped around the foreground bounding box and
divided by the p95 of its own foreground voxels, exactly as in training.
Output: CSV with one row per file and one column per target (CL_pons, CL_WC,
CL_WCB), in Centiloid units.

Expected input: ADNI-style "Coreg, Avg, Std Img and Vox Siz, Uniform 6mm Res"
18F-florbetapir (AV45) or 18F-florbetaben volumes. The released weights were
trained on ADNI only; performance on other scanners, tracers or
reconstruction protocols has not been validated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import unstandardize  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import preprocess_volume  # noqa: E402


def load_release(ckpt_path: str | Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt or "model_name" not in ckpt:
        raise SystemExit(f"{ckpt_path} is not a release checkpoint; "
                         "create one with tools/export_checkpoint.py")
    model = build_model(ckpt["model_name"], dropout=ckpt.get("dropout", 0.2),
                        n_targets=len(ckpt["targets"]), **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nii", nargs="+", help="native-space PET NIfTI file(s)")
    ap.add_argument("--ckpt", required=True, help="release checkpoint (.pt)")
    ap.add_argument("--out", default=None, help="CSV path (default: print only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, ckpt = load_release(args.ckpt, args.device)
    targets, stats, pp = ckpt["targets"], ckpt["target_stats"], ckpt["preprocess"]
    print(f"model {ckpt['model_name']} (run {ckpt.get('source_run', '?')}, "
          f"epoch {ckpt.get('epoch', '?')}) on {args.device}")

    rows = []
    for path in args.nii:
        try:
            vol = preprocess_volume(path, **pp).astype(np.float32)
        except Exception as e:  # unreadable / empty volume
            print(f"SKIP {path}: {e}")
            continue
        x = torch.from_numpy(vol)[None, None].to(args.device)
        pred = unstandardize(model(x).float().cpu().numpy(), stats, targets)[0]
        rows.append({"file": str(path), **{t: float(v) for t, v in zip(targets, pred)}})
        print(f"{Path(path).name}: " + "  ".join(f"{t} {v:6.1f}" for t, v in zip(targets, pred)))

    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
