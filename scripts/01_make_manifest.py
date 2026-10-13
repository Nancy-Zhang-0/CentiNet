"""Step 1: manifest, QC for inferior FOV truncation, subject-level split.

1. Join PET file list with the label table on scan_id; report and drop scans
   missing a file or any of the three CL values (saved to dropped_scans.csv).
2. QC: inferior_coverage_mm per scan; lowest 5% get qc_fov_flag = 1 and are
   excluded from the preliminary split (kept in the manifest).
3. Subject-level 70/10/20 split, seed 42, stratified on baseline CL_WC bin
   and tracer set. Asserts no subject appears in two splits.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402
from src.preprocessing import inferior_coverage_mm  # noqa: E402
from src.split import make_subject_split, assert_split_integrity  # noqa: E402

_CFG = None


def _qc_one(args: tuple[str, str]) -> tuple[str, float]:
    scan_id, pet_path = args
    try:
        cov = inferior_coverage_mm(
            pet_path, voxel_size=_CFG["voxel_size"],
            fg_threshold_frac=_CFG["fg_frac"], min_slice_voxels=_CFG["min_vox"])
    except Exception as e:
        print(f"QC failed for {scan_id}: {e}")
        cov = float("nan")
    return scan_id, cov


def main() -> None:
    global _CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--reuse_qc", action="store_true",
                    help="reuse inferior_coverage_mm from an existing manifest.csv")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    targets = cfg.columns.targets

    labels = pd.read_csv(cfg.resolve_path(cfg.paths.labels_csv))
    files = pd.read_csv(cfg.resolve_path(cfg.paths.pet_files_csv))
    df = labels.merge(files, on="scan_id", how="left")

    # -- drop scans missing a PET file or any valid CL value ------------------
    # non-finite (NaN/inf) Centiloid values are failed quantifications
    missing_file = df["pet_path"].isna()
    missing_cl = ~np.isfinite(df[targets].to_numpy(dtype=float)).all(axis=1)
    dropped = df[missing_file | missing_cl].copy()
    dropped["drop_reason"] = np.where(missing_file[missing_file | missing_cl], "missing_pet_file",
                                      "missing_or_nonfinite_cl")
    dropped.loc[missing_file & missing_cl, "drop_reason"] = "missing_pet_file_and_cl"
    df = df[~(missing_file | missing_cl)].reset_index(drop=True)
    dropped_path = cfg.resolve_path(cfg.paths.dropped_csv)
    dropped.to_csv(dropped_path, index=False)
    print(f"kept {len(df)} scans; dropped {len(dropped)} "
          f"({int(missing_file.sum())} missing file, {int(missing_cl.sum())} missing CL) "
          f"-> {dropped_path}")

    # -- QC: inferior field-of-view coverage ---------------------------------
    _CFG = {"voxel_size": cfg.preprocess.voxel_size, "fg_frac": cfg.qc.fg_threshold_frac,
            "min_vox": cfg.qc.min_slice_voxels}
    manifest_path = cfg.resolve_path(cfg.paths.manifest_csv)
    results: dict[str, float] = {}
    if args.reuse_qc and manifest_path.exists():
        prev = pd.read_csv(manifest_path)
        results = dict(zip(prev["scan_id"], prev["inferior_coverage_mm"]))
        print(f"reusing QC coverage for {len(results)} scans from {manifest_path}")
    todo = [(s, p) for s, p in zip(df["scan_id"], df["pet_path"]) if s not in results]
    if todo:
        n_workers = args.n_workers or cfg.preprocess.n_workers
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            results.update(dict(ex.map(_qc_one, todo, chunksize=8)))
    df["inferior_coverage_mm"] = df["scan_id"].map(results)
    thr = df["inferior_coverage_mm"].quantile(cfg.qc.fov_flag_quantile)
    df["qc_fov_flag"] = ((df["inferior_coverage_mm"] <= thr)
                         | df["inferior_coverage_mm"].isna()).astype(int)
    print(f"QC: coverage 5th percentile = {thr:.1f} mm; "
          f"flagged {int(df['qc_fov_flag'].sum())}/{len(df)} scans")

    # -- supervisor-approved outlier exclusion (anti-join on ID + SCANDATE) ---
    df["outlier_flag"] = 0
    df["outlier_tier"] = ""
    outlier_csv = Path(cfg.paths.outlier_csv) if cfg.paths.outlier_csv else None
    if outlier_csv and outlier_csv.exists():
        fl = pd.read_csv(outlier_csv)[["ID", "SCANDATE", "Tier"]].rename(
            columns={"ID": "subject_id", "SCANDATE": "scan_date", "Tier": "_tier"})
        # a scan may be flagged under several tiers; keep one row per scan so the
        # left-merge cannot duplicate manifest rows
        fl = fl.drop_duplicates(["subject_id", "scan_date"])
        df = df.merge(fl, on=["subject_id", "scan_date"], how="left")
        assert df["scan_id"].is_unique, "outlier merge duplicated manifest rows"
        df["outlier_flag"] = df["_tier"].notna().astype(int)
        df["outlier_tier"] = df["_tier"].fillna("")
        df = df.drop(columns="_tier")
        matched = int(df["outlier_flag"].sum())
        print(f"outlier list: {len(fl)} flagged scans, {matched} matched in manifest; "
              f"tiers: {df.loc[df.outlier_flag == 1, 'outlier_tier'].value_counts().to_dict()}")
    else:
        print("no outlier list applied")

    # -- subject-level stratified split (eligible scans only) ----------------
    eligible = df[(df["qc_fov_flag"] == 0) & (df["outlier_flag"] == 0)]
    assignment = make_subject_split(
        eligible, seed=cfg.split.seed, ratios=tuple(cfg.split.ratios),
        cl_wc_bins=list(cfg.split.cl_wc_bins), cl_wc_col=cfg.columns.primary_target)
    df["split"] = df["subject_id"].map(assignment).fillna("excluded")
    assert_split_integrity(df)

    manifest_path = cfg.resolve_path(cfg.paths.manifest_csv)
    df.to_csv(manifest_path, index=False)
    print(f"manifest -> {manifest_path}")

    # -- split summary -------------------------------------------------------
    lines = ["# Split summary", "",
             f"- seed: {cfg.split.seed}, ratios: {cfg.split.ratios} (by subject)",
             f"- QC flag threshold: inferior_coverage_mm <= {thr:.1f} "
             f"(lowest {cfg.qc.fov_flag_quantile:.0%}), flagged scans excluded from split",
             ""]
    clean = (df["qc_fov_flag"] == 0) & (df["outlier_flag"] == 0)
    for split in ["train", "val", "test", "excluded"]:
        in_split = df["split"] == split
        sub = df[in_split & clean] if split != "excluded" else df[in_split]
        n_qc = int((in_split & (df["qc_fov_flag"] == 1)).sum())
        n_out = int((in_split & (df["outlier_flag"] == 1)).sum())
        lines.append(f"## {split}")
        lines.append(f"- subjects: {sub['subject_id'].nunique()}, scans: {len(sub)} "
                     f"(held out: {n_qc} QC-flagged, {n_out} outlier-flagged)")
        if len(sub):
            tr = sub.groupby("tracer").size().to_dict()
            lines.append(f"- tracer: {tr}")
            cl = sub[cfg.columns.primary_target]
            lines.append(f"- CL_WC: mean {cl.mean():.1f}, sd {cl.std():.1f}, "
                         f"median {cl.median():.1f}, range [{cl.min():.1f}, {cl.max():.1f}]")
            bins = pd.cut(cl, bins=list(cfg.split.cl_wc_bins))
            lines.append(f"- CL_WC bins: {bins.value_counts().sort_index().to_dict()}")
        lines.append("")
    summary_path = cfg.resolve_path(cfg.paths.split_summary)
    summary_path.write_text("\n".join(str(x) for x in lines))
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
