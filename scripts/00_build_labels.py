"""Build the PET file list and label table from the processed ADNI tree.

Scans `paths.pet_root` (see configs/default.yaml) for AV45 (tracer FBP) and FBB scans.
For each scan it reads processed/AMY_{X}_Centiloid.csv (columns ID, CG, PONS,
WC, WCB) into CL_pons / CL_WC / CL_WCB and records the native-space PET NIfTI
processed/AMY_{X}_PET.nii (the ADNI "Coreg, Avg, Std Img and Vox Siz, Uniform
6mm Res" series converted to NIfTI; no spatial normalization).

Outputs:
  data/pet_files.csv  scan_id, pet_path                (only scans with a file)
  data/labels.csv     scan_id, subject_id, scan_date, tracer, CL_*, site, sex, diagnosis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    pet_root = Path(cfg.paths.pet_root)
    tracer_map = cfg.cohort.tracers  # folder prefix -> tracer label

    # optional demographics: subject-level sex / baseline diagnosis (3-group)
    demo = {}
    group_csv = Path(cfg.paths.group_csv)
    if cfg.paths.group_csv and group_csv.is_file():
        gdf = pd.read_csv(group_csv).drop_duplicates("Subject")
        g3 = {"CN": "CN", "SMC": "CN", "EMCI": "MCI", "MCI": "MCI", "LMCI": "MCI", "AD": "AD"}
        for _, r in gdf.iterrows():
            demo[r["Subject"]] = {"sex": r.get("Sex"), "age_baseline": r.get("Age"),
                                  "diagnosis": g3.get(r.get("Group"), "Other")}

    label_rows, file_rows = [], []
    subjects = sorted(d.name for d in pet_root.iterdir() if d.is_dir() and "_S_" in d.name)
    for subj in subjects:
        for folder in sorted((pet_root / subj).iterdir()):
            prefix = folder.name.split("_Coreg", 1)[0]
            if prefix not in tracer_map or not folder.is_dir():
                continue  # skips AV45-Early, tau tracers, and T1 folders
            tracer = tracer_map[prefix]
            for scan_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
                date = scan_dir.name[:10]
                scan_id = f"{subj}_{tracer}_{date}"
                proc = scan_dir / "processed"
                pet_path = proc / f"AMY_{prefix}_PET.nii"
                cl_csv = proc / f"AMY_{prefix}_Centiloid.csv"
                cl = {"CL_pons": None, "CL_WC": None, "CL_WCB": None}
                if cl_csv.exists():
                    row = pd.read_csv(cl_csv).iloc[0]
                    cl = {"CL_pons": row["PONS"], "CL_WC": row["WC"], "CL_WCB": row["WCB"]}
                d = demo.get(subj, {})
                label_rows.append({
                    "scan_id": scan_id, "subject_id": subj, "scan_date": date,
                    "tracer": tracer, **cl, "site": subj.split("_")[0],
                    "sex": d.get("sex"), "age_baseline": d.get("age_baseline"),
                    "diagnosis": d.get("diagnosis"),
                })
                if pet_path.exists():
                    file_rows.append({"scan_id": scan_id, "pet_path": str(pet_path)})

    labels = pd.DataFrame(label_rows)
    files = pd.DataFrame(file_rows)
    assert labels["scan_id"].is_unique, "scan_id collision in label table"
    assert files["scan_id"].is_unique, "scan_id collision in PET file list"

    out_labels = cfg.resolve_path(cfg.paths.labels_csv)
    out_files = cfg.resolve_path(cfg.paths.pet_files_csv)
    out_labels.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_labels, index=False)
    files.to_csv(out_files, index=False)
    print(f"labels: {len(labels)} scans, {labels['subject_id'].nunique()} subjects "
          f"-> {out_labels}")
    print(f"pet files found: {len(files)} -> {out_files}")
    print(labels.groupby("tracer").size().to_string())


if __name__ == "__main__":
    main()
