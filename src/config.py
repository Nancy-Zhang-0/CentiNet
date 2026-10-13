"""Single config dataclass loaded from YAML with dotted CLI overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsCfg:
    pet_root: str = ""
    group_csv: str = ""
    outlier_csv: str = ""
    pet_files_csv: str = "data/pet_files.csv"
    labels_csv: str = "data/labels.csv"
    manifest_csv: str = "data/manifest.csv"
    dropped_csv: str = "data/dropped_scans.csv"
    split_summary: str = "data/split_summary.md"
    cache_dir: str = "data/cache"
    qc_png_dir: str = "data/qc_png"
    target_stats: str = "data/target_stats.json"
    runs_dir: str = "runs"


@dataclass
class ColumnsCfg:
    subject_id: str = "subject_id"
    scan_id: str = "scan_id"
    scan_date: str = "scan_date"
    tracer: str = "tracer"
    pet_path: str = "pet_path"
    targets: list = field(default_factory=lambda: ["CL_pons", "CL_WC", "CL_WCB"])
    primary_target: str = "CL_WC"
    site: str = "site"


@dataclass
class CohortCfg:
    tracers: dict = field(default_factory=lambda: {"AV45": "FBP", "FBB": "FBB"})


@dataclass
class QCCfg:
    fg_threshold_frac: float = 0.10
    min_slice_voxels: int = 10
    fov_flag_quantile: float = 0.05


@dataclass
class SplitCfg:
    seed: int = 42
    ratios: list = field(default_factory=lambda: [0.70, 0.10, 0.20])
    cl_wc_bins: list = field(default_factory=lambda: [float("-inf"), 0, 20, 40, 100, float("inf")])


@dataclass
class PreprocessCfg:
    voxel_size: float = 2.0
    crop_size: int = 96
    norm_percentile: float = 95.0
    fg_threshold_frac: float = 0.10
    n_qc_png: int = 20
    n_workers: int = 12


@dataclass
class AugmentCfg:
    rotate_deg: float = 15.0
    translate_vox: float = 8.0
    scale_range: list = field(default_factory=lambda: [0.9, 1.1])
    flip_prob: float = 0.5
    intensity_scale: list = field(default_factory=lambda: [0.5, 2.0])
    noise_std: float = 0.02


@dataclass
class TrainCfg:
    model: str = "resnet3d_18"
    seed: int = 0
    batch_size: int = 8
    epochs: int = 100
    lr: float | None = None
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 15
    huber_delta: float = 1.0
    dropout: float = 0.2
    amp: bool = True
    num_workers: int = 8
    pretrained_path: str | None = None
    model_kwargs: dict = field(default_factory=dict)  # e.g. norm, pool, drop_path
    wandb: bool = False
    wandb_project: str = "deep-amyloid"


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    columns: ColumnsCfg = field(default_factory=ColumnsCfg)
    cohort: CohortCfg = field(default_factory=CohortCfg)
    qc: QCCfg = field(default_factory=QCCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def resolve_path(self, p: str) -> Path:
        """Resolve a config path relative to the project root."""
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def to_dict(self) -> dict:
        return asdict(self)

    def dump_yaml(self, out_path: str | Path) -> None:
        with open(out_path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


_SECTIONS = {
    "paths": PathsCfg, "columns": ColumnsCfg, "cohort": CohortCfg, "qc": QCCfg,
    "split": SplitCfg, "preprocess": PreprocessCfg, "augment": AugmentCfg, "train": TrainCfg,
}


def _coerce_numeric(value: Any) -> Any:
    """YAML 1.1 reads '5e-5' as a string ('5.0e-5' is required for a float).
    Convert float-parseable strings so exponent-form CLI overrides behave."""
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _apply_overrides(d: dict, overrides: list[str]) -> dict:
    """Apply 'section.key=value' overrides; values parsed with yaml.safe_load."""
    d = copy.deepcopy(d)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key.path=value, got: {ov}")
        key, _, raw = ov.partition("=")
        value = _coerce_numeric(yaml.safe_load(raw))
        node: Any = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return d


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge `overlay` into `base` (overlay wins)."""
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(yaml_path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    use_default = yaml_path is None
    yaml_path = Path(yaml_path) if yaml_path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    # machine-specific values live in git-ignored configs/local.yaml, overlaid
    # on the tracked default (only when no explicit --config was given)
    local_path = PROJECT_ROOT / "configs" / "local.yaml"
    if use_default and local_path.exists():
        with open(local_path) as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    raw = _apply_overrides(raw, overrides or [])
    cfg = Config()
    for section, cls in _SECTIONS.items():
        if section in raw:
            known = {k: v for k, v in raw[section].items() if k in cls.__dataclass_fields__}
            unknown = set(raw[section]) - set(known)
            if unknown:
                raise ValueError(f"Unknown keys in config section '{section}': {sorted(unknown)}")
            setattr(cfg, section, cls(**known))
    return cfg
