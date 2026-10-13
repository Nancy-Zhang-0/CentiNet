"""Config loading: CLI override parsing, numeric coercion (B2)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config


def test_exponent_form_overrides_become_floats():
    # YAML 1.1 parses '5e-5' as a string; the loader must coerce it
    cfg = load_config(overrides=["train.lr=5e-5", "train.weight_decay=1e-2",
                                 "augment.noise_std=3e-2"])
    assert isinstance(cfg.train.lr, float) and cfg.train.lr == pytest.approx(5e-5)
    assert cfg.train.weight_decay == pytest.approx(0.01)
    assert cfg.augment.noise_std == pytest.approx(0.03)


def test_non_numeric_overrides_untouched():
    cfg = load_config(overrides=["train.model=vit3d_small", "train.lr=null",
                                 "train.amp=false",
                                 "columns.targets=[CL_WC]"])
    assert cfg.train.model == "vit3d_small"
    assert cfg.train.lr is None
    assert cfg.train.amp is False
    assert cfg.columns.targets == ["CL_WC"]


def test_dotted_model_kwargs_override():
    cfg = load_config(overrides=["train.model_kwargs.norm=gn",
                                 "train.model_kwargs.drop_path=1e-1"])
    assert cfg.train.model_kwargs["norm"] == "gn"
    assert cfg.train.model_kwargs["drop_path"] == pytest.approx(0.1)
