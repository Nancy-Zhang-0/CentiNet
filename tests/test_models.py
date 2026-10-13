"""Every registered model maps [B, 1, 96, 96, 96] -> [B, 3]."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import MODEL_REGISTRY, build_model

EXPECTED = {"resnet3d_10", "resnet3d_18", "resnet3d_34",
            "cnn_transformer_hybrid", "resnet_transformer_hybrid",
            "cnn_conv_control", "vit3d_small", "swinunetr_encoder"}


def test_registry_complete():
    assert EXPECTED <= set(MODEL_REGISTRY)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_output_shape(name):
    torch.manual_seed(0)
    model = build_model(name, dropout=0.2)
    model.eval()
    x = torch.randn(2, 1, 96, 96, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("no_such_model")


@pytest.mark.parametrize("kwargs", [
    {"norm": "gn"}, {"norm": "in"},
    {"pool": "cls+mean"}, {"drop_path": 0.1},
    {"norm": "gn", "pool": "cls+mean", "drop_path": 0.1},
])
def test_resnet_hybrid_variants(kwargs):
    torch.manual_seed(0)
    model = build_model("resnet_transformer_hybrid", dropout=0.2, **kwargs)
    x = torch.randn(2, 1, 96, 96, 96)
    model.train()
    y = model(x)                      # train mode exercises DropPath
    assert y.shape == (2, 3)
    model.eval()
    with torch.no_grad():
        assert model(x).shape == (2, 3)


@pytest.mark.parametrize("norm", ["gn", "in"])
def test_resnet3d_norm_variants(norm):
    torch.manual_seed(0)
    model = build_model("resnet3d_10", dropout=0.2, norm=norm)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 1, 96, 96, 96))
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("norm", ["gn", "in"])
def test_conv_control_norm_variants(norm):
    torch.manual_seed(0)
    model = build_model("cnn_conv_control", dropout=0.2, norm=norm)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(2, 1, 96, 96, 96))
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()
