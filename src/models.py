"""Model registry. Every model maps [B, 1, 96, 96, 96] -> [B, 3].

Shared design: encoder -> feature vector -> Dropout(p) -> Linear(feat_dim, 3).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

MODEL_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls
    return deco


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def load_pretrained_flexible(model: nn.Module, ckpt_path: str, submodule_prefix: str = "") -> str:
    """Best-effort partial load: strip 'module.'/'state_dict', match by name+shape."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "net"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    src = {k.removeprefix("module."): v for k, v in ckpt.items()}
    if submodule_prefix:
        src = {f"{submodule_prefix}{k}": v for k, v in src.items()}
    own = model.state_dict()
    matched = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    msg = f"Loaded {len(matched)}/{len(own)} tensors from {ckpt_path}"
    print(msg)
    return msg


class RegressionHead(nn.Module):
    def __init__(self, feat_dim: int, n_targets: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, n_targets)

    def forward(self, x):
        return self.fc(self.dropout(x))


# ---------------------------------------------------------------- 3D ResNet --

def make_norm(norm: str, ch: int) -> nn.Module:
    """Normalization layer factory: 'bn' | 'gn' (8 groups) | 'in' (affine)."""
    if norm == "bn":
        return nn.BatchNorm3d(ch)
    if norm == "gn":
        return nn.GroupNorm(8, ch)
    if norm == "in":
        return nn.InstanceNorm3d(ch, affine=True)
    raise ValueError(f"Unknown norm '{norm}' (expected bn|gn|in)")


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class PreNormBlock(nn.Module):
    """Pre-norm transformer block with stochastic depth (used when drop_path > 0;
    matches nn.TransformerEncoderLayer(norm_first=True, GELU, 4x MLP) otherwise)."""

    def __init__(self, dim: int, heads: int, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.dp1(self.attn(h, h, h, need_weights=False)[0])
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, ch: int, stride: int = 1, downsample: nn.Module | None = None,
                 norm: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = make_norm(norm, ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = make_norm(norm, ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """MedicalNet-style 3D ResNet (BasicBlock) regression backbone."""

    def __init__(self, layers: list[int], n_targets: int = 3, dropout: float = 0.2,
                 norm: str = "bn", pretrained_path: str | None = None):
        super().__init__()
        self._norm_kind = norm
        self.in_ch = 64
        self.conv1 = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = make_norm(norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = RegressionHead(512, n_targets, dropout)
        if pretrained_path:
            load_pretrained_flexible(self, pretrained_path)

    def _make_layer(self, ch: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_ch, ch, 1, stride=stride, bias=False),
                make_norm(self._norm_kind, ch))
        layers = [BasicBlock3d(self.in_ch, ch, stride, downsample, norm=self._norm_kind)]
        self.in_ch = ch
        layers += [BasicBlock3d(ch, ch, norm=self._norm_kind) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(self.pool(x).flatten(1))


@register("resnet3d_10")
class ResNet3D10(ResNet3D):
    def __init__(self, **kw):
        super().__init__([1, 1, 1, 1], **kw)


@register("resnet3d_18")
class ResNet3D18(ResNet3D):
    def __init__(self, **kw):
        super().__init__([2, 2, 2, 2], **kw)


@register("resnet3d_34")
class ResNet3D34(ResNet3D):
    def __init__(self, **kw):
        super().__init__([3, 4, 6, 3], **kw)


# ------------------------------------------------------------- Transformers --

def sincos_pos_embed_3d(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 3D sin-cos positional embedding, [grid_size^3, embed_dim]."""
    assert embed_dim % 6 == 0, "embed_dim must be divisible by 6"
    dim_axis = embed_dim // 3
    coords = np.arange(grid_size, dtype=np.float64)
    omega = 1.0 / (10000 ** (np.arange(dim_axis // 2, dtype=np.float64) / (dim_axis / 2)))
    per_axis = []
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=0).reshape(3, -1)
    for ax in range(3):
        angles = np.outer(grid[ax], omega)  # [N, dim_axis/2]
        per_axis.append(np.concatenate([np.sin(angles), np.cos(angles)], axis=1))
    emb = np.concatenate(per_axis, axis=1)  # [N, embed_dim]
    return torch.from_numpy(emb).float()


def _encoder_layers(dim: int, heads: int, depth: int, dropout: float = 0.0) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


@register("cnn_transformer_hybrid")
class CNNTransformerHybrid(nn.Module):
    """Small 3D CNN stem (96 -> 6 feature map, 216 tokens) + 4 transformer layers."""

    def __init__(self, dim: int = 384, depth: int = 4, heads: int = 6,
                 n_targets: int = 3, dropout: float = 0.2, pretrained_path: str | None = None):
        super().__init__()
        chs = [32, 64, 128, dim]
        blocks, in_ch = [], 1
        for ch in chs:  # 4 stride-2 stages: 96 -> 48 -> 24 -> 12 -> 6
            blocks += [nn.Conv3d(in_ch, ch, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm3d(ch), nn.ReLU(inplace=True),
                       nn.Conv3d(ch, ch, 3, padding=1, bias=False),
                       nn.BatchNorm3d(ch), nn.ReLU(inplace=True)]
            in_ch = ch
        self.stem = nn.Sequential(*blocks)
        n_tokens = 6 ** 3
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))  # learnable 3D pos embed
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.encoder = _encoder_layers(dim, heads, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = RegressionHead(dim, n_targets, dropout)
        if pretrained_path:
            load_pretrained_flexible(self, pretrained_path)

    def forward(self, x):
        f = self.stem(x)                                   # [B, dim, 6, 6, 6]
        tokens = f.flatten(2).transpose(1, 2)              # [B, 216, dim]
        tokens = tokens + self.pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(z[:, 0]))


@register("cnn_conv_control")
class CNNConvControl(nn.Module):
    """Parameter-matched control for cnn_transformer_hybrid: no attention.

    Identical CNN stem, but the 4 transformer layers (~7.18M params) are
    replaced by two 3x3x3 conv blocks at 384 channels (~7.97M params, i.e. the
    control has slightly MORE parameters — conservative for the claim that
    attention, not capacity, drives the hybrid's advantage), then GAP -> head.
    On the 6^3 token grid two 3^3 convs give a 5^3 receptive field (near- but
    not fully global), versus attention's global mixing from the first layer.
    """

    def __init__(self, dim: int = 384, n_targets: int = 3, dropout: float = 0.2,
                 norm: str = "bn", pretrained_path: str | None = None):
        super().__init__()
        chs = [32, 64, 128, dim]
        blocks, in_ch = [], 1
        for ch in chs:  # identical to CNNTransformerHybrid.stem (norm swappable)
            blocks += [nn.Conv3d(in_ch, ch, 3, stride=2, padding=1, bias=False),
                       make_norm(norm, ch), nn.ReLU(inplace=True),
                       nn.Conv3d(ch, ch, 3, padding=1, bias=False),
                       make_norm(norm, ch), nn.ReLU(inplace=True)]
            in_ch = ch
        self.stem = nn.Sequential(*blocks)
        self.mixer = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1, bias=False),
            make_norm(norm, dim), nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, 3, padding=1, bias=False),
            make_norm(norm, dim), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = RegressionHead(dim, n_targets, dropout)
        if pretrained_path:
            load_pretrained_flexible(self, pretrained_path)

    def forward(self, x):
        f = self.mixer(self.stem(x))                       # [B, dim, 6, 6, 6]
        return self.head(self.pool(f).flatten(1))


@register("resnet_transformer_hybrid")
class ResNetTransformerHybrid(nn.Module):
    """Ablation of cnn_transformer_hybrid: residual (ResNet-18-style) stem.

    Stem: conv1(7, s2) -> maxpool -> BasicBlock stages [2, 2, 2]
    (64@24^3 -> 128@12^3 -> 256@6^3) -> 1x1x1 projection to `dim`.
    Transformer head is identical to cnn_transformer_hybrid (4 layers,
    dim 384, 6 heads, learnable pos embed, CLS pooling), so any performance
    difference is attributable to the stem.
    """

    def __init__(self, dim: int = 384, depth: int = 4, heads: int = 6,
                 n_targets: int = 3, dropout: float = 0.2, norm: str = "bn",
                 pool: str = "cls", drop_path: float = 0.0,
                 pretrained_path: str | None = None):
        super().__init__()
        assert pool in ("cls", "cls+mean")
        self.pool_mode = pool
        self._norm_kind = norm
        self.conv1 = nn.Conv3d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = make_norm(norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)
        self.in_ch = 64
        self.layer1 = self._make_layer(64, 2, stride=1)    # 24^3
        self.layer2 = self._make_layer(128, 2, stride=2)   # 12^3
        self.layer3 = self._make_layer(256, 2, stride=2)   # 6^3
        self.proj = nn.Conv3d(256, dim, 1)
        n_tokens = 6 ** 3
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if drop_path > 0:
            self.encoder = nn.Sequential(
                *[PreNormBlock(dim, heads, drop_path) for _ in range(depth)])
        else:
            self.encoder = _encoder_layers(dim, heads, depth)
        self.norm = nn.LayerNorm(dim)
        feat_dim = dim * 2 if pool == "cls+mean" else dim
        self.head = RegressionHead(feat_dim, n_targets, dropout)
        if pretrained_path:
            load_pretrained_flexible(self, pretrained_path)

    def _make_layer(self, ch: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != ch:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_ch, ch, 1, stride=stride, bias=False),
                make_norm(self._norm_kind, ch))
        layers = [BasicBlock3d(self.in_ch, ch, stride, downsample, norm=self._norm_kind)]
        self.in_ch = ch
        layers += [BasicBlock3d(ch, ch, norm=self._norm_kind) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def stem(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.proj(self.layer3(self.layer2(self.layer1(x))))

    def forward(self, x):
        f = self.stem(x)                                   # [B, dim, 6, 6, 6]
        tokens = f.flatten(2).transpose(1, 2) + self.pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))
        z = self.norm(z)
        feat = z[:, 0]
        if self.pool_mode == "cls+mean":
            feat = torch.cat([feat, z[:, 1:].mean(dim=1)], dim=1)
        return self.head(feat)


@register("vit3d_small")
class ViT3DSmall(nn.Module):
    """3D ViT: patch 16 (216 tokens), dim 384, depth 8, heads 6, sin-cos pos, CLS."""

    def __init__(self, img_size: int = 96, patch_size: int = 16, dim: int = 384,
                 depth: int = 8, heads: int = 6, n_targets: int = 3,
                 dropout: float = 0.2, pretrained_path: str | None = None):
        super().__init__()
        assert img_size % patch_size == 0
        grid = img_size // patch_size
        self.patch_embed = nn.Conv3d(1, dim, patch_size, stride=patch_size)
        pos = sincos_pos_embed_3d(dim, grid)               # [216, dim], fixed
        self.register_buffer("pos_embed", pos.unsqueeze(0))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.encoder = _encoder_layers(dim, heads, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = RegressionHead(dim, n_targets, dropout)
        if pretrained_path:
            load_pretrained_flexible(self, pretrained_path)

    def forward(self, x):
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, 216, dim]
        tokens = tokens + self.pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(z[:, 0]))


@register("swinunetr_encoder")
class SwinUNETREncoder(nn.Module):
    """MONAI SwinUNETR encoder; GAP of the deepest feature map -> linear head.

    `pretrained_path` accepts the public CT self-supervised weights
    (model_swinvit.pt); loaded via SwinUNETR.load_from with a flexible fallback.
    """

    def __init__(self, feature_size: int = 48, n_targets: int = 3, dropout: float = 0.2,
                 pretrained_path: str | None = None):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        try:
            self.backbone = SwinUNETR(in_channels=1, out_channels=1, feature_size=feature_size,
                                      spatial_dims=3)
        except TypeError:  # older MONAI requires img_size
            self.backbone = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=1,
                                      feature_size=feature_size, spatial_dims=3)
        feat_dim = 16 * feature_size
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = RegressionHead(feat_dim, n_targets, dropout)
        if pretrained_path:
            try:
                weights = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                self.backbone.load_from(weights=weights)
                print(f"Loaded SwinUNETR SSL weights via load_from: {pretrained_path}")
            except Exception as e:
                print(f"load_from failed ({e}); falling back to flexible loading")
                load_pretrained_flexible(self.backbone, pretrained_path, submodule_prefix="swinViT.")

    def forward(self, x):
        hidden = self.backbone.swinViT(x, normalize=True)
        deepest = hidden[4]                                # [B, 16*fs, 3, 3, 3]
        return self.head(self.pool(deepest).flatten(1))
