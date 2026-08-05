"""cnn_transformer backbone (~2M params; Kaggle 1st-place pattern, docs/research/task3a)
+ CE / ArcFace heads. Input (B, 160, 65, 10) -> 512-d L2-normalized embedding.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config


class DropPath(nn.Module):
    """Per-sample residual-branch dropout (stochastic depth)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
        return x * mask / keep


class ConvBlock(nn.Module):
    """Depthwise Conv1D + pointwise + BatchNorm + Swish, residual with DropPath."""

    def __init__(self, dim: int, kernel: int, drop_path: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return x + self.drop_path(self.body(x))


class BNTransformerLayer(nn.Module):
    """Post-norm transformer encoder layer with BatchNorm instead of LayerNorm (per 3A)."""

    def __init__(self, dim: int, heads: int, ffn_mult: int, drop_path: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.SiLU(), nn.Linear(dim * ffn_mult, dim))
        self.norm1, self.norm2 = nn.BatchNorm1d(dim), nn.BatchNorm1d(dim)
        self.drop_path = DropPath(drop_path)

    @staticmethod
    def _bn(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        x = self._bn(self.norm1, x + self.drop_path(self.attn(x, x, x, need_weights=False)[0]))
        return self._bn(self.norm2, x + self.drop_path(self.ffn(x)))


class Embedder(nn.Module):
    """(B, T, 65, 10) -> (B, embed_dim), L2-normalized."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.stem = nn.Linear(cfg.n_nodes * cfg.n_channels, cfg.dim)
        self.conv = nn.ModuleList(
            ConvBlock(cfg.dim, cfg.conv_kernel, cfg.drop_path) for _ in range(cfg.n_conv_blocks))
        self.transformer = nn.ModuleList(
            BNTransformerLayer(cfg.dim, cfg.n_heads, cfg.ffn_mult, cfg.drop_path)
            for _ in range(cfg.n_transformer_layers))
        self.head = nn.Linear(cfg.dim, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = x[..., 9].mean(dim=2)                      # (B, T) mean node confidence
        h = self.stem(x.flatten(2))                         # (B, T, dim)
        h = h.transpose(1, 2)
        for block in self.conv:
            h = block(h)
        h = h.transpose(1, 2)
        for layer in self.transformer:
            h = layer(h)
        weight = weight.unsqueeze(-1) + 1e-6                # confidence-weighted mean pool
        pooled = (h * weight).sum(dim=1) / weight.sum(dim=1)
        return F.normalize(self.head(pooled), dim=1)


class ArcFaceHead(nn.Module):
    """Additive angular margin logits: s * cos(theta + m) on the target class.

    Runs in fp32 with autocast disabled: under fp16 the acos clamp epsilon rounds
    away and cos values at +-1 produce inf gradients.
    """

    def __init__(self, embed_dim: int, n_classes: int, s: float, m: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = s, m
        # past theta = pi - m, cos(theta + m) turns non-monotonic and would REWARD a
        # badly-wrong target; standard guard falls back to the linear penalty cos - m*sin(m)
        self.cos_floor = math.cos(math.pi - m)
        self.linear_penalty = math.sin(math.pi - m) * m

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            cos = F.linear(embedding.float(), F.normalize(self.weight, dim=1).float())
            if labels is None:
                return self.s * cos
            theta = torch.acos(cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            margined = torch.where(cos > self.cos_floor,
                                   torch.cos(theta + self.m), cos - self.linear_penalty)
            onehot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
            return self.s * (onehot * margined + (1.0 - onehot) * cos)


class SignModel(nn.Module):
    """Embedder + classification head selected by cfg.loss ('ce' | 'arcface')."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.loss in ("ce", "arcface"), cfg.loss
        self.cfg = cfg
        self.embedder = Embedder(cfg)
        if cfg.loss == "ce":
            self.head = nn.Linear(cfg.embed_dim, cfg.n_classes)
        else:
            self.head = ArcFaceHead(cfg.embed_dim, cfg.n_classes, cfg.arcface_s, cfg.arcface_m)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        embedding = self.embedder(x)
        if isinstance(self.head, ArcFaceHead):
            logits = self.head(embedding, labels)
        else:
            logits = self.head(embedding)
        return embedding, logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
