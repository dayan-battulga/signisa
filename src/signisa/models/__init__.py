"""cnn_transformer backbone (~2M params; Kaggle 1st-place pattern, docs/research/task3a)
+ CE / ArcFace heads. Input (B, T, 65, 10) + real-frame mask + (B, 2) side features
-> 512-d L2-normalized embedding. Batches are padded to the batch maximum, so every
sequence-consuming step is mask-aware.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config
from ..data import side_features


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

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:  # (B, T, C), (B, T)
        attended = self.attn(x, x, x, key_padding_mask=pad, need_weights=False)[0]
        x = self._bn(self.norm1, x + self.drop_path(attended))
        return self._bn(self.norm2, x + self.drop_path(self.ffn(x)))


class Embedder(nn.Module):
    """(B, T, 65, 10) + (B, T) real-frame mask + (B, 2) side -> (B, embed_dim), L2-normalized.

    ponytail: the BatchNorms still see padded frames. Length-bucketed batching
    (signisa.train.LengthBucketSampler) keeps the padding fraction small; masked
    BatchNorm is the upgrade if the pad fraction ever gets large.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.stem = nn.Linear(cfg.n_nodes * cfg.n_channels, cfg.dim)
        self.conv = nn.ModuleList(
            ConvBlock(cfg.dim, cfg.conv_kernel, cfg.drop_path) for _ in range(cfg.n_conv_blocks))
        self.transformer = nn.ModuleList(
            BNTransformerLayer(cfg.dim, cfg.n_heads, cfg.ffn_mult, cfg.drop_path)
            for _ in range(cfg.n_transformer_layers))
        self.side_norm = nn.BatchNorm1d(2)  # duration (s) and log peak speed live on
        self.head = nn.Linear(cfg.dim + 2, cfg.embed_dim)  # different scales from the pooled dims

    def forward(self, x: torch.Tensor, mask: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        pad = mask < 0.5
        # padded frames are re-zeroed after every step that has a bias: the depthwise
        # convs read k//2 frames past a sequence's end, and must see zeros there or a
        # short sequence's tail depends on whatever else shares its batch
        h = self.stem(x.flatten(2)) * mask.unsqueeze(-1)    # (B, T, dim)
        h = h.transpose(1, 2)
        for block in self.conv:
            h = block(h) * mask.unsqueeze(1)
        h = h.transpose(1, 2)
        for layer in self.transformer:
            h = layer(h, pad)
        # confidence-weighted mean pool over REAL frames only
        weight = (x[..., 9].mean(dim=2) * mask).unsqueeze(-1)
        pooled = (h * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
        return F.normalize(self.head(torch.cat([pooled, self.side_norm(side)], dim=1)), dim=1)


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, side: torch.Tensor,
                labels: torch.Tensor | None = None):
        embedding = self.embedder(x, mask, side)
        if isinstance(self.head, ArcFaceHead):
            logits = self.head(embedding, labels)
        else:
            logits = self.head(embedding)
        return embedding, logits


@torch.no_grad()
def embedding_of(model: SignModel, pre) -> np.ndarray:
    """One preprocess() result -> embedding. The single-attempt inference seam:
    mask and side features have to be built the same way training built them."""
    x = torch.from_numpy(pre.tensor)[None]
    side = torch.from_numpy(side_features(pre.duration_s, pre.peak_speed))[None]
    return model.embedder(x, torch.ones(x.shape[:2]), side)[0].numpy()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: SignModel, path) -> None:
    """State dict + the metadata needed to reload without guessing."""
    torch.save({"state_dict": model.state_dict(), "loss": model.cfg.loss,
                "landmark_version": model.cfg.landmark_version}, path)


def load_checkpoint(path) -> SignModel:
    """Load either format: new metadata dicts, or legacy raw v1 state dicts
    (loss inferred from the presence of the CE head bias)."""
    loaded = torch.load(path, map_location="cpu")
    if "state_dict" in loaded:
        cfg = Config(loss=loaded["loss"], landmark_version=loaded["landmark_version"])
        state = loaded["state_dict"]
    else:
        cfg = Config(loss="ce" if "head.bias" in loaded else "arcface")
        state = loaded
    model = SignModel(cfg)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError("checkpoint architecture doesn't match the default Config "
                           f"(dim/embed_dim/n_classes/landmark_version): {e}") from e
    model.eval()
    return model
