"""PyTorch port of hoyso48's Kaggle asl-signs 1st-place recipe (MIT), for one diagnostic:
does THEIR preprocessing + model + training reach ~80% top-1 on OUR 17/4 signer split,
where signisa's own pipeline sits at ~50%? (memory.md, session 17.)

Source: github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution
(one notebook; AWP from github.com/hoyso48/tf-utils). Deliberately separate from
signisa.preprocess / signisa.models — mixing in our pipeline would defeat the test.
Inference is verified against their published TFLite weights (tests/test_kaggle1st.py).
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .models import DropPath
from .preprocess.kaggle import load_holistic

# Landmark rows of the Kaggle 543-row frame, in their gather order (118 points).
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321,
       375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318,
       324, 308]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE

# Their flip_lr pairs (left list i <-> right list i); midline lip/nose points stay put.
_LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
_RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
_PAIRS = [*zip(LHAND, RHAND), *zip(_LLIP, _RLIP), *zip(LEYE, REYE), (98, 327)]
_PARTNER = {a: b for pair in _PAIRS for a, b in (pair, pair[::-1])}
FLIP_PERM = np.array([POINT_LANDMARKS.index(_PARTNER.get(p, p)) for p in POINT_LANDMARKS])

# Their normalization anchor is Kaggle row 17 — the lower-lip centre, though the writeup
# calls it the nose. The TFLite gathers [17], so that is what we copy.
ANCHOR = POINT_LANDMARKS.index(17)
MAX_LEN = 384
N_FEATURES = 3 * 2 * len(POINT_LANDMARKS)  # (xy, dxy, d2xy) per point = 708
N_CLASSES = 250


def points_of(holistic: np.ndarray) -> np.ndarray:
    """(T, 543, 3) Kaggle frames -> (T', 118, 2). xy only: their Preprocess scales x and y
    each by its own std and drops z, so z never reaches a feature. Frames holding none of
    the points are dropped (their filter_nans_tf) — but never all of them: one NaN frame
    becomes one zero-feature frame instead of an empty sequence."""
    x = holistic[:, POINT_LANDMARKS, :2]
    kept = x[~np.isnan(x).all(axis=(1, 2))]
    return kept if len(kept) else x[:1]


def features_of(points: np.ndarray) -> np.ndarray:
    """Their Preprocess layer: (T, 118, 2) -> (min(T, 384), 708) float32. Centre on the
    anchor's mean, scale each axis by its NaN-aware std about it (statistics over the FULL
    sequence, before the 384-frame crop), then xy + 1- and 2-frame differences, NaN -> 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = ~np.isnan(points)
        anchor = np.where(valid, points, 0)[:, ANCHOR].sum(0) / valid[:, ANCHOR].sum(0)
        anchor = np.where(np.isnan(anchor), 0.5, anchor)
        dev = np.where(valid, points - anchor, 0)
        std = np.sqrt((dev * dev).sum((0, 1)) / valid.sum((0, 1)))
        x = ((points - anchor) / std)[:MAX_LEN]
    dx, dx2 = np.zeros_like(x), np.zeros_like(x)
    dx[:-1] = x[1:] - x[:-1]
    dx2[:-2] = x[2:] - x[:-2]
    out = np.concatenate([v.reshape(len(x), -1) for v in (x, dx, dx2)], axis=1)
    return np.where(np.isnan(out), 0, out).astype(np.float32)


def flipped(points: np.ndarray) -> np.ndarray:
    """Their flip_lr: mirror x about the image centre and swap every left/right pair."""
    x = points[:, FLIP_PERM].copy()
    x[..., 0] = 1 - x[..., 0]
    return x


def resampled(points: np.ndarray, length: int, nearest: bool) -> np.ndarray:
    """tf.image.resize along time with half-pixel centres. The lerp is written a + (b-a)w
    as in TF, so a NaN neighbour poisons the output even at w = 0 — same as theirs.
    ponytail: their bicubic third runs as linear here; port Keys' kernel if it matters."""
    src = (np.arange(length) + 0.5) * len(points) / length - 0.5
    if nearest:
        return points[np.minimum(np.floor(src + 0.5).astype(int), len(points) - 1)]
    src = np.clip(src, 0, len(points) - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(points) - 1)
    w = (src - lo)[:, None, None]
    return points[lo] + (points[hi] - points[lo]) * w


def augmented(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Their augment_fn — same order, probabilities and ranges, in raw image coordinates."""
    x = points
    if rng.random() < 0.8:
        x = resampled(x, max(1, int(rng.uniform(0.5, 1.5) * len(x))), nearest=rng.random() < 1 / 3)
    if rng.random() < 0.5:
        x = flipped(x)
    x = x[rng.integers(0, np.clip(len(x) - MAX_LEN, 1, MAX_LEN)):][:MAX_LEN]
    if rng.random() < 0.75:  # scale -> shear -> rotate about the (sheared) centre -> shift
        x = x * rng.uniform(0.8, 1.2)
        s = rng.uniform(-0.15, 0.15)
        shear_x, shear_y = (0.0, s) if rng.random() < 0.5 else (s, 0.0)
        x = x @ np.array([[1, shear_x], [shear_y, 1]])
        centre = np.array([0.5 + shear_y, 0.5 + shear_x])
        t = np.radians(rng.uniform(-30, 30))
        x = (x - centre) @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) + centre
        x = x + rng.uniform(-0.1, 0.1)  # one scalar for both axes, as theirs
    x = x.copy()
    if rng.random() < 0.5:  # blank one span of 20-40% of the frames
        n = int(len(x) * rng.uniform(0.2, 0.4))
        start = rng.integers(0, np.clip(len(x) - n, 1, len(x)))
        x[start:start + n] = np.nan
    if rng.random() < 0.5:  # blank every point inside a random square
        corner, size = rng.random(2), rng.uniform(0.2, 0.4)
        x[((x > corner) & (x < corner + size)).all(-1)] = np.nan
    return x


def asl_signs_root(input_dir="/kaggle/input") -> Path:
    """The dir holding asl-signs' train.csv beside train_landmark_files/, at ANY mount depth
    (Kaggle nests mounts unpredictably — memory.md session 15). Never walks the parquets."""
    roots = []
    for d, dirs, files in os.walk(input_dir):
        if "train.csv" in files and "train_landmark_files" in dirs:
            roots.append(Path(d))
        dirs[:] = [x for x in dirs if x != "train_landmark_files"]
    assert len(roots) == 1, (
        f"expected one asl-signs root under {input_dir}, found {roots} — is the asl-signs "
        f"competition data attached? Attached: {sorted(str(p) for p in Path(input_dir).glob('*/*'))}")
    return roots[0]


def _points_of_file(path) -> np.ndarray:  # top-level so the process pool can pickle it
    return points_of(load_holistic(path))


def points_cache(paths, labels, workers: int) -> dict:
    """All sequences as one ragged float32 array (frames + offsets + lengths) + labels.
    One array, not 94k: forked DataLoader workers then share it copy-on-write."""
    with ProcessPoolExecutor(workers) as pool:
        tracks = list(pool.map(_points_of_file, paths, chunksize=256))
    lengths = np.array([len(t) for t in tracks])
    return {"frames": np.concatenate(tracks), "lengths": lengths,
            "offsets": np.concatenate([[0], np.cumsum(lengths)[:-1]]),
            "labels": np.asarray(labels)}


class PointsDataset(Dataset):
    """Rows of a points_cache -> ((T, 708) features, Kaggle class index)."""

    def __init__(self, cache: dict, rows: np.ndarray, augment: bool):
        self.frames, self.augment = cache["frames"], augment
        self.offsets, self.lengths, self.labels = (
            cache[k][rows] for k in ("offsets", "lengths", "labels"))

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, i):
        x = self.frames[self.offsets[i]:self.offsets[i] + self.lengths[i]]
        if self.augment:
            # torch reseeds each worker every epoch; a numpy generator built in __init__
            # would be cloned into every worker and repeat the same augmentations
            x = augmented(x, np.random.default_rng(torch.randint(2**62, (1,)).item()))
        return torch.from_numpy(features_of(x)), int(self.labels[i])


def padded(batch):
    """(features, label) samples -> zero-padded (B, T, 708), (B, T) real-frame mask, labels."""
    x = pad_sequence([f for f, _ in batch], batch_first=True)
    lengths = torch.tensor([len(f) for f, _ in batch])
    return x, torch.arange(x.shape[1]) < lengths[:, None], torch.tensor([y for _, y in batch])


class MaskedBatchNorm(nn.BatchNorm1d):
    """BatchNorm statistics over real frames only, as Keras BN honours the Masking layer.
    Keras defaults: eps 1e-3, momentum 0.95 (torch's 0.05)."""

    def __init__(self, dim: int):
        super().__init__(dim, eps=1e-3, momentum=0.05)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        out = torch.zeros_like(x)
        out[mask] = super().forward(x[mask]).to(x.dtype)
        return out


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, C)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1)


class ECA(nn.Module):
    """Efficient channel attention: a k=5 conv across the time-pooled channels gates them."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 5, padding=2, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.conv(_masked_mean(x, mask).unsqueeze(1)))  # (B, 1, C)
        return x * gate


class ConvBlock(nn.Module):
    """expand (swish) -> CAUSAL depthwise k=17 -> BN -> ECA -> project, residual + DropPath.
    Causal padding also means right-side batch padding can never leak into real frames."""

    def __init__(self, dim: int, kernel: int = 17, expand: int = 2):
        super().__init__()
        self.kernel = kernel
        self.expand = nn.Linear(dim, dim * expand)
        self.dwconv = nn.Conv1d(dim * expand, dim * expand, kernel, groups=dim * expand, bias=False)
        self.bn = MaskedBatchNorm(dim * expand)
        self.eca = ECA()
        self.project = nn.Linear(dim * expand, dim)
        self.drop = DropPath(0.2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.expand(x)).transpose(1, 2)
        h = self.dwconv(F.pad(h, (self.kernel - 1, 0))).transpose(1, 2)
        return x + self.drop(self.project(self.eca(self.bn(h, mask), mask)))


class TransformerBlock(nn.Module):
    """Pre-BN attention + pre-BN swish FFN (expand 2), DropPath 0.2 on both branches."""

    def __init__(self, dim: int, heads: int = 4, expand: int = 2):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = MaskedBatchNorm(dim), MaskedBatchNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim * expand, bias=False)
        self.down = nn.Linear(dim * expand, dim, bias=False)
        self.drop = DropPath(0.2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        # their qkv layout: per head, [q | k | v] chunks of the 3c/heads slice
        q, k, v = (self.qkv(self.norm1(x, mask)).view(b, t, self.heads, -1)
                   .transpose(1, 2).chunk(3, dim=-1))
        attn = (q @ k.transpose(-2, -1)) * c ** -0.5  # full-dim scale, not per-head (TFLite: 0.0721)
        attn = attn.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        h = (F.dropout(attn, 0.2, self.training) @ v).transpose(1, 2).reshape(b, t, c)
        x = x + self.drop(self.proj(h))
        return x + self.drop(self.down(F.silu(self.up(self.norm2(x, mask)))))


class Kaggle1stNet(nn.Module):
    """(B, T, 708) + (B, T) mask -> (B, 250) logits. 3 conv + transformer, twice; dim 192."""

    def __init__(self, dim: int = 192):
        super().__init__()
        self.stem = nn.Linear(N_FEATURES, dim, bias=False)
        self.stem_bn = MaskedBatchNorm(dim)
        self.blocks = nn.ModuleList(
            [ConvBlock(dim) if i % 4 < 3 else TransformerBlock(dim) for i in range(8)])
        self.top = nn.Linear(dim, 2 * dim)
        self.classifier = nn.Linear(2 * dim, N_CLASSES)
        self.late_dropout = False  # the trainer switches their 0.8 dropout on at its start epoch
        for m in self.modules():  # Keras defaults: glorot-uniform kernels, zero biases
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.stem_bn(self.stem(x), mask)
        for block in self.blocks:
            h = block(h, mask)
        pooled = _masked_mean(self.top(h), mask)
        return self.classifier(F.dropout(pooled, 0.8, self.training and self.late_dropout))


@dataclass
class RecipeConfig:
    """Their CFG cell on one GPU. They ran 8 TPU replicas x 64 = 512 per step at 8 x 5e-4;
    here 8 accumulated micro-batches of 64 — BN statistics and AWP gradients per 64 as on
    each replica, and a 64 x 384-frame worst case fits a 16 GB GPU where 512 x 384 would not."""
    epochs: int = 300
    batch_size: int = 512
    micro_batch: int = 64
    lr: float = 4e-3
    awp_lambda: float = 0.2
    awp_start_epoch: int = 15
    late_dropout_start_epoch: int = 15
    seed: int = 42
    num_workers: int = 4
    time_budget_h: float = 11.0  # stop (and still evaluate) before Kaggle's 12 h kill


@torch.no_grad()
def logits_of(model: Kaggle1stNet, dataset: Dataset, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 250) logits and (N,) labels, in dataset order."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, collate_fn=padded, num_workers=2)
    pairs = [(model(x.to(device), mask.to(device)).float().cpu(), y) for x, mask, y in loader]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


def train_kaggle1st(cfg: RecipeConfig, train_ds: Dataset, val_ds: Dataset,
                    device: str) -> tuple[Kaggle1stNet, dict]:
    """Their fit: CE + label smoothing 0.1, RAdam + Lookahead(5, 0.5), cosine to 1e-6 with
    no warmup, AWP and the late 0.8 dropout from epoch 15. Random (not length-bucketed)
    batches, like their shuffle buffer. Returns the best-val-loss model (their
    ModelCheckpoint) and per-epoch history."""
    torch.manual_seed(cfg.seed)
    model = Kaggle1stNet().to(device)
    params = list(model.parameters())
    loader = DataLoader(train_ds, batch_size=cfg.micro_batch, shuffle=True, drop_last=True,
                        collate_fn=padded, num_workers=cfg.num_workers,
                        pin_memory=device.startswith("cuda"),
                        persistent_workers=cfg.num_workers > 0)
    accum = cfg.batch_size // cfg.micro_batch
    assert accum * cfg.micro_batch == cfg.batch_size and len(loader) >= accum, (
        f"batch_size {cfg.batch_size} must be a multiple of micro_batch {cfg.micro_batch} "
        f"and fit in one epoch of {len(train_ds)} sequences")
    optimizer = torch.optim.RAdam(params, lr=cfg.lr, eps=1e-7, decoupled_weight_decay=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * (len(loader) // accum), eta_min=1e-6)
    slow = [p.detach().clone() for p in params]  # Lookahead's slow weights
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    def loss_of(x, mask, y):
        with torch.autocast(device_type="cuda", enabled=use_amp):
            return F.cross_entropy(model(x, mask), y, label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [], "val_top1": [], "hours": [], "awp_skipped": []}
    best_loss, best_state, step, started = math.inf, None, 0, time.perf_counter()
    for epoch in range(cfg.epochs):
        epoch_started = time.perf_counter()
        model.train()
        model.late_dropout = epoch >= cfg.late_dropout_start_epoch
        awp = epoch >= cfg.awp_start_epoch
        losses, awp_skips = [], 0
        for i, (x, mask, y) in enumerate(loader):
            x, mask, y = (t.to(device, non_blocking=True) for t in (x, mask, y))
            if i % accum == 0:  # an epoch's < accum leftover micro-batches are dropped here
                optimizer.zero_grad(set_to_none=True)
            loss = loss_of(x, mask, y)
            deltas = []
            if awp:  # their AWP, per replica: step every weight by lambda * g/||g||, grad there
                # scaled exactly like the backward below: the scaler only watches that one, so
                # a larger scale here overflowed unseen and switched AWP off (session 18)
                grads = torch.autograd.grad(scaler.scale(loss / accum), params)
                norms = torch.stack([g.norm() for g in grads])
                if torch.isfinite(norms).all():  # an fp16-overflow step: the scaler skips it anyway
                    deltas = [g * (cfg.awp_lambda / n.clamp_min(1e-12)) for g, n in zip(grads, norms)]
                    with torch.no_grad():
                        torch._foreach_add_(params, deltas)
                else:
                    awp_skips += 1
                loss = loss_of(x, mask, y)
            scaler.scale(loss / accum).backward()
            if deltas:
                with torch.no_grad():
                    torch._foreach_sub_(params, deltas)
            losses.append(loss.item())
            if (i + 1) % accum:
                continue
            for group in optimizer.param_groups:
                # tfa RAdam decays by lr * wd_t, and their wd_t schedule is 0.1 * lr
                group["weight_decay"] = 0.1 * group["lr"]
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            if step % 5 == 0:  # Lookahead sync: slow += 0.5 (fast - slow); fast = slow
                with torch.no_grad():
                    for p, s in zip(params, slow):
                        s.lerp_(p, 0.5)
                        p.copy_(s)

        logits, labels = logits_of(model, val_ds, device)
        val_loss = F.cross_entropy(logits, labels, label_smoothing=0.1).item()
        hours = (time.perf_counter() - started) / 3600
        for key, value in (("train_loss", float(np.mean(losses))), ("val_loss", val_loss),
                           ("val_top1", (logits.argmax(1) == labels).float().mean().item()),
                           ("hours", hours), ("awp_skipped", awp_skips / len(losses))):
            history[key].append(value)
        epoch_h = (time.perf_counter() - epoch_started) / 3600
        print(f"epoch {epoch + 1}/{cfg.epochs} loss {history['train_loss'][-1]:.3f} "
              f"val_loss {val_loss:.3f} val_top1 {history['val_top1'][-1]:.3f} "
              f"awp_skipped {history['awp_skipped'][-1]:.1%} | "
              f"{hours:.2f} h, ~{hours + epoch_h * (cfg.epochs - epoch - 1):.1f} h projected",
              flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if hours + epoch_h > cfg.time_budget_h:
            print(f"TIME BUDGET: stopping after epoch {epoch + 1} of {cfg.epochs} — the cosine "
                  f"schedule did NOT finish; lower EPOCHS for a faithful run")
            break

    model.load_state_dict(best_state)
    return model, history
