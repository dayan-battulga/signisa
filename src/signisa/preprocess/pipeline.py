"""Preprocessing chain (research Tasks 2 + 3B): raw holistic landmarks -> (T_OUT, 65, 10) tensor.

Order of operations:
    mirror (left-dominant only) -> fill short gaps
    -> root-center -> shoulder-width scale -> rotate to canonical frame
    -> One Euro filter (in normalized units) -> side features (duration, peak speed)
    -> zero-fill remaining gaps -> resample to T_OUT
    -> velocity + bone vectors + confidence channel.

Notes anchored in the research docs:
- Zero-filling long occlusion gaps (rather than interpolating) follows the Kaggle
  asl-signs winners: absence becomes a learnable feature, flagged by the confidence channel.
- Resampling to a fixed T_OUT removes execution-speed differences between fluent
  signers and slow novices (3B's domain-shift mitigation). Raw duration and peak
  speed are preserved as side features because movement speed/tension can be
  lexically contrastive (backlog flag #2).
"""

from dataclasses import dataclass

import numpy as np

from .landmarks import (
    HOLISTIC_INDICES,
    L_SHOULDER,
    MIRROR_PERM,
    N_FEATURES,
    N_NODES,
    NOSE,
    PARENT,
    R_SHOULDER,
)

T_OUT = 160
MAX_GAP_FRAMES = 3
_EPS = 1e-6

@dataclass(frozen=True)
class Preprocessed:
    tensor: np.ndarray   # (T_OUT, 65, 10) float32, NaN-free
    duration_s: float    # raw attempt duration before resampling
    peak_speed: float    # max per-frame displacement * fps, in shoulder-width units


def select_nodes(holistic: np.ndarray) -> np.ndarray:
    """(T, 543, 3) -> (T, 65, 3)."""
    return holistic[:, HOLISTIC_INDICES]


def mirrored(seq: np.ndarray) -> np.ndarray:
    """Reflect a left-dominant sequence into right-dominant canonical space."""
    out = seq[:, MIRROR_PERM].copy()
    out[..., 0] = -out[..., 0]
    return out


def fill_short_gaps(seq: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    """Linearly interpolate missing runs of <= max_gap frames per node; leave longer gaps NaN."""
    out = seq.copy()
    t = np.arange(seq.shape[0])
    for node in range(seq.shape[1]):
        missing = np.isnan(seq[:, node]).any(axis=1)
        if not missing.any() or missing.all():
            continue
        for start, stop in _runs(missing):
            interior = start > 0 and stop < len(missing)
            if interior and (stop - start) <= max_gap:
                valid = ~missing
                for c in range(seq.shape[2]):
                    out[start:stop, node, c] = np.interp(
                        t[start:stop], t[valid], seq[valid, node, c]
                    )
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of consecutive True runs."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def one_euro(seq: np.ndarray, fps: float, min_cutoff: float = 1.0, beta: float = 0.7) -> np.ndarray:
    """One Euro filter over time, vectorized across nodes/channels; NaN frames pass through."""
    dt = 1.0 / fps
    out = np.full_like(seq, np.nan)
    x_prev = None
    dx_prev = np.zeros(seq.shape[1:])
    for t in range(seq.shape[0]):
        x = seq[t]
        valid = ~np.isnan(x)
        if x_prev is None:
            if valid.any():
                x_prev = np.where(valid, x, np.nan)
                out[t] = x_prev
            continue
        dx = np.where(valid & ~np.isnan(x_prev), (x - x_prev) / dt, 0.0)
        dx_hat = _lowpass(dx, dx_prev, _alpha(1.0, dt))
        cutoff = min_cutoff + beta * np.abs(dx_hat)
        x_hat = np.where(
            valid & ~np.isnan(x_prev), _lowpass(x, x_prev, _alpha(cutoff, dt)), x
        )
        out[t] = x_hat
        x_prev = np.where(np.isnan(x_hat), x_prev, x_hat)  # hold state through gaps
        dx_prev = np.where(np.isnan(dx_hat), dx_prev, dx_hat)
    return out


def _alpha(cutoff, dt: float):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _lowpass(x, x_prev, alpha):
    return alpha * x + (1.0 - alpha) * x_prev


def root_centered(seq: np.ndarray) -> np.ndarray:
    """Translate so the shoulder midpoint is the origin; missing roots borrow the nearest valid one."""
    root = (seq[:, L_SHOULDER] + seq[:, R_SHOULDER]) / 2.0
    root = _nearest_valid_fill(root)
    return seq - root[:, None, :]


def _nearest_valid_fill(points: np.ndarray) -> np.ndarray:
    """(T, 3) with NaN rows replaced by the nearest valid row (sequence median if none)."""
    invalid = np.isnan(points).any(axis=1)
    if not invalid.any():
        return points
    if invalid.all():
        return np.zeros_like(points)
    valid_idx = np.flatnonzero(~invalid)
    nearest = valid_idx[np.abs(np.arange(len(points))[:, None] - valid_idx).argmin(axis=1)]
    return points[nearest]


def scaled(seq: np.ndarray) -> np.ndarray:
    """Divide by the median shoulder width so distances are in shoulder-width units."""
    widths = np.linalg.norm(seq[:, L_SHOULDER] - seq[:, R_SHOULDER], axis=1)
    width = np.nanmedian(widths)
    if not np.isfinite(width) or width < _EPS:
        return seq
    return seq / width


def canonical(seq: np.ndarray) -> np.ndarray:
    """Rotate so the mean shoulder line is the X axis and shoulders->nose is the Y axis.

    Means (not medians) keep this step exactly rotation-equivariant.
    """
    x_dir = _unit(np.nanmean(seq[:, R_SHOULDER] - seq[:, L_SHOULDER], axis=0))
    up_dir = _unit(np.nanmean(seq[:, NOSE], axis=0))  # root is the shoulder midpoint
    if x_dir is None or up_dir is None:
        return seq
    z_dir = _unit(np.cross(x_dir, up_dir))
    if z_dir is None:
        return seq
    y_dir = np.cross(z_dir, x_dir)
    rotation = np.stack([x_dir, y_dir, z_dir])  # rows: new basis
    return seq @ rotation.T


def _unit(v: np.ndarray):
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < _EPS:
        return None
    return v / norm


def zero_filled(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace remaining NaN nodes with the origin; return (seq, presence) with presence in {0,1}."""
    present = ~np.isnan(seq).any(axis=2)
    return np.where(present[..., None], seq, 0.0), present.astype(np.float32)


def resampled(values: np.ndarray, t_out: int) -> np.ndarray:
    """Linearly resample the time axis of (T, ...) to (t_out, ...)."""
    t_in = values.shape[0]
    if t_in == 1:
        return np.repeat(values, t_out, axis=0)
    src = np.linspace(0.0, 1.0, t_in)
    dst = np.linspace(0.0, 1.0, t_out)
    flat = values.reshape(t_in, -1)
    out = np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return out.reshape((t_out,) + values.shape[1:])


def peak_speed_of(seq: np.ndarray, fps: float) -> float:
    """Max frame-to-frame displacement speed across nodes, in shoulder-width units / second."""
    step = np.linalg.norm(np.diff(seq, axis=0), axis=2)  # (T-1, 65)
    if step.size == 0 or np.isnan(step).all():
        return 0.0
    return float(np.nanmax(step) * fps)


def preprocess(holistic: np.ndarray, fps: float = 30.0, left_dominant: bool = False) -> Preprocessed:
    """Full chain: (T, 543, 3) raw holistic landmarks -> Preprocessed tensor + side features."""
    seq = select_nodes(holistic).astype(np.float64)
    if left_dominant:
        seq = mirrored(seq)
    seq = fill_short_gaps(seq)
    seq = canonical(scaled(root_centered(seq)))
    seq = one_euro(seq, fps)  # after normalization so filter params act in subject-independent units

    duration_s = seq.shape[0] / fps
    peak = peak_speed_of(seq, fps)

    coords, presence = zero_filled(seq)
    coords = resampled(coords, T_OUT)
    confidence = np.clip(resampled(presence[..., None], T_OUT), 0.0, 1.0)

    tensor = with_derived_channels(coords, confidence)
    assert tensor.shape == (T_OUT, N_NODES, N_FEATURES)
    return Preprocessed(tensor=tensor, duration_s=duration_s, peak_speed=peak)


def with_derived_channels(coords: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """(T, 65, 3) coords + (T, 65, 1) confidence -> (T, 65, 10) xyz+velocity+bone+confidence.

    Shared with signisa.data, which stores only xyz+confidence and re-derives the rest.
    """
    velocity = np.diff(coords, axis=0, prepend=coords[:1])
    bones = coords - coords[:, PARENT]
    return np.concatenate([coords, velocity, bones, confidence], axis=2).astype(np.float32)
