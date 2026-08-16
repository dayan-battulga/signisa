"""Torch Dataset over precomputed ragged float16 shards (native-length sequences).

Stored channels are xyz + confidence; velocity and bone channels are re-derived
here (signisa.preprocess.pipeline.with_derived_channels) so the reconstructed
(T, N, 10) float32 matches preprocess() output to float16 storage rounding
(measured max error ~4e-3 on real data). Augmentations run BEFORE derivation.

Sequences keep their native length, so batches are padded to the batch maximum by
pad_collate and carry a real-frame mask. Random horizontal flip IS applied at train
time (canonical-space mirror via MIRROR_PERM): it symmetrizes training only —
inference still runs in canonical right-dominant space, so the locked "no naive
horizontal-flip augmentation" rule (which is about inference-time handedness) holds.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import SHARD_SCHEMA
from .config import AugmentConfig
from .preprocess.landmarks import LANDMARK_SETS
from .preprocess.pipeline import resampled, with_derived_channels


def side_features(duration_s: float, peak_speed: float) -> np.ndarray:
    """The rhythm scalars fed to the model alongside the sequence.

    peak_speed is heavy-tailed (a single tracker jitter spike dominates the max),
    so it goes in as log1p; duration_s is already well-scaled in seconds.
    """
    return np.array([duration_s, np.log1p(max(peak_speed, 0.0))], dtype=np.float32)


def augmented(stored: np.ndarray, side: np.ndarray, cfg: AugmentConfig,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Train-time augmentation of one stored (T, N, 4) sequence and its side features."""
    duration, log_speed = float(side[0]), float(side[1])

    # speed scale: signing s x faster is T/s frames, duration/s seconds, s x the peak speed
    speed = rng.uniform(cfg.speed_min, cfg.speed_max)
    n = max(2, round(stored.shape[0] / speed))
    stored = resampled(stored.astype(np.float32), n)
    duration /= speed
    log_speed = float(np.log1p(np.expm1(log_speed) * speed))

    # random temporal crop: a learner's recording window rarely brackets the sign exactly.
    # ponytail: peak speed is left alone — cropping can only lower a max, never raise it.
    keep = max(2, round(n * rng.uniform(cfg.crop_min_frac, 1.0)))
    start = int(rng.integers(0, n - keep + 1))
    stored = stored[start:start + keep]
    duration *= keep / n

    t, n_nodes = stored.shape[:2]
    coords, confidence = stored[..., :3], stored[..., 3]
    present = confidence > 0

    # affine jitter about the origin (rotation in the canonical xy plane) + noise,
    # present nodes only — missing nodes stay at the origin with confidence 0
    theta = np.deg2rad(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scale = 1.0 + rng.uniform(-cfg.scale, cfg.scale)
    trans = rng.uniform(-cfg.translation, cfg.translation, size=3)
    jittered = (coords @ rot.T) * scale + trans
    jittered += rng.normal(0.0, cfg.noise_sigma, coords.shape)
    stored[..., :3] = np.where(present[..., None], jittered, coords)

    # node dropout: whole-sequence, like a landmark the tracker never found
    stored[:, rng.random(n_nodes) < cfg.node_dropout_p] = 0.0

    # temporal masking: random spans totaling up to mask_total_frac of frames
    target = rng.integers(0, int(cfg.mask_total_frac * t) + 1)
    masked = 0
    while masked < target:
        span = min(int(rng.integers(cfg.mask_span_min, cfg.mask_span_max + 1)),
                   int(target - masked))  # clamp so spans never exceed the budget
        span = min(span, t)
        begin = int(rng.integers(0, t - span + 1))
        stored[begin:begin + span] = 0.0
        masked += span
    return stored, np.array([duration, log_speed], dtype=np.float32)


def mirrored_stored(stored: np.ndarray, version: str = "v1") -> np.ndarray:
    """Flip a stored (T, N, 4) tensor's orientation in canonical space.

    Exactly equivalent to having mirrored the raw sequence before preprocess
    (every pipeline step is mirror-equivariant; verified to zero error on real
    data — z keeps its sign because the canonical frame re-derives it as x*up).
    """
    lset = LANDMARK_SETS[version]
    assert stored.shape[1] == lset.n_nodes, (
        f"{stored.shape} is not a {version} tensor")  # v1 perm would silently truncate v2
    out = stored[:, lset.mirror_perm].copy()
    out[..., 0] = -out[..., 0]
    return out


@lru_cache(maxsize=1)
def _load_shards(tensors_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Every shard's frames concatenated, plus (n_seq + 1,) slice offsets.

    ponytail: whole dataset in RAM (float16; train/val/eval instances share ONE copy
    via this cache). Switch the build to raw .npy + mmap_mode='r' if a Kaggle instance
    can't hold it.
    """
    frames, lengths = [], []
    for path in sorted(Path(tensors_dir).glob("shard_*.npz")):
        shard = np.load(path)
        schema = int(shard["schema_version"]) if "schema_version" in shard else 1
        assert schema == SHARD_SCHEMA, (
            f"{path.name} is shard schema {schema}, expected {SHARD_SCHEMA} — "
            "rebuild with scripts/build_training_tensors.py")
        frames.append(shard["frames"])
        lengths.append(shard["lengths"])
    all_frames = np.concatenate(frames)
    all_frames.setflags(write=False)
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(lengths))])
    return all_frames, offsets


def pad_collate(batch):
    """(tensor, side, label) samples -> (padded x, real-frame mask, side, labels)."""
    lengths = [sample[0].shape[0] for sample in batch]
    x = torch.zeros(len(batch), max(lengths), *batch[0][0].shape[1:])
    mask = torch.zeros(len(batch), max(lengths))
    for i, (tensor, _, _) in enumerate(batch):
        x[i, :lengths[i]], mask[i, :lengths[i]] = tensor, 1.0
    side = torch.stack([sample[1] for sample in batch])
    labels = torch.tensor([sample[2] for sample in batch])
    return x, mask, side, labels


class ShardDataset(Dataset):
    """Native-length (T, N, 10) float32 tensor + (2,) side features + label per sequence.

    participants: optional set of participant_ids to keep (train/val splits).
    augment: train-time augmentation flag — keep False for validation.
    """

    def __init__(self, tensors_dir, augment: bool = False,
                 aug_config: AugmentConfig | None = None, participants=None):
        tensors_dir = Path(tensors_dir)
        index = pd.read_csv(tensors_dir / "index.csv")
        self.landmark_version = index.landmark_version.iloc[0]
        self.frames, self.offsets = _load_shards(str(tensors_dir))
        n_nodes = LANDMARK_SETS[self.landmark_version].n_nodes
        assert self.frames.shape[1:] == (n_nodes, 4), (
            f"{self.frames.shape} does not match landmark_version={self.landmark_version}")
        assert len(index) == len(self.offsets) - 1, (
            "index.csv/shard mismatch — stale shards in out-dir?")
        index["row"] = np.arange(len(index))  # position before filtering = shard slot
        if participants is not None:
            # str-normalized: merged indexes mix "ac_<id>" strings with numeric PopSign
            # ids, and callers pass either type
            keep = {str(p) for p in participants}
            index = index[index.participant_id.astype(str).isin(keep)].reset_index(drop=True)
        self.index = index
        self.lengths = np.diff(self.offsets)[index.row.to_numpy()]
        self.augment = augment
        self.aug_config = aug_config or AugmentConfig()

    def __len__(self) -> int:
        return len(self.index)

    def stored(self, i: int) -> np.ndarray:
        """The raw (T, N, 4) float16 slice for dataset position i, unaugmented."""
        start = self.offsets[self.index.iloc[i]["row"]]
        return self.frames[start:start + self.lengths[i]]

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        arr = self.stored(i).astype(np.float32)
        side = side_features(row["duration_s"], row["peak_speed"])
        if self.augment:
            # torch seeds each DataLoader worker differently; numpy state would fork identically
            rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,)).item()))
            if rng.random() < self.aug_config.flip_p:
                arr = mirrored_stored(arr, self.landmark_version)
            arr, side = augmented(arr, side, self.aug_config, rng)
        tensor = with_derived_channels(arr[..., :3], arr[..., 3:], self.landmark_version)
        return torch.from_numpy(tensor), torch.from_numpy(side), int(row["canonical_label_id"])
