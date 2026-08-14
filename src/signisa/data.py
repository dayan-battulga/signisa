"""Torch Dataset over precomputed (160, 65, 4) float16 shards.

Stored channels are xyz + confidence; velocity and bone channels are re-derived
here (signisa.preprocess.pipeline.with_derived_channels) so the reconstructed
(160, 65, 10) float32 matches preprocess() output to float16 storage rounding
(measured max error ~4e-3 on real data). Augmentations run
BEFORE derivation so derived channels stay consistent. No horizontal flipping
ever — sequences are already in canonical right-dominant space.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import AugmentConfig
from .preprocess.landmarks import MIRROR_PERM
from .preprocess.pipeline import with_derived_channels


def augmented(coords: np.ndarray, confidence: np.ndarray, cfg: AugmentConfig,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Train-time augmentation of ((T,65,3) coords, (T,65,1) confidence), in place-ish."""
    t, n, _ = coords.shape
    present = confidence[..., 0] > 0

    # affine jitter about the origin (rotation in the canonical xy plane) + noise,
    # present nodes only — missing nodes stay at the origin with confidence 0
    theta = np.deg2rad(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scale = 1.0 + rng.uniform(-cfg.scale, cfg.scale)
    trans = rng.uniform(-cfg.translation, cfg.translation, size=3)
    jittered = (coords @ rot.T) * scale + trans
    jittered += rng.normal(0.0, cfg.noise_sigma, coords.shape)
    coords = np.where(present[..., None], jittered, coords)

    # node dropout: whole-sequence, like a landmark the tracker never found
    dropped = rng.random(n) < cfg.node_dropout_p
    coords[:, dropped] = 0.0
    confidence[:, dropped] = 0.0

    # temporal masking: random spans totaling up to mask_total_frac of frames
    target = rng.integers(0, int(cfg.mask_total_frac * t) + 1)
    masked = 0
    while masked < target:
        span = min(int(rng.integers(cfg.mask_span_min, cfg.mask_span_max + 1)),
                   int(target - masked))  # clamp so spans never exceed the budget
        start = int(rng.integers(0, t - span + 1))
        coords[start:start + span] = 0.0
        confidence[start:start + span] = 0.0
        masked += span
    return coords, confidence


def mirrored_stored(stored: np.ndarray) -> np.ndarray:
    """Flip a stored (T, 65, 4) tensor's orientation in canonical space.

    Exactly equivalent to having mirrored the raw sequence before preprocess
    (every pipeline step is mirror-equivariant; verified to zero error on real
    data — z keeps its sign because the canonical frame re-derives it as x*up).
    """
    out = stored[:, MIRROR_PERM].copy()
    out[..., 0] = -out[..., 0]
    return out


@lru_cache(maxsize=1)
def _load_shards(tensors_dir: str) -> np.ndarray:
    """All shards as one read-only array, shared by every ShardDataset over the dir.

    ponytail: whole dataset in RAM (~8 GB float16 for the full 94k; train/val/eval
    instances share ONE copy via this cache). Switch the build to raw .npy +
    mmap_mode='r' if a Kaggle instance can't hold it.
    """
    shards = sorted(Path(tensors_dir).glob("shard_*.npz"))
    tensors = np.concatenate([np.load(p)["tensors"] for p in shards])
    tensors.setflags(write=False)
    return tensors


class ShardDataset(Dataset):
    """(160, 65, 10) float32 tensor + canonical label id per sequence.

    participants: optional set of participant_ids to keep (train/val splits).
    augment: train-time augmentation flag — keep False for validation.
    """

    def __init__(self, tensors_dir, augment: bool = False,
                 aug_config: AugmentConfig | None = None, participants=None):
        tensors_dir = Path(tensors_dir)
        index = pd.read_csv(tensors_dir / "index.csv")
        self.tensors = _load_shards(str(tensors_dir))
        assert self.tensors.shape[1:] == (160, 65, 4), self.tensors.shape
        assert len(index) == len(self.tensors), "index.csv/shard mismatch — stale shards in out-dir?"
        index["row"] = np.arange(len(index))  # position before filtering = shard slot
        if participants is not None:
            index = index[index.participant_id.isin(set(participants))].reset_index(drop=True)
        self.index = index
        self.augment = augment
        self.aug_config = aug_config or AugmentConfig()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index.iloc[i]
        arr = self.tensors[row["row"]].astype(np.float32)
        coords, confidence = arr[..., :3].copy(), arr[..., 3:].copy()
        if self.augment:
            # torch seeds each DataLoader worker differently; numpy state would fork identically
            rng = np.random.default_rng(int(torch.randint(0, 2**31, (1,)).item()))
            coords, confidence = augmented(coords, confidence, self.aug_config, rng)
        tensor = with_derived_channels(coords, confidence)
        return torch.from_numpy(tensor), int(row["canonical_label_id"])
