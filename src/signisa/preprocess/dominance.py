"""Dominant-hand detection for canonical mirroring (CLAUDE.md locked decision).

Per-hand score = fraction-of-frames-present x mean wrist speed. Handedness is a
person trait: training data resolves it per participant by majority vote
(majority_dominance); per-sequence hand_dominance is the fallback for unseen
participants at inference.
"""

from collections.abc import Iterable

import numpy as np

from .landmarks import LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET

# Higher score must beat lower by 1.2x, else "ambiguous" — hysteresis so
# symmetric two-handed signs never flip on noise.
HYSTERESIS = 1.2


def wrist_score(wrist: np.ndarray, fps: float) -> float:
    """Presence-weighted mean speed of one (T, 3) wrist track; 0.0 if never seen."""
    present = ~np.isnan(wrist).any(axis=1)
    if not present.any():
        return 0.0
    steps = np.linalg.norm(np.diff(wrist, axis=0), axis=1)  # NaN when either frame missing
    mean_speed = 0.0 if np.isnan(steps).all() else float(np.nanmean(steps)) * fps
    return float(present.mean()) * mean_speed


def dominance_from_wrists(left_wrist: np.ndarray, right_wrist: np.ndarray,
                          fps: float = 30.0) -> str:
    """'left' / 'right' / 'ambiguous' from the two (T, 3) hand-wrist tracks."""
    left, right = wrist_score(left_wrist, fps), wrist_score(right_wrist, fps)
    higher, lower = max(left, right), min(left, right)
    if higher == 0.0 or higher < HYSTERESIS * lower:
        return "ambiguous"
    return "left" if left > right else "right"


def hand_dominance(holistic: np.ndarray, fps: float = 30.0) -> str:
    """Per-sequence dominance for one raw (T, 543, 3) holistic array."""
    return dominance_from_wrists(
        holistic[:, LEFT_HAND_OFFSET], holistic[:, RIGHT_HAND_OFFSET], fps)


def majority_dominance(votes: Iterable[str]) -> str:
    """Participant handedness from per-sequence votes; ambiguous ignored, ties -> 'right'."""
    tally = sum((v == "left") - (v == "right") for v in votes)
    return "left" if tally > 0 else "right"
