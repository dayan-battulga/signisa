import numpy as np

from signisa.preprocess.dominance import (
    dominance_from_wrists,
    hand_dominance,
    majority_dominance,
)
from signisa.preprocess.kaggle import load_wrists
from signisa.preprocess.landmarks import (
    FACE_OFFSET,
    LEFT_HAND_OFFSET,
    N_HOLISTIC,
    POSE_OFFSET,
    RIGHT_HAND_OFFSET,
)
from signisa.preprocess.pipeline import preprocess

RNG = np.random.default_rng(11)

# 543-space mirror pairs for the indices the pipeline actually selects.
_POSE_PAIRS = [(11, 12), (13, 14), (15, 16)]
_FACE_PAIRS = [(70, 300), (63, 293), (107, 336), (33, 263), (133, 362), (61, 291)]


def mirrored_holistic(holistic: np.ndarray) -> np.ndarray:
    """Manually built mirror twin: swap hand blocks + L/R pose/face points, negate x."""
    out = holistic.copy()
    out[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 21] = holistic[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 21]
    out[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 21] = holistic[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 21]
    for a, b in _POSE_PAIRS:
        out[:, POSE_OFFSET + a] = holistic[:, POSE_OFFSET + b]
        out[:, POSE_OFFSET + b] = holistic[:, POSE_OFFSET + a]
    for a, b in _FACE_PAIRS:
        out[:, FACE_OFFSET + a] = holistic[:, FACE_OFFSET + b]
        out[:, FACE_OFFSET + b] = holistic[:, FACE_OFFSET + a]
    out[..., 0] = -out[..., 0]
    return out


def left_handed_holistic(t: int = 40) -> np.ndarray:
    """Left hand moving, right hand absent, face/pose asymmetric enough to be real."""
    base = RNG.uniform(0.2, 0.8, size=(1, N_HOLISTIC, 3))
    drift = np.cumsum(RNG.normal(0, 0.004, size=(t, N_HOLISTIC, 3)), axis=0)
    holistic = base + drift
    holistic[:, POSE_OFFSET + 11] = [0.40, 0.45, 0.0] + drift[:, POSE_OFFSET + 11]
    holistic[:, POSE_OFFSET + 12] = [0.60, 0.45, 0.0] + drift[:, POSE_OFFSET + 12]
    holistic[:, POSE_OFFSET + 0] = [0.50, 0.25, 0.0] + drift[:, POSE_OFFSET + 0]
    sweep = np.linspace(0.0, 0.3, t)[:, None]
    holistic[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 21, 0] += sweep
    holistic[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 21] = np.nan
    return holistic


def test_left_sequence_detected_and_mirror_twin_matches():
    lefty = left_handed_holistic()
    assert hand_dominance(lefty) == "left"
    righty = mirrored_holistic(lefty)
    assert hand_dominance(righty) == "right"
    np.testing.assert_allclose(
        preprocess(lefty, left_dominant=True).tensor,
        preprocess(righty).tensor,
        atol=1e-5,
    )


def test_symmetric_two_handed_motion_is_ambiguous():
    holistic = left_handed_holistic()
    # copy the left hand onto the right, reflected: same presence, same speed
    lh = holistic[:, LEFT_HAND_OFFSET:LEFT_HAND_OFFSET + 21]
    holistic[:, RIGHT_HAND_OFFSET:RIGHT_HAND_OFFSET + 21] = lh * [-1, 1, 1] + [1, 0, 0]
    assert hand_dominance(holistic) == "ambiguous"


def test_dominance_edge_cases():
    t = np.linspace(0, 1, 30)[:, None]
    moving = np.hstack([t, t, t])
    still = np.full((30, 3), 0.5)
    absent = np.full((30, 3), np.nan)
    assert dominance_from_wrists(moving, absent) == "left"
    assert dominance_from_wrists(absent, moving) == "right"
    assert dominance_from_wrists(absent, absent) == "ambiguous"
    assert dominance_from_wrists(still, still) == "ambiguous"  # present but static


def test_majority_dominance():
    assert majority_dominance(["left", "left", "right", "ambiguous"]) == "left"
    assert majority_dominance(["left", "right"]) == "right"  # tie -> canonical
    assert majority_dominance([]) == "right"
    assert majority_dominance(["ambiguous", "ambiguous"]) == "right"


def test_load_wrists_matches_holistic(tmp_path):
    import pandas as pd

    from signisa.preprocess.kaggle import load_holistic

    rows = []
    for frame in (0, 1, 2):
        for kind, offset in [("face", 0), ("left_hand", 468), ("pose", 489), ("right_hand", 522)]:
            n = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}[kind]
            for idx in range(n):
                missing = kind == "right_hand" and frame == 1  # real data NaNs all coords
                rows.append({"frame": frame, "type": kind, "landmark_index": idx,
                             "x": np.nan if missing else 0.1 * frame + 0.01 * idx,
                             "y": np.nan if missing else 0.2,
                             "z": np.nan if missing else 0.3})
    path = tmp_path / "seq.parquet"
    pd.DataFrame(rows).to_parquet(path)

    left, right = load_wrists(path)
    holistic = load_holistic(path)
    np.testing.assert_allclose(left, holistic[:, 468], atol=1e-6)
    np.testing.assert_allclose(right, holistic[:, 522], atol=1e-6, equal_nan=True)
    assert np.isnan(right[1]).all()
