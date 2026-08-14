import numpy as np
import pytest

from signisa.preprocess.kaggle import load_holistic
from signisa.preprocess.landmarks import (
    HOLISTIC_INDICES,
    MIRROR_PERM,
    N_HOLISTIC,
    N_NODES,
)
from signisa.preprocess.pipeline import T_OUT, mirrored, preprocess

RNG = np.random.default_rng(7)


def synthetic_holistic(t: int = 48) -> np.ndarray:
    """Smooth random-walk landmarks with plausible shoulder geometry."""
    base = RNG.uniform(0.2, 0.8, size=(1, N_HOLISTIC, 3))
    drift = np.cumsum(RNG.normal(0, 0.004, size=(t, N_HOLISTIC, 3)), axis=0)
    holistic = (base + drift).astype(np.float64)
    holistic[:, 489 + 11] = [0.40, 0.45, 0.0] + drift[:, 489 + 11]  # L shoulder
    holistic[:, 489 + 12] = [0.60, 0.45, 0.0] + drift[:, 489 + 12]  # R shoulder
    holistic[:, 489 + 0] = [0.50, 0.25, 0.0] + drift[:, 489 + 0]    # nose (above shoulders)
    return holistic


def test_output_shape_dtype_and_no_nans():
    result = preprocess(synthetic_holistic())
    assert result.tensor.shape == (T_OUT, N_NODES, 10)
    assert result.tensor.dtype == np.float32
    assert np.isfinite(result.tensor).all()


def test_translation_invariance():
    holistic = synthetic_holistic()
    shifted = holistic + np.array([0.31, -0.12, 0.05])
    np.testing.assert_allclose(
        preprocess(holistic).tensor, preprocess(shifted).tensor, atol=1e-5
    )


def test_scale_invariance():
    holistic = synthetic_holistic()
    np.testing.assert_allclose(
        preprocess(holistic).tensor, preprocess(holistic * 2.7).tensor, atol=1e-5
    )


def test_roll_rotation_invariance():
    holistic = synthetic_holistic()
    theta = np.deg2rad(25)
    roll = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    )
    rotated = holistic @ roll.T
    np.testing.assert_allclose(
        preprocess(holistic).tensor, preprocess(rotated).tensor, atol=1e-4
    )


def test_static_input_has_zero_motion():
    holistic = np.repeat(synthetic_holistic(1), 40, axis=0)
    result = preprocess(holistic)
    assert result.peak_speed == pytest.approx(0.0, abs=1e-6)
    velocity = result.tensor[..., 3:6]
    assert np.abs(velocity).max() == pytest.approx(0.0, abs=1e-5)


def test_fully_missing_hand_becomes_zero_confidence():
    holistic = synthetic_holistic()
    holistic[:, 468:489] = np.nan  # left hand missing for the whole attempt
    tensor = preprocess(holistic).tensor
    left_hand = tensor[:, 0:21]
    assert np.abs(left_hand[..., 9]).max() == pytest.approx(0.0)   # confidence channel
    assert np.abs(left_hand[..., 0:3]).max() == pytest.approx(0.0)  # coords at origin
    assert np.isfinite(tensor).all()


def test_short_gap_is_recovered():
    holistic = synthetic_holistic()
    holistic[20:22, 468:489] = np.nan  # 2-frame occlusion
    tensor = preprocess(holistic).tensor
    assert tensor[:, 0:21, 9].min() > 0.5  # confidence stays high after gap fill
    assert np.isfinite(tensor).all()


def test_mirror_is_involution_and_swaps_hands():
    seq = RNG.normal(size=(5, N_NODES, 3))
    np.testing.assert_allclose(mirrored(mirrored(seq)), seq, atol=1e-12)
    marker = seq[0, 3]  # a left-hand node
    np.testing.assert_allclose(mirrored(seq)[0, 24, 1:], marker[1:])  # lands in right hand, y/z kept
    assert mirrored(seq)[0, 24, 0] == pytest.approx(-marker[0])       # x negated


def test_mirror_permutation_is_valid():
    assert sorted(MIRROR_PERM.tolist()) == list(range(N_NODES))
    assert len(np.unique(HOLISTIC_INDICES)) == N_NODES


def test_landmark_sets_v1_v2():
    from signisa.preprocess.landmarks import LANDMARK_SETS

    v1, v2 = LANDMARK_SETS["v1"], LANDMARK_SETS["v2"]
    assert v1.n_nodes == 65 and v2.n_nodes == 99
    assert (v1.mirror_perm == MIRROR_PERM).all()  # v1 unchanged by the versioning
    for s in (v1, v2):
        assert sorted(s.mirror_perm.tolist()) == list(range(s.n_nodes))
        assert (s.mirror_perm[s.mirror_perm] == np.arange(s.n_nodes)).all()  # involution
        assert len(np.unique(s.holistic_indices)) == s.n_nodes
        assert (s.parent[49:] == 42).all()  # every face node hangs off the nose
    assert (v2.holistic_indices[:59] == v1.holistic_indices[:59]).all()  # shared prefix


def test_v2_preprocess_shape():
    result = preprocess(synthetic_holistic(), version="v2")
    assert result.tensor.shape == (T_OUT, 99, 10)
    assert np.isfinite(result.tensor).all()


def test_kaggle_loader_round_trip(tmp_path):
    import pandas as pd

    frames, rows = [0, 0, 2], []
    entries = [(0, "face", 5), (0, "left_hand", 0), (2, "pose", 11)]
    for frame, kind, idx in entries:
        rows.append(
            {"frame": frame, "type": kind, "landmark_index": idx, "x": 0.1, "y": 0.2, "z": 0.3}
        )
    path = tmp_path / "seq.parquet"
    pd.DataFrame(rows).to_parquet(path)

    holistic = load_holistic(path)
    assert holistic.shape == (2, N_HOLISTIC, 3)  # frames 0 and 2 -> two rows
    np.testing.assert_allclose(holistic[0, 5], [0.1, 0.2, 0.3], atol=1e-6)      # face + 5
    np.testing.assert_allclose(holistic[0, 468], [0.1, 0.2, 0.3], atol=1e-6)    # left hand + 0
    np.testing.assert_allclose(holistic[1, 489 + 11], [0.1, 0.2, 0.3], atol=1e-6)
    assert np.isnan(holistic[1, 0]).all()


def test_random_nan_patches_never_leak():
    holistic = synthetic_holistic(60)
    for _ in range(30):
        t0 = RNG.integers(0, 55)
        node = RNG.integers(0, N_HOLISTIC)
        holistic[t0 : t0 + RNG.integers(1, 8), node] = np.nan
    assert np.isfinite(preprocess(holistic).tensor).all()
