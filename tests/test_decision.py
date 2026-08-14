"""Decision-policy tests on a hand-built fake db — no model, no training."""

import json

import numpy as np
import pytest

from signisa.decision import DecisionConfig, Verdict, verify_attempt

DIM = 8
E = np.eye(DIM)


def unit(v: np.ndarray) -> list[float]:
    return (v / np.linalg.norm(v)).tolist()


def at_cosine(target: np.ndarray, ortho: np.ndarray, cos: float) -> np.ndarray:
    """Unit vector at an exact cosine to unit `target`, tilted toward unit `ortho`."""
    return cos * target + np.sqrt(1.0 - cos**2) * ortho


def fake_db() -> dict:
    # "mom" with confusable "dad" at cosine 0.8; "book" isolated; "bad" has inverted
    # thresholds (far5 < eer, the small-n artifact). E[7] stays unused by every
    # centroid so garbage vectors can be built orthogonal to all of them.
    dad = unit(0.8 * E[0] + 0.6 * E[1])
    sign = lambda centroid, confusables, eer=0.6, far5=0.8: {
        "centroid": centroid, "confusables": confusables,
        "eer_threshold": eer, "low_far_threshold": far5,
    }
    # "girl" has two in-db rivals (brother closer than aunt) plus an
    # out-of-curriculum rival "farm" living in confusable_centroids
    return {"signs": {
        "mom": sign(E[0].tolist(), ["dad"]),
        "dad": sign(dad, ["mom"]),
        "book": sign(E[2].tolist(), []),
        "bad": sign(E[3].tolist(), [], eer=0.7, far5=0.5),
        "girl": sign(E[4].tolist(), ["aunt", "brother", "farm"]),
        "aunt": sign(unit(0.8 * E[4] + 0.6 * E[5]), []),
        "brother": sign(unit(0.9 * E[4] + np.sqrt(1 - 0.81) * E[5]), []),
        "untrained": {"centroid": None, "confusables": [],
                      "eer_threshold": None, "low_far_threshold": None},
    }, "confusable_centroids": {
        "farm": unit(0.97 * E[4] + np.sqrt(1 - 0.97**2) * E[6]),
    }}


def test_clean_accept():
    v = verify_attempt(E[0], "mom", fake_db())
    assert v.accepted and v.reason == "ok"
    assert v.score == pytest.approx(1.0)
    assert v.best_confusable == "dad" and v.margin == pytest.approx(0.2)


def test_confusable_rejection_names_the_offender():
    db = fake_db()
    dad = np.array(db["signs"]["dad"]["centroid"])
    halfway = (E[0] + dad) / np.linalg.norm(E[0] + dad)
    v = verify_attempt(halfway, "mom", db)
    assert not v.accepted and v.reason == "confusable"
    assert v.best_confusable == "dad"
    assert v.margin < 0.05


def test_garbage_rejection():
    v = verify_attempt(E[7], "mom", fake_db())  # orthogonal to every centroid
    assert not v.accepted and v.reason == "not_signing"


def test_below_threshold_rejection():
    attempt = at_cosine(E[0], E[7], 0.5)  # above tau_bg 0.2, below eer 0.6; far from dad
    v = verify_attempt(attempt, "mom", fake_db())
    assert not v.accepted and v.reason == "inaccurate"
    assert v.score == pytest.approx(0.5)


def test_borderline_flips_with_user_level():
    attempt = at_cosine(E[2], E[7], 0.7)  # book: between eer 0.6 and far5 0.8
    db = fake_db()
    lenient = verify_attempt(attempt, "book", db, DecisionConfig(user_level=0.0))
    strict = verify_attempt(attempt, "book", db, DecisionConfig(user_level=1.0))
    assert lenient.accepted
    assert not strict.accepted and strict.reason == "inaccurate"
    assert lenient.threshold == pytest.approx(0.6)
    assert strict.threshold == pytest.approx(0.8)


def test_inverted_thresholds_clamp_strict_ward():
    v = verify_attempt(E[3], "bad", fake_db(), DecisionConfig(user_level=1.0))
    assert v.threshold_clamped
    assert v.threshold == pytest.approx(0.7)  # clamped to eer, never leveled leniant-ward
    assert v.accepted


def test_untrained_sign_raises():
    with pytest.raises(ValueError, match="no trained centroid"):
        verify_attempt(E[0], "untrained", fake_db())


def test_closest_of_several_rivals_wins_including_out_of_curriculum():
    # attempt = girl centroid: rivals score aunt 0.8 < brother 0.9 < farm 0.97;
    # farm comes from confusable_centroids and its margin 0.03 < 0.05 rejects
    v = verify_attempt(E[4], "girl", fake_db())
    assert not v.accepted and v.reason == "confusable"
    assert v.best_confusable == "farm"
    assert v.margin == pytest.approx(0.03)


def test_zero_or_nonfinite_embedding_raises():
    with pytest.raises(ValueError, match="invalid embedding"):
        verify_attempt(np.zeros(DIM), "mom", fake_db())
    with pytest.raises(ValueError, match="invalid embedding"):
        verify_attempt(np.full(DIM, np.nan), "mom", fake_db())


def test_user_level_is_clamped():
    attempt = at_cosine(E[2], E[7], 0.7)
    wild = verify_attempt(attempt, "book", fake_db(), DecisionConfig(user_level=5.0))
    assert wild.threshold == pytest.approx(0.8)  # clamped to level 1, not extrapolated


def test_verdict_json_round_trip():
    v = verify_attempt(E[0], "mom", fake_db())
    restored = Verdict(**json.loads(json.dumps(v.to_dict())))
    assert restored == v
