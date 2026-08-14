"""Decision policy (task3b Part 2): threshold interpolated by user level, garbage
gate, margin-over-confusables. Operates on a curriculum_db dict as written by
signisa.eval (centroids + eer/low-FAR thresholds filled after Phase 1 training).

Pure numpy — no torch, no model. The embedding comes in already computed.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DecisionConfig:
    user_level: float = 0.0    # 0 = beginner (lenient EER point) .. 1 = strict (low-FAR point)
    margin_delta: float = 0.05  # required score margin over every confusable centroid (task3b)
    tau_bg: float = 0.2        # garbage gate on max centroid cosine; energy-style placeholder
    #                            until Phase 4 open-set calibration fills weibull_params


@dataclass
class Verdict:
    accepted: bool
    reason: str                # "ok" | "not_signing" | "inaccurate" | "confusable"
    score: float               # cosine(embedding, target centroid)
    threshold: float           # the interpolated per-sign threshold actually applied
    best_confusable: str | None = None  # closest rival centroid (the offender on rejection)
    margin: float | None = None         # score minus that rival's cosine
    threshold_clamped: bool = False     # far5 < eer inverted at small n; clamped stricter-ward

    def to_dict(self) -> dict:
        return asdict(self)


def verify_attempt(embedding: np.ndarray, target_gloss: str, db: dict,
                   config: DecisionConfig = DecisionConfig()) -> Verdict:
    """task3b decision chain: garbage gate -> per-sign threshold -> margin-over-confusables."""
    target = db["signs"][target_gloss]
    if target["centroid"] is None or target["eer_threshold"] is None:
        raise ValueError(f"{target_gloss} has no trained centroid/thresholds — "
                         "use a curriculum_db_trained.json from signisa.eval")
    embedding = np.asarray(embedding, dtype=np.float64)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm < 1e-8:
        # NaN scores would sail through every `<` reject gate straight to accept
        raise ValueError("invalid embedding: zero or non-finite")
    embedding = embedding / norm

    centroids = {gloss: np.asarray(entry["centroid"])
                 for gloss, entry in db["signs"].items() if entry["centroid"] is not None}
    score = float(embedding @ centroids[target_gloss])

    eer, far5 = target["eer_threshold"], target["low_far_threshold"]
    clamped = far5 < eer  # ordering can invert at small n; clamp to the stricter point
    far5 = max(far5, eer)
    level = float(np.clip(config.user_level, 0.0, 1.0))
    threshold = eer + level * (far5 - eer)

    # rivals: confusables of the target — curriculum centroids plus the
    # out-of-curriculum ones eval writes under "confusable_centroids"
    rival_centroids = {**{g: np.asarray(c) for g, c in db.get("confusable_centroids", {}).items()},
                       **centroids}
    rival_scores = {gloss: float(embedding @ rival_centroids[gloss])
                    for gloss in target["confusables"] if gloss in rival_centroids}
    best_confusable = max(rival_scores, key=rival_scores.get) if rival_scores else None
    margin = score - rival_scores[best_confusable] if best_confusable else None

    def verdict(accepted: bool, reason: str) -> Verdict:
        return Verdict(accepted=accepted, reason=reason, score=score, threshold=threshold,
                       best_confusable=best_confusable, margin=margin,
                       threshold_clamped=clamped)

    if max(float(embedding @ c) for c in centroids.values()) < config.tau_bg:
        return verdict(False, "not_signing")
    if score < threshold:
        return verdict(False, "inaccurate")
    if margin is not None and margin < config.margin_delta:
        return verdict(False, "confusable")
    return verdict(True, "ok")
