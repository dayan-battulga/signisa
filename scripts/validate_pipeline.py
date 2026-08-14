"""Validate the preprocessing pipeline against real Kaggle asl-signs parquets (backlog 0.3).

v1-only: the node-group slices and face-geometry ranges hardcode the 65-node
layout. v2 lip geometry is covered by tests/test_training.py instead.

Usage: python scripts/validate_pipeline.py [samples_dir] [train_csv]
Defaults: data/samples data/meta/train.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from signisa.preprocess.kaggle import load_holistic
from signisa.preprocess.pipeline import preprocess

GROUPS = {"left_hand": slice(0, 21), "right_hand": slice(21, 42),
          "body": slice(42, 49), "face": slice(49, 65)}
# 65-space ids of the raw face selection, for the geometry check.
BROWS, EYES, MOUTH = range(49, 55), range(55, 59), range(59, 65)
MIRROR_FACE_PAIRS = [(49, 54), (50, 53), (51, 52), (55, 58), (56, 57), (59, 60)]
NOSE = 42


def main() -> None:
    samples_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples")
    train_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/meta/train.csv")
    paths = sorted(samples_dir.glob("*.parquet"))
    if not paths:
        sys.exit(f"no parquets in {samples_dir}")
    sign_of, participant_of = {}, {}
    if train_csv.exists():
        meta = pd.read_csv(train_csv)
        sign_of = dict(zip(meta.sequence_id.astype(str), meta.sign))
        participant_of = dict(zip(meta.sequence_id.astype(str), meta.participant_id))

    from signisa.preprocess.dominance import hand_dominance, majority_dominance
    from signisa.preprocess.pipeline import select_nodes

    # Handedness is a person trait: per-sequence votes -> per-participant majority.
    holistics = {p: load_holistic(p) for p in paths}
    votes = {p: hand_dominance(h) for p, h in holistics.items()}
    by_participant: dict = {}
    for p in paths:
        by_participant.setdefault(participant_of.get(p.stem, p.stem), []).append(votes[p])
    left_participants = {pid for pid, vs in by_participant.items()
                         if majority_dominance(vs) == "left"}

    face_frames = []  # raw selected face rows for the geometry check
    n_mirrored = 0
    print(f"{'sequence':>12} {'sign':>12} {'T_raw':>5} {'shape':>13} "
          f"{'LH%0':>5} {'RH%0':>5} {'body%0':>6} {'face%0':>6} {'dur_s':>6} {'peak':>6} "
          f"{'dom':>9} {'mir':>3}")
    for path in paths:
        holistic = holistics[path]
        mirror = participant_of.get(path.stem, path.stem) in left_participants
        n_mirrored += mirror
        result = preprocess(holistic, left_dominant=mirror)
        assert np.isfinite(result.tensor).all(), f"NaN/inf in output for {path.name}"
        conf = result.tensor[..., 9]
        # 1 - mean confidence, not (conf == 0): resampling smears gap boundaries into
        # fractional values, so exact-zero counting undercounts missing spans.
        zeros = {name: 100.0 * (1.0 - conf[:, sl].mean()) for name, sl in GROUPS.items()}
        sign = sign_of.get(path.stem, "?")
        print(f"{path.stem:>12} {sign:>12} {holistic.shape[0]:>5} {str(result.tensor.shape):>13} "
              f"{zeros['left_hand']:>5.1f} {zeros['right_hand']:>5.1f} {zeros['body']:>6.1f} "
              f"{zeros['face']:>6.1f} {result.duration_s:>6.2f} {result.peak_speed:>6.2f} "
              f"{votes[path]:>9} {'y' if mirror else '':>3}")
        sel = select_nodes(holistic)
        face_nodes = [NOSE] + list(BROWS) + list(EYES) + list(MOUTH)  # nose too: geometry check reads it
        face_frames.append(sel[~np.isnan(sel[:, face_nodes]).any(axis=(1, 2))])

    check_face_geometry(np.concatenate(face_frames))
    print(f"mirroring: {len(left_participants)}/{len(by_participant)} participants left-dominant "
          f"-> {n_mirrored}/{len(paths)} sequences mirrored")
    print("OK: all outputs NaN-free")


def check_face_geometry(frames: np.ndarray) -> None:
    """Empirically verify the 16 face-mesh indices on raw (unnormalized) landmarks.

    MediaPipe y increases downward: brows must sit above eyes, eyes above mouth.
    Mirror pairs must straddle the nose x on opposite sides.
    """
    if frames.size == 0:
        print("WARN: no frames with a full face — geometry check skipped")
        return
    brow_y = frames[:, BROWS, 1].mean()
    eye_y = frames[:, EYES, 1].mean()
    mouth_y = frames[:, MOUTH, 1].mean()
    assert brow_y < eye_y < mouth_y, f"vertical order wrong: brows {brow_y:.3f}, eyes {eye_y:.3f}, mouth {mouth_y:.3f}"
    nose_x = frames[:, NOSE, 0]
    for a, b in MIRROR_FACE_PAIRS:
        da = (frames[:, a, 0] - nose_x).mean()
        db = (frames[:, b, 0] - nose_x).mean()
        assert da * db < 0, f"mirror pair ({a},{b}) on same side of nose: {da:.3f}, {db:.3f}"
    print(f"face geometry OK: brow_y {brow_y:.3f} < eye_y {eye_y:.3f} < mouth_y {mouth_y:.3f}; "
          f"all 6 mirror pairs straddle the nose")


if __name__ == "__main__":
    main()
