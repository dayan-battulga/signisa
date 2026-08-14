"""Landmark sets selected from MediaPipe Holistic's 543 landmarks.

v1 (65 nodes): 21 left hand + 21 right hand + 7 body + 16 face
    (6 brows, 4 eye corners, 2 mouth corners + 4 midline lips)
v2 (99 nodes): same hands/body/brows/eyes + the standard 40-point MediaPipe lip
    set (outer + inner rings) — task3a: Kaggle winners carried 18-40 lip points
    because PopSign players mouth the words.

Layout (both versions): 0-20 left hand, 21-41 right hand, 42-48 body
(nose, L/R shoulder, L/R elbow, L/R wrist), 49-54 brows, 55-58 eye corners,
then the version's lip tail. "L"/"R" are the subject's left/right.
"""

from dataclasses import dataclass, field

import numpy as np

# Row offsets of each landmark group in the Kaggle asl-signs holistic layout.
FACE_OFFSET = 0
LEFT_HAND_OFFSET = 468
POSE_OFFSET = 489
RIGHT_HAND_OFFSET = 522
N_HOLISTIC = 543

N_FEATURES = 10  # xyz(3) + velocity(3) + bone(3) + confidence(1)

_POSE_POINTS = [0, 11, 12, 13, 14, 15, 16]  # nose, L/R shoulder, L/R elbow, L/R wrist

_BROWS = [70, 63, 107, 336, 293, 300]  # right outer/mid/inner, left inner/mid/outer
_EYES = [33, 133, 362, 263]            # right outer/inner, left inner/outer corners
_MOUTH_V1 = [61, 291, 0, 13, 14, 17]   # corners + midline lips
# Standard MediaPipe FACEMESH_LIPS 40-point set: outer ring then inner ring.
_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
               291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
               308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Subject right <-> left mirror pairs of the canonical face model, for every face
# point either version selects. Midline points (0, 13, 14, 17) are self-paired.
_FACE_MIRROR_PAIRS = [
    (70, 300), (63, 293), (107, 336),                       # brows
    (33, 263), (133, 362),                                  # eye corners
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),   # outer lip
    (37, 267), (39, 269), (40, 270), (185, 409),
    (78, 308), (95, 324), (88, 318), (178, 402), (87, 317),    # inner lip
    (82, 312), (81, 311), (80, 310), (191, 415),
]
_FACE_MIRROR = {a: b for a, b in _FACE_MIRROR_PAIRS} | {b: a for a, b in _FACE_MIRROR_PAIRS}
_FACE_MIDLINE = {0, 13, 14, 17}  # the only face points allowed to self-pair

# Named node ids in selection space — identical in both versions.
NOSE = 42
L_SHOULDER, R_SHOULDER = 43, 44
L_ELBOW, R_ELBOW = 45, 46
L_WRIST, R_WRIST = 47, 48
_FACE_BASE = 49

# Kinematic parent of each hand node; bone vector = node - parent.
_HAND_PARENT = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
_POSE_PAIRS = [(L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST)]


@dataclass(frozen=True)
class LandmarkSet:
    version: str
    face_points: tuple[int, ...]
    n_nodes: int = field(init=False)
    holistic_indices: np.ndarray = field(init=False)
    parent: np.ndarray = field(init=False)
    mirror_perm: np.ndarray = field(init=False)

    def __post_init__(self):
        n = _FACE_BASE + len(self.face_points)
        indices = np.array(
            [LEFT_HAND_OFFSET + i for i in range(21)]
            + [RIGHT_HAND_OFFSET + i for i in range(21)]
            + [POSE_OFFSET + i for i in _POSE_POINTS]
            + [FACE_OFFSET + i for i in self.face_points])
        parent = np.array(
            [L_WRIST if i == 0 else _HAND_PARENT[i] for i in range(21)]
            + [R_WRIST if i == 0 else 21 + _HAND_PARENT[i] for i in range(21)]
            + [NOSE, NOSE, NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW]
            + [NOSE] * len(self.face_points))
        perm = np.arange(n)
        perm[0:21], perm[21:42] = np.arange(21, 42), np.arange(0, 21)  # swap hands
        for left, right in _POSE_PAIRS:
            perm[left], perm[right] = right, left
        for pos, fp in enumerate(self.face_points):
            assert fp in _FACE_MIRROR or fp in _FACE_MIDLINE, (
                f"face point {fp} has no mirror pair and is not declared midline")
            partner = _FACE_MIRROR.get(fp, fp)  # midline points are their own partner
            perm[_FACE_BASE + pos] = _FACE_BASE + self.face_points.index(partner)
        for name, value in [("n_nodes", n), ("holistic_indices", indices),
                            ("parent", parent), ("mirror_perm", perm)]:
            object.__setattr__(self, name, value)


LANDMARK_SETS = {
    "v1": LandmarkSet("v1", tuple(_BROWS + _EYES + _MOUTH_V1)),
    "v2": LandmarkSet("v2", tuple(_BROWS + _EYES + _LIPS_OUTER + _LIPS_INNER)),
}
LANDMARK_VERSION = "v1"  # default for existing artifacts; v2 is the Phase 1b lever

# v1 aliases — the pre-versioning module API, still used throughout.
N_NODES = LANDMARK_SETS["v1"].n_nodes
HOLISTIC_INDICES = LANDMARK_SETS["v1"].holistic_indices
PARENT = LANDMARK_SETS["v1"].parent
MIRROR_PERM = LANDMARK_SETS["v1"].mirror_perm
