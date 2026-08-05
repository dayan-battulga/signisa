"""65-node landmark set (research Task 2 spec) selected from MediaPipe Holistic's 543 landmarks.

Node layout in our 65-space:
    0-20   left hand (MediaPipe hand order: wrist, thumb chain, index, middle, ring, pinky)
    21-41  right hand
    42-48  body: nose, L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist
    49-64  face: brows (6), eye corners (4), mouth (6)

"L"/"R" are the subject's left/right. Face-mesh index sides follow MediaPipe's
canonical face model.
# TODO: visually verify the 16 face-mesh indices once real sample data is connected.
"""

import numpy as np

# Row offsets of each landmark group in the Kaggle asl-signs holistic layout.
FACE_OFFSET = 0
LEFT_HAND_OFFSET = 468
POSE_OFFSET = 489
RIGHT_HAND_OFFSET = 522
N_HOLISTIC = 543

N_NODES = 65
N_FEATURES = 10  # xyz(3) + velocity(3) + bone(3) + confidence(1)

_POSE_POINTS = [0, 11, 12, 13, 14, 15, 16]  # nose, L/R shoulder, L/R elbow, L/R wrist

_FACE_POINTS = [
    70, 63, 107,      # right brow: outer, mid, inner
    336, 293, 300,    # left brow: inner, mid, outer
    33, 133,          # right eye: outer, inner corner
    362, 263,         # left eye: inner, outer corner
    61, 291,          # mouth corners: right, left
    0, 13, 14, 17,    # lips: upper outer, upper inner, lower inner, lower outer
]

HOLISTIC_INDICES = np.array(
    [LEFT_HAND_OFFSET + i for i in range(21)]
    + [RIGHT_HAND_OFFSET + i for i in range(21)]
    + [POSE_OFFSET + i for i in _POSE_POINTS]
    + [FACE_OFFSET + i for i in _FACE_POINTS]
)

# Named node ids in 65-space.
NOSE = 42
L_SHOULDER, R_SHOULDER = 43, 44
L_ELBOW, R_ELBOW = 45, 46
L_WRIST, R_WRIST = 47, 48

# Kinematic parent of each node; bone vector = node - parent. Root (nose) is its own parent.
_HAND_PARENT = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

PARENT = np.array(
    [L_WRIST if i == 0 else _HAND_PARENT[i] for i in range(21)]              # left hand -> body L wrist
    + [R_WRIST if i == 0 else 21 + _HAND_PARENT[i] for i in range(21)]       # right hand -> body R wrist
    + [NOSE, NOSE, NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW]           # body chain
    + [NOSE] * 16                                                            # face -> nose
)

def _mirror_permutation() -> np.ndarray:
    perm = np.arange(N_NODES)
    perm[0:21], perm[21:42] = np.arange(21, 42), np.arange(0, 21)  # swap hands
    for left, right in [(L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
                        (49, 54), (50, 53), (51, 52),   # brows
                        (55, 58), (56, 57),             # eye corners
                        (59, 60)]:                      # mouth corners
        perm[left], perm[right] = right, left
    return perm

MIRROR_PERM = _mirror_permutation()
