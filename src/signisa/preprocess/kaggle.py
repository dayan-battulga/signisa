"""Loader for Kaggle asl-signs landmark parquet files."""

import numpy as np
import pandas as pd

from .landmarks import (
    FACE_OFFSET,
    LEFT_HAND_OFFSET,
    N_HOLISTIC,
    POSE_OFFSET,
    RIGHT_HAND_OFFSET,
)

_TYPE_OFFSET = {
    "face": FACE_OFFSET,
    "left_hand": LEFT_HAND_OFFSET,
    "pose": POSE_OFFSET,
    "right_hand": RIGHT_HAND_OFFSET,
}

def load_holistic(parquet_path) -> np.ndarray:
    """Read one sequence into a (T, 543, 3) float32 array with NaN for missing landmarks."""
    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    frame_ids = np.sort(df["frame"].unique())
    frame_pos = {f: i for i, f in enumerate(frame_ids)}

    holistic = np.full((len(frame_ids), N_HOLISTIC, 3), np.nan, dtype=np.float32)
    rows = df["frame"].map(frame_pos).to_numpy()
    cols = df["type"].map(_TYPE_OFFSET).to_numpy() + df["landmark_index"].to_numpy()
    holistic[rows, cols] = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    return holistic
