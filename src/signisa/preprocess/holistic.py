"""MediaPipe Tasks HolisticLandmarker -> Kaggle-layout (543, 3) rows.

The legacy mp.solutions.holistic API — the extractor that produced the Kaggle
asl-signs landmarks — no longer exists in any mediapipe build installable today;
only the Tasks API remains, backed by a DIFFERENT model. That is a recorded
extractor seam, not a silent one: extraction outputs carry mediapipe.__version__,
merged tensors carry a domain column, and the per-domain eval section is the
instrument that measures any popsign/asl_citizen extractor shift. The Tasks API
is also what a future browser/live tier runs, so new extractions match
deployment provenance.

mediapipe/cv2 are the [live] extra and imported lazily — everything importing
this module stays usable without them until a landmarker is actually created.
"""

from pathlib import Path

import numpy as np

from .landmarks import (
    FACE_OFFSET,
    LEFT_HAND_OFFSET,
    N_HOLISTIC,
    POSE_OFFSET,
    RIGHT_HAND_OFFSET,
)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
DEFAULT_MODEL = Path("data/models/holistic_landmarker.task")
MAX_SIDE = 640  # downscale before MediaPipe: landmarks are normalized, so only speed changes.
# (The legacy API's model_complexity knob does not exist in the Tasks API — the .task
# bundle bakes its models in; input resolution is the remaining speed lever.)


def downscaled(frame: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Shrink so the long side is <= max_side; no-op when already small or max_side <= 0.

    ponytail: measured on a 720p smoke clip, 640 is ~1.25x faster but detected 38%
    fewer frames (small far-away hands lose crop pixels). Fine for ASL Citizen's
    centered webcam framing in principle — A/B n_detected_frames on ~50 real clips
    (--max-side 0 vs 640) before committing a full extraction.
    """
    import cv2

    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    return cv2.resize(frame, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def create_landmarker(model_path=DEFAULT_MODEL):
    """A VIDEO-mode HolisticLandmarker; one per process, close() when done."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"{model_path} missing — download the model bundle first:\n"
            f"  curl -L --create-dirs -o {model_path} {MODEL_URL}")
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO))


def detect_frame(landmarker, rgb: np.ndarray, timestamp_ms: int):
    """Run one RGB frame; timestamps must strictly increase within a landmarker."""
    import mediapipe as mp

    return landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)


def holistic_row(result) -> np.ndarray:
    """One HolisticLandmarker result -> (543, 3) Kaggle-layout row, NaN when missing.

    The Tasks face mesh has 478 points (468 mesh + 10 iris); the Kaggle layout
    stores the first 468, which are the legacy mesh in the same order.
    """
    row = np.full((N_HOLISTIC, 3), np.nan, dtype=np.float32)
    for landmarks, offset, n in [
        (result.face_landmarks, FACE_OFFSET, 468),
        (result.left_hand_landmarks, LEFT_HAND_OFFSET, 21),
        (result.pose_landmarks, POSE_OFFSET, 33),
        (result.right_hand_landmarks, RIGHT_HAND_OFFSET, 21),
    ]:
        if landmarks:
            row[offset:offset + n] = [[p.x, p.y, p.z] for p in landmarks[:n]]
    return row


def extract_video(video_path, landmarker, max_side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    """One video file -> ((T, 543, 3) float32, fps). Raises on an unreadable file."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("cv2 cannot open the file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or not np.isfinite(fps):
        fps = 30.0  # some containers report 0 — assume the webcam default
    rows = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(downscaled(frame, max_side), cv2.COLOR_BGR2RGB)
            result = detect_frame(landmarker, rgb, int(round(len(rows) * 1000.0 / fps)))
            rows.append(holistic_row(result))
    finally:
        cap.release()
    if not rows:
        raise ValueError("no decodable frames")
    return np.stack(rows), float(fps)
