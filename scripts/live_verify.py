"""Live webcam verdict loop: countdown -> record -> MediaPipe Holistic ->
preprocess -> embedding -> decision policy. Signisa's first real learner data:
--save-dir persists the landmark array + verdict JSON per attempt (Phase 2 seed).

Usage:
    python scripts/live_verify.py --target mom --db curriculum_db_trained.json \
        --checkpoint model.pt [--save-dir data/attempts] [--seconds 2.5] [--camera 0]
    --smoke <parquet> runs the identical post-capture path headless (no camera,
    no mediapipe needed); --untrained swaps in a random-init model.

Needs the [live] extra for the camera path: pip install -e ".[live]"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from signisa.config import Config
from signisa.decision import DecisionConfig, verify_attempt
from signisa.models import SignModel, load_checkpoint
from signisa.preprocess.dominance import hand_dominance
from signisa.preprocess.kaggle import load_holistic
from signisa.preprocess.landmarks import (
    FACE_OFFSET,
    LEFT_HAND_OFFSET,
    N_HOLISTIC,
    POSE_OFFSET,
    RIGHT_HAND_OFFSET,
)
from signisa.preprocess.pipeline import preprocess


def holistic_row(results) -> np.ndarray:
    """One MediaPipe Holistic result -> (543, 3) in the Kaggle layout, NaN when missing."""
    row = np.full((N_HOLISTIC, 3), np.nan, dtype=np.float32)
    for landmarks, offset, n in [
        (results.face_landmarks, FACE_OFFSET, 468),
        (results.left_hand_landmarks, LEFT_HAND_OFFSET, 21),
        (results.pose_landmarks, POSE_OFFSET, 33),
        (results.right_hand_landmarks, RIGHT_HAND_OFFSET, 21),
    ]:
        if landmarks is not None:
            row[offset:offset + n] = [[p.x, p.y, p.z] for p in landmarks.landmark[:n]]
    return row


def capture_attempt(camera: int, seconds: float) -> tuple[np.ndarray, float]:
    """3-2-1 countdown, then record ~seconds of landmarks. Returns (holistic, measured fps)."""
    import cv2                      # [live] extra — imported lazily so --smoke needs neither
    import mediapipe as mp

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {camera}")
    try:
        for count in ("3", "2", "1"):
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("camera read failed")
                cv2.putText(frame, count, (60, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            4, (0, 255, 0), 8)
                cv2.imshow("signisa", frame)
                cv2.waitKey(1)

        rows, t0 = [], time.monotonic()
        with mp.solutions.holistic.Holistic() as holistic:
            while time.monotonic() - t0 < seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rows.append(holistic_row(results))
                cv2.putText(frame, "REC", (60, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 0, 255), 4)
                cv2.imshow("signisa", frame)
                cv2.waitKey(1)
        elapsed = time.monotonic() - t0
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if len(rows) < 2:
        raise SystemExit("captured fewer than 2 frames")
    return np.stack(rows), len(rows) / elapsed  # real webcams don't run at a nominal 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--db", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--untrained", action="store_true",
                        help="random-init model (smoke testing)")
    ap.add_argument("--save-dir", type=Path)
    ap.add_argument("--user-level", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--smoke", type=Path, metavar="PARQUET",
                    help="headless: run the post-capture path on a stored sample")
    args = ap.parse_args()

    model = SignModel(Config()) if args.untrained else load_checkpoint(args.checkpoint)
    model.eval()

    if args.smoke:
        holistic, fps = load_holistic(args.smoke), 30.0
    else:
        holistic, fps = capture_attempt(args.camera, args.seconds)

    result = preprocess(holistic, fps=fps,
                        left_dominant=hand_dominance(holistic, fps) == "left",
                        version=model.cfg.landmark_version)
    with torch.no_grad():
        embedding = model.embedder(torch.from_numpy(result.tensor)[None])[0].numpy()

    db = json.load(args.db.open())
    db_version = db.get("landmark_version")
    assert db_version in (None, model.cfg.landmark_version), (
        f"db centroids are {db_version} but the model is {model.cfg.landmark_version}")
    verdict = verify_attempt(embedding, args.target, db,
                             DecisionConfig(user_level=args.user_level))
    print(json.dumps(verdict.to_dict(), indent=1))

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        np.save(args.save_dir / f"{stamp}_{args.target}.npy", holistic)
        (args.save_dir / f"{stamp}_{args.target}.json").write_text(json.dumps({
            "target": args.target, "fps": round(fps, 2),
            "n_frames": int(holistic.shape[0]),
            "landmark_version": model.cfg.landmark_version,
            "verdict": verdict.to_dict(),
        }, indent=1) + "\n")
        print(f"saved attempt to {args.save_dir}/{stamp}_{args.target}.*")


if __name__ == "__main__":
    main()
