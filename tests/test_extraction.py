"""Extraction plumbing tests with a stub landmarker: Kaggle-layout row conversion,
atomic npz writes, zero-detection rejection, and resume semantics. The real-model
path is exercised manually (extract_holistic.py on local videos), not here."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extract_holistic import extract_one, pending_videos  # noqa: E402

from signisa.preprocess.holistic import holistic_row  # noqa: E402


def fake_result(face=0, pose=0, left=0, right=0):
    """A HolisticLandmarker-shaped result with n constant landmarks per part."""
    point = SimpleNamespace(x=0.5, y=0.5, z=0.1)
    return SimpleNamespace(face_landmarks=[point] * face, pose_landmarks=[point] * pose,
                           left_hand_landmarks=[point] * left,
                           right_hand_landmarks=[point] * right)


class StubLandmarker:
    """Yields one canned result per frame; empty results once the list runs out."""

    def __init__(self, results):
        self.results = list(results)
        self.timestamps = []

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return self.results.pop(0) if self.results else fake_result()

    def close(self):
        pass


def write_video(path: Path, n_frames: int = 5) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.release()


def test_holistic_row_layout_and_iris_truncation():
    empty = holistic_row(fake_result())
    assert empty.shape == (543, 3) and np.isnan(empty).all()

    # the Tasks face mesh ships 478 points; the Kaggle layout keeps the first 468
    full = holistic_row(fake_result(face=478, pose=33, left=21, right=21))
    assert np.isfinite(full).all()

    partial = holistic_row(fake_result(pose=33))
    assert np.isfinite(partial[489:522]).all()        # pose block filled
    assert np.isnan(partial[:468]).all()              # face missing
    assert np.isnan(partial[468:489]).all() and np.isnan(partial[522:]).all()  # hands missing


def test_extract_one_writes_npz_and_rejects_zero_detection(tmp_path):
    video = tmp_path / "clip1.mp4"
    write_video(video, n_frames=5)

    stub = StubLandmarker([fake_result(pose=33)] * 3)  # 3 detected, 2 empty frames
    status, stem, detail = extract_one(video, tmp_path, landmarker=stub)
    assert (status, stem) == ("ok", "clip1")
    assert stub.timestamps == sorted(stub.timestamps)  # strictly increasing ints
    assert len(set(stub.timestamps)) == len(stub.timestamps)
    with np.load(tmp_path / "clip1.npz") as z:
        assert z["holistic"].shape == (5, 543, 3) and z["holistic"].dtype == np.float16
        assert int(z["n_detected_frames"]) == 3 and int(z["n_frames"]) == 5
        assert float(z["fps"]) > 0
    assert not list(tmp_path.glob("*.part"))          # atomic write left no temp file

    video2 = tmp_path / "clip2.mp4"
    write_video(video2, n_frames=4)
    status, stem, detail = extract_one(video2, tmp_path, landmarker=StubLandmarker([]))
    assert (status, stem) == ("fail", "clip2") and "zero detected" in detail
    assert not (tmp_path / "clip2.npz").exists()


def test_pending_videos_resume_semantics(tmp_path):
    videos = tmp_path / "videos"
    out = tmp_path / "out"
    videos.mkdir(), out.mkdir()
    for name in ("a.mp4", "b.MOV", "c.mp4", "d.mp4"):
        (videos / name).touch()
    (out / "a.npz").touch()                              # completed
    with (out / "failures.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["file", "reason"], ["b", "zero detected frames of 9"]])
    (out / "c.npz.part").touch()                         # crashed mid-write

    pending = pending_videos(videos, out)
    assert [v.stem for v in pending] == ["c", "d"]       # a done, b failed, c retried
    assert not (out / "c.npz.part").exists()             # stale partial cleaned up

    (videos / "sub").mkdir()
    (videos / "sub" / "a.mp4").touch()                   # duplicate stem across dirs
    with pytest.raises(AssertionError, match="duplicate video stems"):
        pending_videos(videos, out)
