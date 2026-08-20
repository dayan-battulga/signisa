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


def test_pending_videos_counts_prior_run_outputs_as_done(tmp_path):
    """Chained Kaggle sessions attach prior versions read-only via done_dirs."""
    videos = tmp_path / "videos"
    out = tmp_path / "out"
    prior1, prior2 = tmp_path / "v1" / "extracted", tmp_path / "v2" / "extracted"
    for d in (videos, out, prior1, prior2):
        d.mkdir(parents=True)
    for name in ("a.mp4", "b.mp4", "c.mp4", "d.mp4", "e.mp4"):
        (videos / name).touch()
    (prior1 / "a.npz").touch()
    with (prior1 / "failures.csv").open("w", newline="") as f:
        import csv as _csv
        _csv.writer(f).writerows([["file", "reason"], ["b", "zero detected frames of 3"]])
    (prior2 / "c.npz").touch()
    (out / "d.npz").touch()  # this run's own progress still counts too

    pending = pending_videos(videos, out, done_dirs=[prior1, prior2])
    assert [v.stem for v in pending] == ["e"]
    # without the priors, only this run's output is known
    assert [v.stem for v in pending_videos(videos, out)] == ["a", "b", "c", "e"]


def test_downscaled_caps_long_side():
    from signisa.preprocess.holistic import downscaled

    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = downscaled(big)
    assert small.shape == (360, 640, 3)                      # aspect kept, long side 640
    tall = downscaled(np.zeros((1280, 720, 3), dtype=np.uint8))
    assert tall.shape == (640, 360, 3)
    already = np.zeros((480, 640, 3), dtype=np.uint8)
    assert downscaled(already) is already                    # no-op, no copy
    assert downscaled(big, 0) is big                         # 0 = full resolution


def test_time_budget_stops_cleanly_with_exit_zero(tmp_path):
    pytest.importorskip("mediapipe")
    model = ROOT / "data/models/holistic_landmarker.task"
    if not model.exists():
        pytest.skip("no local model bundle")
    videos = tmp_path / "videos"
    videos.mkdir()
    for i in range(4):
        write_video(videos / f"clip{i}.mp4", n_frames=3)

    import subprocess
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/extract_holistic.py"),
         "--videos-dir", str(videos), "--out-dir", str(tmp_path / "out"),
         "--model", str(model), "--workers", "1", "--time-budget-h", "1e-7"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr                 # a saved version, not a SIGKILL
    assert "time budget" in proc.stdout and "remaining" in proc.stdout
    # the budget tripped after the first completed clip; queued ones were cancelled
    done = len(list((tmp_path / "out").glob("*.npz")))
    failures = tmp_path / "out" / "failures.csv"
    import csv as _csv
    failed = sum(1 for _ in _csv.DictReader(failures.open())) if failures.exists() else 0
    assert 1 <= done + failed < 4


def test_shard_partition_is_deterministic_and_complete(tmp_path):
    videos = tmp_path / "videos"
    out = tmp_path / "out"
    videos.mkdir(), out.mkdir()
    names = [f"clip{i:03d}.mp4" for i in range(11)]
    for name in names:
        (videos / name).touch()

    shards = [pending_videos(videos, out, i, 4) for i in range(4)]
    stems = [v.stem for s in shards for v in s]
    assert sorted(stems) == sorted(Path(n).stem for n in names)   # disjoint union = all
    assert len(set(stems)) == len(stems)
    assert shards[0] == pending_videos(videos, out, 0, 4)         # deterministic

    # the partition is over the FULL sorted list: another shard's completions
    # never shift this shard's membership
    (out / "clip000.npz").touch()                                 # shard 0's first clip
    assert pending_videos(videos, out, 1, 4) == shards[1]
    assert [v.stem for v in pending_videos(videos, out, 0, 4)] == \
        [v.stem for v in shards[0] if v.stem != "clip000"]

    with pytest.raises(AssertionError, match="shard"):
        pending_videos(videos, out, 4, 4)
