"""Extract MediaPipe Holistic landmarks from a directory of videos (ASL Citizen).

One float16 npz per clip: the FULL (T, 543, 3) Kaggle-layout array (so future
landmark sets derive without re-extraction), fps, n_frames, n_detected_frames,
and the mediapipe version (extractor provenance seam — see preprocess/holistic.py).

Resumable by construction: outputs are written to a .part file and atomically
renamed, so an existing <stem>.npz IS the completed-clip manifest — reruns skip
it. Clips with zero detected frames (or unreadable files) are appended to
<out-dir>/failures.csv and also skipped on restart; delete that file to retry
them. Needs the [live] extra and the model bundle (create_landmarker prints the
curl command if it's missing).

Usage:
    python scripts/extract_holistic.py --videos-dir <asl-citizen>/videos \
        --out-dir data/extracted/asl_citizen [--workers 4] [--limit 3]
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from signisa.preprocess.holistic import DEFAULT_MODEL, create_landmarker, extract_video

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

def _init_worker(model_path: Path, out_dir: Path) -> None:
    global _MODEL_PATH, _OUT_DIR
    _MODEL_PATH, _OUT_DIR = model_path, out_dir


def _extract_job(video: Path) -> tuple[str, str, str]:
    return extract_one(video, _OUT_DIR, model_path=_MODEL_PATH)


def extract_one(video: Path, out_dir: Path, landmarker=None,
                model_path: Path = DEFAULT_MODEL) -> tuple[str, str, str]:
    """-> ("ok", stem, stats) or ("fail", stem, reason). Atomic write, never partial.

    A fresh landmarker per clip unless one is injected (tests): VIDEO-mode tracking
    state and the monotonic-timestamp requirement must not leak across clips.
    """
    own = landmarker is None
    if own:
        landmarker = create_landmarker(model_path)
    try:
        holistic, fps = extract_video(video, landmarker)
    except Exception as e:  # unreadable/corrupt clip must not kill the whole run
        return "fail", video.stem, str(e)
    finally:
        if own:
            landmarker.close()
    detected = int(np.isfinite(holistic[:, :, 0]).any(axis=1).sum())
    if detected == 0:
        return "fail", video.stem, f"zero detected frames of {len(holistic)}"

    final = out_dir / f"{video.stem}.npz"
    part = final.with_suffix(".npz.part")
    with part.open("wb") as f:
        np.savez_compressed(f, holistic=holistic.astype(np.float16), fps=fps,
                            n_frames=len(holistic), n_detected_frames=detected,
                            mediapipe_version=_mediapipe_version())
    os.replace(part, final)
    return "ok", video.stem, f"{len(holistic)} frames, {detected} detected, {fps:.1f} fps"


def _mediapipe_version() -> str:
    try:
        import mediapipe
        return mediapipe.__version__
    except ImportError:  # stub-landmarker tests run without the [live] extra
        return "stub"


def pending_videos(videos_dir: Path, out_dir: Path, shard_index: int = 0,
                   num_shards: int = 1) -> list[Path]:
    """This shard's videos with neither a completed npz nor a logged failure.

    Shards partition the FULL sorted list (interleaved i::n), before the pending
    filter — so the partition is deterministic and identical across sessions no
    matter how much each shard has already completed.
    """
    assert 0 <= shard_index < num_shards, f"shard {shard_index} of {num_shards}"
    videos = sorted(p for p in videos_dir.rglob("*")
                    if p.suffix.lower() in VIDEO_SUFFIXES)
    stems = [v.stem for v in videos]
    assert len(set(stems)) == len(stems), "duplicate video stems — outputs would collide"
    videos = videos[shard_index::num_shards]
    done = {p.stem for p in out_dir.glob("*.npz")}
    failed = set()
    failures = out_dir / "failures.csv"
    if failures.exists():
        with failures.open() as f:
            failed = {row["file"] for row in csv.DictReader(f)}
    for stale in out_dir.glob("*.npz.part"):  # a crash mid-write leaves these
        stale.unlink()
    return [v for v in videos if v.stem not in done and v.stem not in failed]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=0,
                    help="with --num-shards: extract slice i::n of the sorted video list")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--session-budget-h", type=float, default=11.0,
                    help="warn early if the projected shard time exceeds this")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        create_landmarker(args.model)  # fail fast with the download command, before the pool
    videos = pending_videos(args.videos_dir, args.out_dir, args.shard_index, args.num_shards)
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        print("nothing pending — all clips extracted or logged as failures")
        return

    failures_path = args.out_dir / "failures.csv"
    new_file = not failures_path.exists()
    t0 = time.perf_counter()
    ok = failed = 0
    with failures_path.open("a", newline="") as flog:
        fwriter = csv.writer(flog)
        if new_file:
            fwriter.writerow(["file", "reason"])
            flog.flush()  # a hard kill must never leave a headerless CSV for the resume parser
        # ProcessPoolExecutor, not multiprocessing.Pool: a worker killed by a native
        # crash (mediapipe segfault, OOM) makes pending futures raise BrokenProcessPool,
        # where Pool.imap_unordered would hang forever waiting on the lost result
        with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                 initargs=(args.model, args.out_dir)) as pool:
            futures = {pool.submit(_extract_job, v): v for v in videos}
            for future in as_completed(futures):
                status, stem, detail = future.result()
                if status == "ok":
                    ok += 1
                else:
                    failed += 1
                    fwriter.writerow([stem, detail])
                    flog.flush()
                    print(f"FAIL {stem}: {detail}")
                done = ok + failed
                if done % 25 == 0 or done == len(videos):
                    rate = done / (time.perf_counter() - t0)
                    eta_min = (len(videos) - done) / rate / 60 if rate else 0
                    print(f"{done}/{len(videos)} ({rate:.2f} clips/s, ~{eta_min:.0f} min left)")
                if done == min(200, len(videos)):  # early enough to abort and re-shard
                    rate = done / (time.perf_counter() - t0)
                    hours = len(videos) / rate / 3600
                    print(f"projected shard time: {hours:.1f} h for {len(videos)} clips")
                    if hours > args.session_budget_h:
                        print(f"WARNING: over the {args.session_budget_h:g} h session "
                              "budget — raise --num-shards")
    print(f"done: {ok} extracted, {failed} failed -> {args.out_dir} "
          f"(failures in {failures_path.name})")


if __name__ == "__main__":
    main()
