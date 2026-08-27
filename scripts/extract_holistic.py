"""Extract MediaPipe Holistic landmarks from a directory of videos (ASL Citizen).

One float16 npz per clip: the FULL (T, 543, 3) Kaggle-layout array (so future
landmark sets derive without re-extraction), fps, n_frames, n_detected_frames,
and the mediapipe version (extractor provenance seam — see preprocess/holistic.py).

Resumable by construction: outputs are written to a .part file and atomically
renamed, so an existing <stem>.npz IS the completed-clip manifest — reruns skip
it. Clips with zero detected frames (or unreadable files) are appended to
<out-dir>/failures.csv and also skipped on restart; delete that file to retry
them. Chaining: --prior-inputs (auto-defaults to /kaggle/input when it exists)
is scanned RECURSIVELY for prior versions' shard manifests at any mount nesting;
their npz + failures are folded into out-dir so the newest version carries the
complete corpus, and a loud launch gate prints what was found (or a NO PRIOR
MANIFESTS warning) before extraction. --done-dir (repeatable) is the local
no-copy alternative. --time-budget-h stops cleanly (exit 0) once the budget
elapses, so a session-capped platform still SAVES everything finished.
Needs the [live] extra and the model bundle (create_landmarker prints the curl
command if it's missing).

Usage:
    python scripts/extract_holistic.py --videos-dir <asl-citizen>/videos \
        --out-dir data/extracted/asl_citizen [--workers 4] [--limit 3]
"""

import argparse
import csv
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from signisa.preprocess.holistic import (
    DEFAULT_MODEL,
    MAX_SIDE,
    create_landmarker,
    extract_video,
    first_frame_dims,
)

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
RECREATE_EVERY = 500  # clips per landmarker instance: reuse for speed, recreate as a leak guard
MANIFEST_NAME = "shard_manifest.json"

_LANDMARKER, _TS_OFFSET, _CLIPS_DONE, _DIMS = None, 0, 0, None


def _drop_landmarker() -> None:
    global _LANDMARKER
    if _LANDMARKER is not None:
        try:
            _LANDMARKER.close()
        except Exception:
            pass  # a failed graph often cannot close; the fresh instance is the fix
        _LANDMARKER = None


def _init_worker(model_path: Path, out_dir: Path, max_side: int) -> None:
    global _MODEL_PATH, _OUT_DIR, _MAX_SIDE
    _MODEL_PATH, _OUT_DIR, _MAX_SIDE = model_path, out_dir, max_side
    # mediapipe/TFLite native code sprays INFO/WARNING lines on fd 2 at every
    # landmarker creation; send each worker's stderr to its own file so the parent's
    # progress lines stay readable. Python exceptions still reach the parent via the
    # future, so nothing actionable is hidden.
    logs = out_dir / "worker_logs"
    logs.mkdir(exist_ok=True)
    fd = os.open(logs / f"worker-{os.getpid()}.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 2)
    os.close(fd)


def _extract_job(video: Path) -> tuple[str, str, str]:
    """Runs in a worker: one landmarker per process, reused across clips.

    Reuse (verified bit-identical to fresh instances on same-size clips) needs three
    disciplines. VIDEO-mode timestamps must strictly increase per instance, so each
    clip starts past the previous clip's last timestamp. The graph RET_CHECK-crashes
    when frame dimensions change between calls, so a clip with different downscaled
    dims gets a fresh instance. And a mid-clip exception leaves the frames-fed count
    unknown — a desynced offset would spuriously fail EVERY later clip in this
    worker — so any exception discards the instance too.
    """
    global _LANDMARKER, _TS_OFFSET, _CLIPS_DONE, _DIMS
    try:
        dims = first_frame_dims(video, _MAX_SIDE)
    except Exception as e:
        return "fail", video.stem, str(e)
    if _LANDMARKER is not None and (dims != _DIMS or _CLIPS_DONE >= RECREATE_EVERY):
        _drop_landmarker()
    if _LANDMARKER is None:
        _LANDMARKER = create_landmarker(_MODEL_PATH)
        _TS_OFFSET, _CLIPS_DONE, _DIMS = 0, 0, dims
    status, stem, detail, next_ts = extract_one(
        video, _OUT_DIR, _LANDMARKER, max_side=_MAX_SIDE, ts_offset_ms=_TS_OFFSET)
    if next_ts is None:  # exception mid-clip: offset unknown, graph state suspect
        _drop_landmarker()
    else:
        _TS_OFFSET, _CLIPS_DONE = next_ts, _CLIPS_DONE + 1
    return status, stem, detail


def extract_one(video: Path, out_dir: Path, landmarker, max_side: int = MAX_SIDE,
                ts_offset_ms: int = 0) -> tuple[str, str, str, int | None]:
    """-> (status, stem, detail, next ts offset — None when an exception makes the
    landmarker's fed-frame count unknown). Atomic write, never partial."""
    try:
        holistic, fps = extract_video(video, landmarker, max_side, ts_offset_ms)
    except Exception as e:  # unreadable/corrupt clip must not kill the whole run
        return "fail", video.stem, str(e), None
    next_ts = ts_offset_ms + int(round(len(holistic) * 1000.0 / fps)) + 1000
    detected = int(np.isfinite(holistic[:, :, 0]).any(axis=1).sum())
    if detected == 0:
        return "fail", video.stem, f"zero detected frames of {len(holistic)}", next_ts

    final = out_dir / f"{video.stem}.npz"
    part = final.with_suffix(".npz.part")
    with part.open("wb") as f:
        np.savez_compressed(f, holistic=holistic.astype(np.float16), fps=fps,
                            n_frames=len(holistic), n_detected_frames=detected,
                            mediapipe_version=_mediapipe_version())
    os.replace(part, final)
    return ("ok", video.stem,
            f"{len(holistic)} frames, {detected} detected, {fps:.1f} fps", next_ts)


def _mediapipe_version() -> str:
    try:
        import mediapipe
        return mediapipe.__version__
    except ImportError:  # stub-landmarker tests run without the [live] extra
        return "stub"


def find_prior_manifests(root: Path, max_depth: int = 6) -> list[Path]:
    """Every shard_manifest.json under root, at ANY nesting.

    Kaggle mount layouts move around — flat /kaggle/input/<slug>/ one day, nested
    /kaggle/input/datasets/<user>/<slug>/ the next (a shallow */ glob silently found
    nothing and a chained canary re-extracted from scratch). Bounded recursive walk:
    depth-capped, and descent stops at any directory holding video files — that's
    the raw dataset tree, never an extraction output.
    """
    found: list[Path] = []

    def walk(d: Path, depth: int) -> None:
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        if any(e.name == MANIFEST_NAME for e in entries):
            found.append(d / MANIFEST_NAME)
        if depth >= max_depth or any(e.suffix.lower() in VIDEO_SUFFIXES for e in entries):
            return
        for e in sorted(entries):
            if e.is_dir():
                walk(e, depth + 1)

    walk(root, 0)
    return sorted(found)


def manifest_clips_dir(manifest: Path) -> Path:
    """Where a manifest's npz files live: an extracted/ sibling (notebook-written
    manifests) or the manifest's own directory (script-written ones)."""
    sibling = manifest.parent / "extracted"
    return sibling if sibling.is_dir() else manifest.parent


def manifest_slug(manifest: Path) -> str:
    """The prior OUTPUT's name, not the npz dir's: an in-dir manifest sits inside
    extracted/, whose parent is the published version."""
    clips = manifest_clips_dir(manifest)
    return clips.parent.name if clips.name == "extracted" else clips.name


def fold_prior_outputs(manifests: list[Path], out_dir: Path) -> int:
    """Copy prior versions' npz into out_dir and merge every failures.csv into
    out_dir's own, so the newest version carries the complete corpus (c4ad52d).
    Idempotent: same-stem files are skipped, failure rows dedup by stem."""
    copied = 0
    failures: dict[str, str] = {}
    own = out_dir / "failures.csv"
    if own.exists():
        with own.open() as f:
            failures = {r["file"]: r["reason"] for r in csv.DictReader(f)}
    for m in manifests:
        clips = manifest_clips_dir(m)
        for p in clips.glob("*.npz"):
            if not (out_dir / p.name).exists():
                shutil.copy(p, out_dir / p.name)
                copied += 1
        prior = clips / "failures.csv"
        if prior.exists():
            with prior.open() as f:
                for r in csv.DictReader(f):
                    failures.setdefault(r["file"], r["reason"])
    if failures:
        with own.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "reason"])
            w.writerows(sorted(failures.items()))
    return copied


def _print_input_tree(root: Path, max_entries: int = 15) -> None:
    print(f"=== attached inputs under {root} (depth 2) ===")
    for top in sorted(p for p in root.iterdir() if p.is_dir()):
        print(f"  {top.name}/")
        children = sorted(top.iterdir())
        for child in children[:max_entries]:
            print(f"    {child.name}{'/' if child.is_dir() else ''}")
        if len(children) > max_entries:
            print(f"    ... ({len(children) - max_entries} more)")


def pending_videos(videos_dir: Path, out_dir: Path, shard_index: int = 0,
                   num_shards: int = 1, done_dirs: list[Path] = ()) -> list[Path]:
    """This shard's videos with neither a completed npz nor a logged failure.

    Shards partition the FULL sorted list (interleaved i::n), before the pending
    filter — so the partition is deterministic and identical across sessions no
    matter how much each shard has already completed. done_dirs are prior runs'
    read-only outputs (chained Kaggle versions): their npz and failures count as
    completed exactly like out_dir's own.
    """
    assert 0 <= shard_index < num_shards, f"shard {shard_index} of {num_shards}"
    videos = sorted(p for p in videos_dir.rglob("*")
                    if p.suffix.lower() in VIDEO_SUFFIXES)
    # a wrong/unmounted dir must fail loudly, not report "0 remaining" and exit 0
    assert videos, f"no video files under {videos_dir}"
    stems = [v.stem for v in videos]
    assert len(set(stems)) == len(stems), "duplicate video stems — outputs would collide"
    videos = videos[shard_index::num_shards]
    done, failed = set(), set()
    for d in [out_dir, *done_dirs]:
        done |= {p.stem for p in d.glob("*.npz")}
        failures = d / "failures.csv"
        if failures.exists():
            with failures.open() as f:
                failed |= {row["file"] for row in csv.DictReader(f)}
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
    ap.add_argument("--done-dir", type=Path, action="append", default=[],
                    help="repeatable: prior runs' outputs whose clips count as completed")
    ap.add_argument("--time-budget-h", type=float, default=None,
                    help="stop cleanly (exit 0) after this many hours; unset = run to completion")
    ap.add_argument("--max-side", type=int, default=MAX_SIDE,
                    help="downscale frames to this long side before MediaPipe; 0 = full res")
    # auto-default so already-imported Kaggle notebook copies pick the chaining fix up
    # from the repo clone alone, with no notebook re-import
    ap.add_argument("--prior-inputs", type=Path,
                    default=Path("/kaggle/input") if Path("/kaggle/input").is_dir() else None,
                    help="root scanned recursively for prior extraction manifests to fold in")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    if args.prior_inputs and args.prior_inputs.is_dir():
        # launch gate: these lines are how a chained run is judged safe to leave running
        _print_input_tree(args.prior_inputs)
        # one manifest per distinct clips dir: outputs can carry both a root-level and
        # an in-dir manifest for the same files, and our own out_dir is never a prior
        seen = {args.out_dir.resolve()}
        for m in find_prior_manifests(args.prior_inputs):
            clips = manifest_clips_dir(m).resolve()
            if clips not in seen:
                seen.add(clips)
                manifests.append(m)
        for m in manifests:
            print(f"prior manifest: {m} ({len(list(manifest_clips_dir(m).glob('*.npz')))} clips)")
        if manifests:
            copied = fold_prior_outputs(manifests, args.out_dir)
            print(f"chained_on: {[manifest_slug(m) for m in manifests]} — folded {copied} prior npz")
        else:
            print("!!! NO PRIOR MANIFESTS FOUND under "
                  f"{args.prior_inputs} — fresh run, nothing to resume !!!")
    videos = pending_videos(args.videos_dir, args.out_dir, args.shard_index,
                            args.num_shards, args.done_dir)
    n_done = len(list(args.out_dir.glob("*.npz")))
    print(f"shard {args.shard_index}/{args.num_shards}: {n_done} done, "
          f"{len(videos)} remaining")
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        _write_manifest(args, manifests, ok=0, failed=0, remaining=0)
        print("nothing pending — all clips extracted or logged as failures")
        return
    if not args.model.exists():
        create_landmarker(args.model)  # fail fast with the download command, before the pool

    print(f"worker stderr (mediapipe/TFLite log spam) -> {args.out_dir / 'worker_logs'}")
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
                                 initargs=(args.model, args.out_dir, args.max_side)) as pool:
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
                if done % 100 == 0 or done == len(videos):
                    rate = done / (time.perf_counter() - t0)
                    eta_min = (len(videos) - done) / rate / 60 if rate else 0
                    print(f"{done}/{len(videos)} ({rate:.2f} clips/s, ~{eta_min:.0f} min left)")
                if done == min(200, len(videos)):  # informational: the budget hard-stops anyway
                    rate = done / (time.perf_counter() - t0)
                    hours = len(videos) / rate / 3600
                    print(f"measured {rate:.2f} clips/s -> projected {hours:.1f} h "
                          f"for this shard's {len(videos)} clips"
                          + (f" (budget {args.time_budget_h:g} h: needs "
                             f"~{-(-hours // args.time_budget_h):.0f} chained run(s))"
                             if args.time_budget_h else ""))
                if (args.time_budget_h
                        and time.perf_counter() - t0 > args.time_budget_h * 3600):
                    print(f"time budget {args.time_budget_h:g} h reached — stopping cleanly")
                    # cancel every queued clip; wait only for the <= workers in flight
                    # (their atomic writes still land and the next run skips them)
                    pool.shutdown(wait=True, cancel_futures=True)
                    break
    remaining = len(videos) - ok - failed
    _write_manifest(args, manifests, ok, failed, remaining)
    print(f"done: {ok} extracted, {failed} failed, {remaining} remaining "
          f"-> {args.out_dir} (failures in {failures_path.name})")


def _write_manifest(args, manifests: list[Path], ok: int, failed: int,
                    remaining: int) -> None:
    """The merged manifest lives IN the output dir, so recursive discovery finds a
    chained version no matter how Kaggle nests its mount."""
    npz = list(args.out_dir.glob("*.npz"))
    failures = args.out_dir / "failures.csv"
    n_failed = sum(1 for _ in csv.DictReader(failures.open())) if failures.exists() else 0
    (args.out_dir / MANIFEST_NAME).write_text(json.dumps({
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "n_extracted": len(npz), "n_new_this_run": ok, "n_failed": n_failed,
        "n_remaining": remaining, "chained_on": [manifest_slug(m) for m in manifests],
        "size_gb": round(sum(p.stat().st_size for p in npz) / 1e9, 2)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
