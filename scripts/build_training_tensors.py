"""Precompute training tensors: landmarks -> per-participant dominance mirror ->
preprocess -> native-length (T, N, 4) float16 (xyz + confidence), sharded .npz + index.csv.

Two input sources, merged into one shard set with a `domain` column in index.csv:
- Kaggle asl-signs parquets (domain=popsign), as always.
- ASL Citizen npz clips from extract_holistic.py + the mapping CSV from
  map_asl_citizen.py (domain=asl_citizen): --citizen-npz-dir + --citizen-mapping.
Citizen signer ids are already "ac_"-namespaced; participant_id is written as a
string column. Mapping rows whose npz is missing (extraction failures) are
skipped with a count.

Sequences keep their own length (capped at pipeline.MAX_FRAMES) — see the pipeline
docstring for why. Each shard stores its sequences' frames concatenated plus a
lengths array; signisa.data slices them back apart. Typical clips run well under the
old fixed 160 frames, so the ragged total is smaller than the fixed-length one.

Velocity and bone channels are deterministic derivations; signisa.data re-derives
them at load time. Storing them would blow Kaggle's notebook-output cap.

Local smoke test: python scripts/build_training_tensors.py \
    --landmarks-dir data/samples --out-dir data/tensors --limit 30
Full run happens on Kaggle with --landmarks-dir <asl-signs root>.

index.csv row i lives in shard i // SHARD_SIZE at slot i % SHARD_SIZE.
Note: under --limit the dominance vote is scoped to the limited subset, so
mirror flags can differ from the full run — fine for smoke tests only.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from signisa import SHARD_SCHEMA
from signisa.preprocess.dominance import (
    dominance_from_wrists,
    hand_dominance,
    majority_dominance,
)
from signisa.preprocess.kaggle import load_holistic, load_wrists
from signisa.preprocess.pipeline import preprocess

SHARD_SIZE = 500


def resolve_paths(train: pd.DataFrame, landmarks_dir: Path) -> pd.DataFrame:
    """Keep rows whose parquet exists under landmarks_dir (train.csv layout or flat)."""
    def find(row) -> Path | None:
        nested = landmarks_dir / row["path"]
        if nested.exists():
            return nested
        flat = landmarks_dir / f"{row['sequence_id']}.parquet"
        return flat if flat.exists() else None

    train = train.assign(parquet=train.apply(find, axis=1))
    return train[train.parquet.notna()].reset_index(drop=True)


def left_dominant_participants(train: pd.DataFrame, fps: float) -> set[int]:
    """Handedness is a person trait: majority vote over each participant's sequences."""
    votes: dict[int, list[str]] = {}
    for row in train.itertuples():
        left, right = load_wrists(row.parquet)
        votes.setdefault(row.participant_id, []).append(
            dominance_from_wrists(left, right, fps))
    return {pid for pid, vs in votes.items() if majority_dominance(vs) == "left"}


def citizen_clips(npz_dirs: list[Path], mapping_csv: Path, class_of: dict) -> pd.DataFrame:
    """Mapping rows joined to their extracted npz (one or more extraction-shard dirs);
    missing extractions dropped with a count.

    Label ids are re-derived from THIS build's labels file, not trusted from the CSV:
    ids are positional, so a mapping generated against an older training_labels.json
    would silently scramble every citizen label.
    """
    for d in npz_dirs:
        assert d.is_dir(), f"{d} is not a directory"
    npz_of: dict[str, Path] = {}
    for d in npz_dirs:
        for p in d.glob("*.npz"):
            assert p.stem not in npz_of, (
                f"{p.stem} appears in both {npz_of.get(p.stem)} and {p} — "
                "overlapping extraction shards?")
            npz_of[p.stem] = p
    mapping = pd.read_csv(mapping_csv)
    stale = mapping.canonical_label_id != mapping.canonical_label.map(class_of)
    assert not stale.any(), (
        f"{stale.sum()} mapping rows disagree with --labels (e.g. "
        f"{mapping[stale].canonical_label.iloc[0]}) — regenerate asl_citizen_mapping.csv")
    mapping["npz"] = [npz_of.get(Path(f).stem) for f in mapping.videofile]
    missing = mapping.npz.isna()
    assert not missing.all(), (
        f"none of the {len(mapping)} mapped clips have an npz under {npz_dirs}")
    if missing.any():
        print(f"skipping {missing.sum()}/{len(mapping)} mapped clips with no extracted npz "
              "(extraction failures or a shard not attached yet)")
    return mapping[~missing].reset_index(drop=True)


def left_dominant_citizen_signers(clips: pd.DataFrame) -> set[str]:
    votes: dict[str, list[str]] = {}
    for row in clips.itertuples():
        with np.load(row.npz) as z:
            votes.setdefault(row.participant_id, []).append(
                hand_dominance(z["holistic"].astype(np.float32), float(z["fps"])))
    return {pid for pid, vs in votes.items() if majority_dominance(vs) == "left"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=Path("data/meta/train.csv"))
    ap.add_argument("--landmarks-dir", type=Path, default=Path("data/samples"))
    ap.add_argument("--labels", type=Path, default=Path("data/meta/training_labels.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/tensors"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--landmark-version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--citizen-npz-dir", type=Path, action="append",
                    help="repeatable: one dir per extraction shard")
    ap.add_argument("--citizen-mapping", type=Path)
    args = ap.parse_args()
    assert (args.citizen_npz_dir is None) == (args.citizen_mapping is None), (
        "--citizen-npz-dir and --citizen-mapping go together")

    labels = json.load(args.labels.open())
    class_of = labels["sign_to_class"]
    train = resolve_paths(pd.read_csv(args.train_csv), args.landmarks_dir)
    if train.empty:
        raise SystemExit(f"no train.csv parquets found under {args.landmarks_dir}")
    if args.limit is not None:
        train = train.head(args.limit)
    citizen = (citizen_clips(args.citizen_npz_dir, args.citizen_mapping, class_of)
               if args.citizen_npz_dir else pd.DataFrame())

    t0 = time.perf_counter()
    left_pids = left_dominant_participants(train, args.fps)
    left_citizen = left_dominant_citizen_signers(citizen) if len(citizen) else set()
    t1 = time.perf_counter()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows, shard, shard_ids, n_shards = [], [], [], 0

    def flush() -> None:
        nonlocal shard, shard_ids, n_shards
        if not shard:
            return
        np.savez_compressed(args.out_dir / f"shard_{n_shards:04d}.npz",
                            frames=np.concatenate(shard),
                            lengths=np.array([len(s) for s in shard]),
                            sequence_id=np.array([str(s) for s in shard_ids]),
                            landmark_version=np.array(args.landmark_version),
                            schema_version=np.array(SHARD_SCHEMA))
        shard, shard_ids, n_shards = [], [], n_shards + 1

    def add_sequence(holistic: np.ndarray, fps: float, mirrored: bool,
                     sequence_id: str, participant_id: str, label_id: int,
                     domain: str) -> None:
        result = preprocess(holistic, fps=fps, left_dominant=mirrored,
                            version=args.landmark_version)
        tensor16 = result.tensor[..., [0, 1, 2, 9]].astype(np.float16)  # xyz + confidence
        # catches float16 overflow from a degenerate shoulder-width normalization
        assert np.isfinite(tensor16).all(), f"non-finite float16 tensor for {sequence_id}"
        shard.append(tensor16)
        shard_ids.append(sequence_id)
        index_rows.append({
            "sequence_id": sequence_id,
            "participant_id": participant_id,
            "canonical_label_id": label_id,
            "duration_s": round(result.duration_s, 4),
            "peak_speed": round(result.peak_speed, 4),
            "mirrored": mirrored,
            "landmark_version": args.landmark_version,
            "n_frames": tensor16.shape[0],
            "domain": domain,
        })
        if len(shard) == SHARD_SIZE:
            flush()

    for row in train.itertuples():
        add_sequence(load_holistic(row.parquet), args.fps,
                     row.participant_id in left_pids, str(row.sequence_id),
                     str(row.participant_id), class_of[row.sign], "popsign")
    for row in citizen.itertuples():
        with np.load(row.npz) as z:
            holistic, fps = z["holistic"].astype(np.float32), float(z["fps"])
        add_sequence(holistic, fps, row.participant_id in left_citizen,
                     Path(row.videofile).stem, row.participant_id,
                     int(row.canonical_label_id), "asl_citizen")
    flush()
    t2 = time.perf_counter()

    pd.DataFrame(index_rows).to_csv(args.out_dir / "index.csv", index=False)
    n = len(index_rows)
    frames = [r["n_frames"] for r in index_rows]
    domains = pd.Series([r["domain"] for r in index_rows]).value_counts().to_dict()
    print(f"frames per sequence: min {min(frames)} median {int(np.median(frames))} "
          f"max {max(frames)} (fixed-160 storage would be {160 * n / sum(frames):.1f}x larger)")
    print(f"{n} sequences {domains} ({args.landmark_version}) -> {n_shards} shard(s) "
          f"in {args.out_dir}; {len(left_pids) + len(left_citizen)} left-dominant signers, "
          f"{sum(r['mirrored'] for r in index_rows)} sequences mirrored")
    print(f"throughput: dominance pass {n / (t1 - t0):.1f} seq/s, "
          f"tensor pass {n / (t2 - t1):.1f} seq/s, overall {n / (t2 - t0):.1f} seq/s")


if __name__ == "__main__":
    main()
