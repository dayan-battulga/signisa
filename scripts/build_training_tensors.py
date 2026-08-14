"""Precompute training tensors: parquet -> per-participant dominance mirror ->
preprocess -> (160, 65, 4) float16 (xyz + confidence), sharded .npz + index.csv.

Velocity and bone channels are deterministic derivations; signisa.data re-derives
them at load time. Storing them would blow Kaggle's notebook-output cap
(10ch = ~20 GB for 94,477 sequences; 4ch = ~8 GB).

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

from signisa.preprocess.dominance import dominance_from_wrists, majority_dominance
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=Path("data/meta/train.csv"))
    ap.add_argument("--landmarks-dir", type=Path, default=Path("data/samples"))
    ap.add_argument("--labels", type=Path, default=Path("data/meta/training_labels.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/tensors"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--landmark-version", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()

    labels = json.load(args.labels.open())
    class_of = labels["sign_to_class"]
    train = resolve_paths(pd.read_csv(args.train_csv), args.landmarks_dir)
    if train.empty:
        raise SystemExit(f"no train.csv parquets found under {args.landmarks_dir}")
    if args.limit is not None:
        train = train.head(args.limit)

    t0 = time.perf_counter()
    left_pids = left_dominant_participants(train, args.fps)
    t1 = time.perf_counter()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows, shard, shard_ids, n_shards = [], [], [], 0

    def flush() -> None:
        nonlocal shard, shard_ids, n_shards
        if not shard:
            return
        np.savez_compressed(args.out_dir / f"shard_{n_shards:04d}.npz",
                            tensors=np.stack(shard), sequence_id=np.array(shard_ids),
                            landmark_version=np.array(args.landmark_version))
        shard, shard_ids, n_shards = [], [], n_shards + 1

    for row in train.itertuples():
        mirrored = row.participant_id in left_pids
        result = preprocess(load_holistic(row.parquet), fps=args.fps, left_dominant=mirrored,
                            version=args.landmark_version)
        tensor16 = result.tensor[..., [0, 1, 2, 9]].astype(np.float16)  # xyz + confidence
        # catches float16 overflow from a degenerate shoulder-width normalization
        assert np.isfinite(tensor16).all(), f"non-finite float16 tensor for {row.sequence_id}"
        shard.append(tensor16)
        shard_ids.append(row.sequence_id)
        index_rows.append({
            "sequence_id": row.sequence_id,
            "participant_id": row.participant_id,
            "canonical_label_id": class_of[row.sign],
            "duration_s": round(result.duration_s, 4),
            "peak_speed": round(result.peak_speed, 4),
            "mirrored": mirrored,
            "landmark_version": args.landmark_version,
        })
        if len(shard) == SHARD_SIZE:
            flush()
    flush()
    t2 = time.perf_counter()

    pd.DataFrame(index_rows).to_csv(args.out_dir / "index.csv", index=False)
    n = len(index_rows)
    print(f"{n} sequences ({args.landmark_version}) -> {n_shards} shard(s) in {args.out_dir}; "
          f"{len(left_pids)} left-dominant participants, "
          f"{sum(r['mirrored'] for r in index_rows)} sequences mirrored")
    print(f"throughput: dominance pass {n / (t1 - t0):.1f} seq/s, "
          f"tensor pass {n / (t2 - t1):.1f} seq/s, overall {n / (t2 - t0):.1f} seq/s")


if __name__ == "__main__":
    main()
