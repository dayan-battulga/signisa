"""End-to-end verdict for one attempt: parquet -> dominance -> preprocess ->
embedding -> decision policy -> Verdict JSON on stdout.

Usage:
    python scripts/verify_attempt.py --parquet data/samples/<seq>.parquet \
        --target mom --db curriculum_db_trained.json --checkpoint model.pt
    --untrained swaps the checkpoint for a random-init model (smoke testing).

Dominance here is the per-sequence fallback (unseen participant at inference);
training data uses the per-participant majority vote instead.
"""

import argparse
import json
from pathlib import Path

import torch

from signisa.config import Config
from signisa.decision import DecisionConfig, verify_attempt
from signisa.models import SignModel
from signisa.preprocess.dominance import hand_dominance
from signisa.preprocess.kaggle import load_holistic
from signisa.preprocess.pipeline import preprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--untrained", action="store_true",
                    help="random-init model instead of a checkpoint (smoke test)")
    ap.add_argument("--user-level", type=float, default=0.0)
    args = ap.parse_args()
    if not args.untrained and args.checkpoint is None:
        ap.error("--checkpoint required unless --untrained")

    if args.untrained:
        model = SignModel(Config())
    else:
        state = torch.load(args.checkpoint, map_location="cpu")
        model = SignModel(Config(loss="ce" if "head.bias" in state else "arcface"))
        model.load_state_dict(state)
    model.eval()

    holistic = load_holistic(args.parquet)
    result = preprocess(holistic, left_dominant=hand_dominance(holistic) == "left")
    with torch.no_grad():
        embedding = model.embedder(torch.from_numpy(result.tensor)[None])[0].numpy()

    db = json.load(args.db.open())
    verdict = verify_attempt(embedding, args.target, db,
                             DecisionConfig(user_level=args.user_level))
    print(json.dumps(verdict.to_dict(), indent=1))


if __name__ == "__main__":
    main()
