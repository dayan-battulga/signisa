"""Regression checks for coverage_analysis on the committed metadata files."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import coverage_analysis as ca

META = Path(__file__).parent.parent / "data" / "meta"

pytestmark = pytest.mark.skipif(
    not (META / "asllex_signdata.csv").exists(), reason="metadata not present"
)


def test_matching_and_curriculum_invariants():
    signs = sorted(json.load((META / "sign_to_prediction_index_map.json").open()))
    by_base = ca.load_asllex(META / "asllex_signdata.csv")
    matched = ca.match_signs(signs, by_base)
    assert len(matched) == 233  # 17 documented unmatched signs

    assert len(ca.CURRICULUM_V1) == len(set(ca.CURRICULUM_V1)) == 50
    assert set(ca.CURRICULUM_V1) <= set(matched)

    feats = {s: {f: row[col] for f, col in ca.FEATURES.items()} for s, row in matched.items()}
    strong = ca.strong_graph(ca.confusable_pairs(feats))
    clusters = ca.clusters_within(strong, ca.CURRICULUM_V1)
    assert len(clusters) >= 5


def test_training_label_collisions():
    import pandas as pd

    signs = sorted(json.load((META / "sign_to_prediction_index_map.json").open()))
    train = pd.read_csv(META / "train.csv")
    counts = train.groupby("sign").agg(
        n_examples=("sequence_id", "count"), n_signers=("participant_id", "nunique"))
    matched = ca.match_signs(signs, ca.load_asllex(META / "asllex_signdata.csv"))

    labels = ca.derive_training_labels(signs, matched, counts)
    collisions = {c["label"]: c["members"] for c in labels["classes"] if len(c["members"]) > 1}
    assert collisions == {
        "awake": ["awake", "wake"], "cat": ["cat", "kitty"],
        "dog": ["dog", "puppy"], "sleep": ["nap", "sleep"],
    }
    assert labels["n_classes"] == 246
    label_of = {c["id"]: c["label"] for c in labels["classes"]}
    assert all(label_of[labels["sign_to_class"][s]] == s for s in ca.CURRICULUM_V1)
