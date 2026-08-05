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
    assert len(matched) == 230  # 20 documented unmatched signs

    assert len(ca.CURRICULUM_V1) == len(set(ca.CURRICULUM_V1)) == 50
    assert set(ca.CURRICULUM_V1) <= set(matched)

    feats = {s: {f: row[col] for f, col in ca.FEATURES.items()} for s, row in matched.items()}
    strong = ca.strong_graph(ca.confusable_pairs(feats))
    clusters = ca.clusters_within(strong, ca.CURRICULUM_V1)
    assert len(clusters) >= 5
