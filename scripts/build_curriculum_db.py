"""Build data/meta/curriculum_db.json (backlog 0.4): per-sign phonology, confusables,
and null slots for Phase 1 training outputs (centroid, thresholds, Weibull params).

Usage: python scripts/build_curriculum_db.py [--out data/meta/curriculum_db.json]
Reads the same metadata files as coverage_analysis.py (paths via its defaults).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import coverage_analysis as ca

META = Path("data/meta")


def build_curriculum_db() -> dict:
    signs = sorted(json.load((META / "sign_to_prediction_index_map.json").open()))
    matched = ca.match_signs(signs, ca.load_asllex(META / "asllex_signdata.csv"))
    labels = json.load((META / "training_labels.json").open())
    label_of = {c["id"]: c["label"] for c in labels["classes"]}
    canonical = {s: label_of[labels["sign_to_class"][s]] for s in signs}

    feats = {s: {f: row[col] for f, col in ca.FEATURES.items()} for s, row in matched.items()}
    strong = ca.strong_graph(ca.confusable_pairs(feats))
    clusters = ca.clusters_within(strong, ca.CURRICULUM_V1)
    assert len(clusters) >= 8, f"expected the 8 proposal clusters, got {len(clusters)}"
    cluster_of = {s: comp for comp in clusters for s in comp}

    entries = {}
    for s in ca.CURRICULUM_V1:
        row = matched[s]
        # cluster members UNION strong ASL-LEX neighbors (clusters are connected
        # components, so a member need not be a direct strong neighbor); self drops out
        confusables = sorted(
            ({canonical[n] for n in strong.get(s, set())} | set(cluster_of.get(s, [])))
            - {canonical[s]})
        assert canonical[s] == s, f"curriculum sign {s} is not canonical"
        entries[s] = {
            "label": s,
            "label_id": labels["sign_to_class"][s],
            "asllex_entry": str(row["EntryID"]).strip(),
            "phonology": {
                "handshape": row["Handshape.2.0"],
                "major_location": row["MajorLocation.2.0"],
                "minor_location": _clean(row["MinorLocation.2.0"]),
                "movement": _clean(row["Movement.2.0"]),
                "sign_type": row["SignType.2.0"],
                "one_handed": row["SignType.2.0"] == "OneHanded",
            },
            "cluster": [m for m in cluster_of.get(s, []) if m != s],
            "confusables": confusables,
            "centroid": None,
            "eer_threshold": None,
            "low_far_threshold": None,
            "weibull_params": None,
        }

    canonical_labels = set(label_of.values())
    for s, e in entries.items():
        bad = [c for c in e["confusables"] if c not in canonical_labels]
        assert not bad, f"{s} has confusables that are not canonical labels: {bad}"

    return {"version": 1, "n_signs": len(entries),
            "clusters": clusters, "signs": entries}


def _clean(value):
    return None if pd.isna(value) else value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=META / "curriculum_db.json")
    args = ap.parse_args()
    db = build_curriculum_db()
    args.out.write_text(json.dumps(db, indent=1) + "\n")
    print(f"{db['n_signs']} signs, {len(db['clusters'])} clusters -> {args.out}")


if __name__ == "__main__":
    main()
