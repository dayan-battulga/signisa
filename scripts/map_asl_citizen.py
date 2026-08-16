"""Map ASL Citizen clips onto our 246 canonical training classes.

Reads the ASL Citizen split CSVs (train/val/test; columns Participant ID,
Video file, Gloss, ASL-LEX Code), matches each row to a canonical class, and
writes a mapping CSV plus a coverage report. Signer ids are namespaced
"ac_<id>" so they can never collide with PopSign participant ids.

Matching, most authoritative first:
1. ASL-LEX Code == the class's asllex_entry (exact normalized match only —
   variant suffixes like _2 are DIFFERENT signs and must not base-match).
2. Normalized gloss against every Kaggle member gloss and its ASL-LEX alias
   (the Phase 0 ALIAS table, both directions: "wake" and "mother" both land) —
   but ONLY if the row's code doesn't name a different variant of our class
   (code dog_2 vs our dog_1 = a visually different sign; excluded and reported
   as a variant mismatch, never silently folded in).
Unmatched glosses go to the report; extend CITIZEN_ALIAS from that list after
the download.

Usage:
    python scripts/map_asl_citizen.py --csv-dir <asl-citizen>/splits \
        [--labels data/meta/training_labels.json] [--out-dir data/meta]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from coverage_analysis import ALIAS, normalize  # sibling script, same sys.path

# Citizen gloss -> our canonical gloss, for names the ALIAS table doesn't already
# bridge. Fill from the unmatched list in asl_citizen_coverage.md after download.
CITIZEN_ALIAS: dict[str, str] = {}

REQUIRED = {"participantid", "videofile", "gloss", "asllexcode"}


def load_split_csvs(csv_dir: Path) -> pd.DataFrame:
    """All split CSVs concatenated, columns normalized, split name attached."""
    frames = []
    for path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df.columns = [normalize(c) for c in df.columns]
        missing = REQUIRED - set(df.columns)
        assert not missing, (f"{path.name} lacks {sorted(missing)} "
                             f"(has {sorted(df.columns)})")
        df["split"] = path.stem
        frames.append(df)
    assert frames, f"no CSVs in {csv_dir}"
    return pd.concat(frames, ignore_index=True)


def class_lookups(labels: dict) -> tuple[dict[str, int], dict[str, int], dict[int, str]]:
    """(normalized asllex_entry -> id, normalized gloss/alias -> id, id -> entry)."""
    by_entry = {normalize(c["asllex_entry"]): c["id"]
                for c in labels["classes"] if c["asllex_entry"]}
    entry_of = {c["id"]: normalize(c["asllex_entry"])
                for c in labels["classes"] if c["asllex_entry"]}
    by_gloss = {}
    for member, class_id in labels["sign_to_class"].items():
        for name in (member, ALIAS.get(member)):
            if name is None:
                continue
            key = normalize(name)
            assert by_gloss.get(key, class_id) == class_id, f"gloss collision on {key}"
            by_gloss[key] = class_id
    return by_entry, by_gloss, entry_of


def match_row(row, by_entry: dict[str, int], by_gloss: dict[str, int],
              entry_of: dict[int, str]) -> tuple[int | None, str]:
    """-> (class id or None, how): 'entry' | 'gloss' | 'variant_mismatch' | 'unmatched'."""
    code = "" if pd.isna(row.asllexcode) else normalize(row.asllexcode)
    if code in by_entry:
        return by_entry[code], "entry"
    gloss = normalize(row.gloss)
    class_id = by_gloss.get(normalize(CITIZEN_ALIAS.get(gloss, gloss)))
    if class_id is None:
        return None, "unmatched"
    # A code that isn't our class's entry is a different sign. That includes classes
    # with NO entry (entry_of miss): store is deliberately unmapped because ASL-LEX
    # 'shop_1' is the SHOPPING sign — a Citizen STORE clip carrying shop_1 IS that
    # other sign, and gloss-matching it in would pollute the class.
    if code and code != entry_of.get(class_id):
        return None, "variant_mismatch"
    return class_id, "gloss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("data/meta/training_labels.json"))
    ap.add_argument("--curriculum", type=Path, default=Path("data/meta/curriculum_db.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/meta"))
    args = ap.parse_args()

    labels = json.load(args.labels.open())
    label_of = {c["id"]: c["label"] for c in labels["classes"]}
    by_entry, by_gloss, entry_of = class_lookups(labels)
    rows = load_split_csvs(args.csv_dir)
    results = [match_row(r, by_entry, by_gloss, entry_of) for r in rows.itertuples()]
    rows["canonical_label_id"] = [r[0] for r in results]
    rows["how"] = [r[1] for r in results]

    matched = rows[rows.canonical_label_id.notna()].copy()
    matched["canonical_label_id"] = matched.canonical_label_id.astype(int)
    matched["canonical_label"] = matched.canonical_label_id.map(label_of)
    matched["participant_id"] = "ac_" + matched.participantid.astype(str)
    mapping = matched[["videofile", "gloss", "asllexcode", "participant_id",
                       "split", "canonical_label_id", "canonical_label"]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.out_dir / "asl_citizen_mapping.csv", index=False)

    curriculum = set(json.load(args.curriculum.open())["signs"])
    write_report(args.out_dir / "asl_citizen_coverage.md", rows, matched, label_of,
                 curriculum)
    covered = set(matched.canonical_label)
    print(f"{len(matched)}/{len(rows)} clips matched; classes covered "
          f"{len(covered)}/{len(label_of)}, curriculum {len(covered & curriculum)}/"
          f"{len(curriculum)} -> {args.out_dir}/asl_citizen_mapping.csv")


def write_report(path: Path, rows: pd.DataFrame, matched: pd.DataFrame,
                 label_of: dict, curriculum: set[str]) -> None:
    covered = set(matched.canonical_label)
    all_labels = set(label_of.values())
    variant = (rows[rows.how == "variant_mismatch"]
               .groupby(["gloss", "asllexcode"]).size().sort_values(ascending=False))
    unmatched = (rows[rows.how == "unmatched"]
                 .groupby("gloss").size().sort_values(ascending=False))
    per_sign = matched.groupby("canonical_label").agg(
        n_clips=("videofile", "count"), n_signers=("participant_id", "nunique"),
        splits=("split", lambda s: " ".join(f"{k}:{v}"
                for k, v in s.value_counts().sort_index().items())))
    lines = [
        "# ASL Citizen coverage vs the 246 canonical classes\n",
        "Generated by `scripts/map_asl_citizen.py`.\n",
        f"- {len(matched)}/{len(rows)} clips matched "
        f"({len(matched) / len(rows):.1%}); {matched.participant_id.nunique()} signers.",
        f"- Classes covered: {len(covered)}/{len(all_labels)}; "
        f"curriculum signs covered: {len(covered & curriculum)}/{len(curriculum)}.",
        f"- Curriculum signs WITHOUT Citizen clips: "
        f"{', '.join(sorted(curriculum - covered)) or 'none'}.",
        f"- Classes without Citizen clips: "
        f"{', '.join(sorted(all_labels - covered)) or 'none'}.\n",
        "## Per-sign counts\n",
        "| sign | clips | signers | splits |", "|---|---|---|---|",
        *[f"| {sign} | {r.n_clips} | {r.n_signers} | {r.splits} |"
          for sign, r in per_sign.iterrows()],
        f"\n## Variant mismatches ({variant.sum()} clips excluded)\n",
        "Same English gloss, different ASL-LEX variant — visually different signs,",
        "kept OUT of training on purpose.\n",
        *[f"- {g} ({code}): {n}" for (g, code), n in variant.items()],
        f"\n## Unmatched Citizen glosses ({len(unmatched)} glosses, "
        f"{unmatched.sum()} clips)\n",
        "Add real matches to CITIZEN_ALIAS in map_asl_citizen.py; most of these",
        "are simply outside our 250-sign vocabulary.\n",
        *[f"- {g}: {n}" for g, n in unmatched.items()],
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
