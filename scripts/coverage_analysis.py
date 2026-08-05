"""Coverage analysis for the Kaggle asl-signs 250 vocabulary (backlog Phase 0.2).

Joins train.csv counts with ASL-LEX 2.0 phonology, marks WLASL-2000 overlap,
and emits the v1 50-sign curriculum with deliberate minimal-pair clusters.

Usage:
    python scripts/coverage_analysis.py \
        [--train-csv data/meta/train.csv] \
        [--sign-map data/meta/sign_to_prediction_index_map.json] \
        [--asllex data/meta/asllex_signdata.csv] \
        [--wlasl data/meta/wlasl_v03.json] \
        [--out-dir data/meta]

Outputs: <out-dir>/coverage_report.md and <out-dir>/curriculum_candidates.csv
"""

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

# Kaggle gloss -> ASL-LEX EntryID base, for signs whose English label differs but the
# ASL sign is the same (dad = FATHER, potty = TOILET, ...). Normalized-key space.
ALIAS = {
    "another": "other", "beside": "nextto", "callonphone": "telephone", "dad": "father",
    "eye": "eyes", "food": "eat", "garbage": "trash", "glasswindow": "window",
    "grandma": "grandmother", "grandpa": "grandfather", "haveto": "must", "look": "lookat",
    "minemy": "my", "mom": "mother", "noisy": "noise", "nuts": "nut", "owie": "hurt",
    "police": "policeman", "potty": "toilet", "shoe": "shoes", "tooth": "teeth",
}

FEATURES = {"handshape": "Handshape.2.0", "location": "MajorLocation.2.0", "movement": "Movement.2.0"}

# v1 curriculum: beginner-relevant vocabulary chosen from the 250, with deliberate
# minimal-pair clusters (5-hand-at-head family signs, open/close/blue, milk/yes/shoe,
# black/red/nose, bye/wait/finish, girl/aunt/brother, happy/please, hungry/thirsty).
CURRICULUM_V1 = [
    "hello", "bye", "please", "thankyou", "yes", "no",
    "mom", "dad", "grandma", "grandpa", "boy", "girl", "child", "aunt", "brother",
    "dog", "cat", "bird", "fish", "nose",
    "red", "blue", "green", "yellow", "black", "white",
    "apple", "milk", "water", "drink", "food", "hungry", "thirsty",
    "happy", "sad", "mad", "sick",
    "go", "wait", "sleep", "look", "read", "open", "close", "finish",
    "book", "bed", "car", "shoe", "hat",
]


def normalize(gloss: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(gloss).lower())


def load_asllex(path: Path) -> dict[str, pd.Series]:
    """Normalized base gloss -> first ASL-LEX row (variant _1 preferred, dupes dropped)."""
    lex = pd.read_csv(path, low_memory=False, encoding="latin-1").drop_duplicates("EntryID")
    by_base: dict[str, pd.Series] = {}
    for _, row in lex.sort_values("EntryID").iterrows():
        base = normalize(re.sub(r"_\d+$", "", str(row["EntryID"])))
        by_base.setdefault(base, row)
    return by_base


def match_signs(signs: list[str], by_base: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {s: by_base[k] for s in signs if (k := ALIAS.get(s, normalize(s))) in by_base}


def confusable_pairs(features: dict[str, dict[str, str]]) -> list[tuple[str, str, list[str]]]:
    """Pairs sharing >= 2 of {handshape, major location, movement} (NaN never matches)."""
    pairs = []
    for a, b in combinations(sorted(features), 2):
        shared = [f for f in FEATURES
                  if pd.notna(features[a][f]) and features[a][f] == features[b][f]]
        if len(shared) >= 2:
            pairs.append((a, b, shared))
    return pairs


def strong_graph(pairs: list[tuple[str, str, list[str]]]) -> dict[str, set[str]]:
    """Adjacency over pairs sharing handshape AND location — the real minimal-pair ties."""
    adj: dict[str, set[str]] = {}
    for a, b, shared in pairs:
        if "handshape" in shared and "location" in shared:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def clusters_within(adj: dict[str, set[str]], members: list[str]) -> list[list[str]]:
    """Connected components of size >= 2 in the strong graph restricted to `members`."""
    member_set, seen, out = set(members), set(), []
    for start in members:
        if start in seen or start not in adj:
            continue
        comp, stack = [], [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            stack.extend(adj.get(node, set()) & member_set)
        if len(comp) >= 2:
            out.append(sorted(comp))
    return sorted(out, key=len, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, default=Path("data/meta/train.csv"))
    ap.add_argument("--sign-map", type=Path, default=Path("data/meta/sign_to_prediction_index_map.json"))
    ap.add_argument("--asllex", type=Path, default=Path("data/meta/asllex_signdata.csv"))
    ap.add_argument("--wlasl", type=Path, default=Path("data/meta/wlasl_v03.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/meta"))
    args = ap.parse_args()

    signs = sorted(json.load(args.sign_map.open()))
    train = pd.read_csv(args.train_csv)
    counts = train.groupby("sign").agg(
        n_examples=("sequence_id", "count"), n_signers=("participant_id", "nunique"))

    by_base = load_asllex(args.asllex)
    matched = match_signs(signs, by_base)
    unmatched = sorted(set(signs) - set(matched))
    feats = {s: {f: row[col] for f, col in FEATURES.items()} for s, row in matched.items()}

    wlasl_glosses = {normalize(e["gloss"]) for e in json.load(args.wlasl.open())}
    in_wlasl = {s: normalize(s) in wlasl_glosses for s in signs}

    pairs = confusable_pairs(feats)
    any_adj: dict[str, set[str]] = {}
    for a, b, _ in pairs:
        any_adj.setdefault(a, set()).add(b)
        any_adj.setdefault(b, set()).add(a)
    strong = strong_graph(pairs)

    missing = sorted(set(CURRICULUM_V1) - set(matched))
    assert not missing, f"curriculum signs without ASL-LEX match: {missing}"
    assert len(CURRICULUM_V1) == len(set(CURRICULUM_V1)) == 50
    clusters = clusters_within(strong, CURRICULUM_V1)
    assert len(clusters) >= 5, f"only {len(clusters)} minimal-pair clusters in curriculum"

    cur_set = set(CURRICULUM_V1)
    rows = []
    for s in CURRICULUM_V1:
        rows.append({
            "sign": s,
            "n_examples": counts.loc[s, "n_examples"],
            "n_signers": counts.loc[s, "n_signers"],
            "in_wlasl2000": in_wlasl[s],
            **feats[s],
            "confusables_in_curriculum": " ".join(sorted(strong.get(s, set()) & cur_set)),
            "confusables_strong_250": " ".join(sorted(strong.get(s, set()))),
            "n_confusables_any_250": len(any_adj.get(s, set())),
        })
    curriculum_df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curriculum_df.to_csv(args.out_dir / "curriculum_candidates.csv", index=False)

    write_report(args.out_dir / "coverage_report.md", signs, counts, matched, unmatched,
                 in_wlasl, strong, any_adj, clusters, curriculum_df)
    print(f"matched {len(matched)}/250, {len(pairs)} confusable pairs, "
          f"{len(clusters)} curriculum clusters -> {args.out_dir}/coverage_report.md")


def write_report(path, signs, counts, matched, unmatched, in_wlasl, strong, any_adj,
                 clusters, curriculum_df) -> None:
    n_wlasl = sum(in_wlasl.values())
    lines = [
        "# Coverage report: Kaggle asl-signs 250 x ASL-LEX 2.0 x WLASL-2000\n",
        f"Generated by `scripts/coverage_analysis.py`.\n",
        "## Dataset coverage\n",
        f"- {counts.n_examples.sum()} sequences, 250 signs, {counts.n_signers.max()} participants.",
        f"- Examples per sign: min {counts.n_examples.min()} ({counts.n_examples.idxmin()}), "
        f"median {int(counts.n_examples.median())}, max {counts.n_examples.max()} ({counts.n_examples.idxmax()}).",
        f"- Signers per sign: min {counts.n_signers.min()}, max {counts.n_signers.max()}. "
        "Coverage is near-uniform, so curriculum choice is driven by vocabulary relevance "
        "and phonological structure, not data volume.\n",
        "## ASL-LEX 2.0 match\n",
        f"- Matched {len(matched)}/250 signs ({len(ALIAS)} via the alias table in the script — "
        "same sign, different English label, e.g. dad=FATHER, potty=TOILET, haveto=MUST).",
        f"- **Unmatched ({len(unmatched)}):** {', '.join(unmatched)}.",
        "  - kitty/puppy/nap/wake likely share signs with cat/dog/sleep/awake (collisions, kept unmatched).",
        "  - store~SHOP, jeans~PANTS, sleepy~TIRED are near-misses left unmatched on purpose.\n",
        "## WLASL-2000 overlap\n",
        f"- {n_wlasl}/250 signs also appear in WLASL-2000 (future data-expansion signal).",
        f"- Not in WLASL: {', '.join(s for s in signs if not in_wlasl[s])}.\n",
        "## Confusables\n",
        "- *Any* tie: >= 2 shared of {handshape, major location, movement} (ASL-LEX `.2.0` coding).",
        "- *Strong* tie (minimal-pair grade): shared handshape AND major location.",
        f"- {sum(len(v) for v in strong.values()) // 2} strong pairs among the matched 250.\n",
        "## v1 curriculum (50 signs)\n",
        f"- {len(clusters)} deliberate minimal-pair clusters (strong ties within the 50):",
    ]
    for i, comp in enumerate(clusters, 1):
        lines.append(f"  {i}. {' / '.join(comp)}")
    lines.append("\n| sign | examples | signers | wlasl | handshape | location | movement | confusables in curriculum |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in curriculum_df.iterrows():
        lines.append(f"| {r['sign']} | {r.n_examples} | {r.n_signers} | {'y' if r.in_wlasl2000 else ''} "
                     f"| {r.handshape} | {r.location} | {r.movement} | {r.confusables_in_curriculum} |")
    lines.append("\n## Full 250-sign table\n")
    lines.append("| sign | examples | signers | wlasl | asl-lex | strong confusables (in 250) | any-tie count |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in signs:
        entry = matched[s]["EntryID"] if s in matched else "-"
        lines.append(f"| {s} | {counts.loc[s, 'n_examples']} | {counts.loc[s, 'n_signers']} "
                     f"| {'y' if in_wlasl[s] else ''} | {entry} "
                     f"| {' '.join(sorted(strong.get(s, set())))} | {len(any_adj.get(s, set()))} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
