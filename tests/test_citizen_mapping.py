"""Gloss-mapping tests for map_asl_citizen.py against fixture split CSVs."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "map_asl_citizen.py"

FIXTURE_ROWS = {
    "train": [
        (101, "101-HELLO.mp4", "HELLO", "hello"),          # exact ASL-LEX entry
        (101, "101-MOTHER.mp4", "MOTHER", "mother"),       # entry of our 'mom'
        (102, "102-WAKE.mp4", "WAKE", "awake"),            # member-gloss alias
        (102, "102-XYLOPHONE.mp4", "XYLOPHONE", "xylophone_1"),  # outside our 250
        (103, "103-HELLO2.mp4", "HELLO", "hello_2"),       # different variant: excluded
        (103, "103-KITTY.mp4", "KITTY", None),             # no code -> gloss fallback
    ],
    "val": [(104, "104-DOG.mp4", "PUPPY", "dog_2"),        # variant mismatch via alias
            (104, "104-BIRD.mp4", "BIRD", "bird"),
            # our 'store' class deliberately has NO asllex entry (shop_1 = SHOPPING,
            # a different sign) — a coded Citizen STORE clip must NOT gloss-match in
            (104, "104-STORE.mp4", "STORE", "shop_1")],
}


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("citizen")
    csv_dir = tmp / "splits"
    csv_dir.mkdir()
    for split, rows in FIXTURE_ROWS.items():
        pd.DataFrame(rows, columns=["Participant ID", "Video file", "Gloss",
                                    "ASL-LEX Code"]).to_csv(csv_dir / f"{split}.csv",
                                                            index=False)
    out = tmp / "meta"
    subprocess.run([sys.executable, str(SCRIPT), "--csv-dir", str(csv_dir),
                    "--out-dir", str(out)], check=True, cwd=ROOT)
    return out


def test_mapping_rows_and_namespacing(mapped):
    m = pd.read_csv(mapped / "asl_citizen_mapping.csv")
    by_file = m.set_index("videofile")
    assert by_file.loc["101-HELLO.mp4", "canonical_label"] == "hello"
    assert by_file.loc["101-MOTHER.mp4", "canonical_label"] == "mom"    # inverse alias
    assert by_file.loc["102-WAKE.mp4", "canonical_label"] == "awake"    # label collision member
    assert by_file.loc["103-KITTY.mp4", "canonical_label"] == "cat"     # NaN code, gloss path
    # different ASL-LEX variants are visually different signs — never folded in
    assert "103-HELLO2.mp4" not in by_file.index
    assert "104-DOG.mp4" not in by_file.index
    assert "104-STORE.mp4" not in by_file.index          # null-entry class + code = other sign
    assert "102-XYLOPHONE.mp4" not in by_file.index
    assert m.participant_id.str.startswith("ac_").all()   # can never collide with PopSign
    assert set(m.split) == {"train", "val"}


def test_coverage_report_sections(mapped):
    report = (mapped / "asl_citizen_coverage.md").read_text()
    assert "5/9 clips matched" in report
    assert "## Variant mismatches (3 clips excluded)" in report
    assert "HELLO (hello_2): 1" in report and "PUPPY (dog_2): 1" in report
    assert "STORE (shop_1): 1" in report
    assert "XYLOPHONE: 1" in report                       # unmatched, listed for CITIZEN_ALIAS
    assert "curriculum signs covered: 4/50" in report


def test_missing_column_fails_loudly(tmp_path):
    bad = tmp_path / "splits"
    bad.mkdir()
    pd.DataFrame([{"Participant ID": 1, "Gloss": "HELLO"}]).to_csv(
        bad / "train.csv", index=False)
    proc = subprocess.run([sys.executable, str(SCRIPT), "--csv-dir", str(bad),
                           "--out-dir", str(tmp_path / "out")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0 and "lacks" in proc.stderr
