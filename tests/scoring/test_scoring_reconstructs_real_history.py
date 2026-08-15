"""Property-based validation of the scoring engine (spec Part XV) against
REAL historical ground truth, not synthetic examples.

Uses data/external/vaastav/2022-23/merged_gw.csv — audited in
docs/vaastav_archive_audit.md as grade-A trustworthy for this season, and
critically, from a season BEFORE defensive contributions existed, so our
scoring engine's Phase 2 scope (which excludes defensive contributions)
should reconstruct total_points exactly using only bonus as a given input.

If this ever fails, either our understanding of a scoring rule is wrong,
or the config in configs/seasons/2026_27.yaml has drifted from what these
rules actually were (both are worth knowing immediately).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from apex_fpl.rules import scoring

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "external" / "vaastav" / "2022-23" / "merged_gw.csv"

POS_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def _load_rows():
    if not DATA_PATH.exists():
        pytest.skip(f"historical data not present at {DATA_PATH}; fetch it before running this test")
    with DATA_PATH.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def rules():
    return scoring.load_scoring_rules("2026_27")


@pytest.fixture(scope="module")
def rows():
    return _load_rows()


def _to_events(row: dict) -> scoring.PlayerMatchEvents:
    return scoring.PlayerMatchEvents(
        position=row["position"],
        minutes=int(row["minutes"]),
        goals_scored=int(row["goals_scored"]),
        assists=int(row["assists"]),
        clean_sheet=row["clean_sheets"] == "1",
        goals_conceded=int(row["goals_conceded"]),
        own_goals=int(row["own_goals"]),
        penalties_saved=int(row["penalties_saved"]),
        penalties_missed=int(row["penalties_missed"]),
        yellow_cards=int(row["yellow_cards"]),
        red_cards=int(row["red_cards"]),
        saves=int(row["saves"]),
        bonus=int(row["bonus"]),
    )


def test_dataset_present_and_nontrivial(rows):
    assert len(rows) > 20000, "expected the full 2022-23 merged_gw.csv (~26k rows)"


def test_scoring_engine_reconstructs_total_points_for_full_season(rows, rules):
    """Empirically verified 2026-08-14: 0/26,505 mismatches across the full
    2022-23 season (a pre-defensive-contribution season, so this baseline's
    excluded DC term never has a chance to matter). Asserting exact match
    rather than a loose tolerance, since that's what was actually observed —
    a looser bound here would understate how strong this result is."""
    mismatches = []
    for row in rows:
        events = _to_events(row)
        computed = scoring.score_player_gameweek(events, rules)
        actual = int(row["total_points"])
        if computed != actual:
            mismatches.append((row["name"], row["GW"], computed, actual, row))

    sample = mismatches[:5]
    assert not mismatches, (
        f"{len(mismatches)}/{len(rows)} rows mismatched — the scoring engine "
        f"previously reconstructed this season's total_points exactly. "
        f"Sample mismatches (name, GW, computed, actual): {sample}"
    )


def test_scoring_engine_exact_match_on_large_sample_of_players_who_played(rows, rules):
    """Same exact-match guarantee, restricted to rows where the player
    actually played (minutes > 0) — the least trivial case."""
    played = [r for r in rows if int(r["minutes"]) > 0]
    assert len(played) > 5000
    mismatches = [
        row for row in played
        if scoring.score_player_gameweek(_to_events(row), rules) != int(row["total_points"])
    ]
    assert not mismatches, f"{len(mismatches)}/{len(played)} mismatched among rows where the player played"
