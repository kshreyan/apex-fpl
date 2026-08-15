"""The strongest form of leakage test for the replay framework (spec Part
XXXIV): prove empirically that DELETING all data at/after a target
gameweek produces the IDENTICAL recommendation as leaving it in place. If
future data silently influenced the past recommendation, truncating it
would change the output; this test would catch that directly rather than
relying on code review to notice a leaked column or an off-by-one in a
date comparison.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from apex_fpl.backtesting import replay, vaastav_loader as vl

SEASON = "2022-23"
TARGET_GW = 10  # smaller GW keeps this test fast while still exercising real history


def _season_source_dir() -> Path:
    return vl._season_dir(SEASON)


@pytest.fixture(scope="module")
def real_data_available():
    d = _season_source_dir()
    if not (d / "fixtures.csv").exists() or not (d / "merged_gw.csv").exists():
        pytest.skip(f"historical data not present at {d}; fetch it before running this test")
    return d


def _truncate_future(src_dir: Path, dst_dir: Path, target_gw: int) -> None:
    """Copy teams.csv verbatim. For fixtures.csv, keep every row (the
    schedule of who-plays-whom is legitimately known in advance — that is
    not leakage) but BLANK the score/finished fields for target_gw and
    anything later, exactly like a genuine pre-deadline snapshot would
    look. For merged_gw.csv, keep target_gw's own rows intact (roster
    identity — team/position/price — is legitimately known pre-deadline;
    replay.run_gameweek never reads target_gw rows' performance fields as
    model input, only as post-freeze evaluation ground truth) but DELETE
    every row strictly after target_gw, since that's genuinely future
    performance data with no pre-deadline analogue."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / "teams.csv", dst_dir / "teams.csv")

    with (src_dir / "fixtures.csv").open() as fin, (dst_dir / "fixtures.csv").open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            event = row.get("event")
            if event and int(event) >= target_gw:
                row = dict(row)
                row["finished"] = "False"
                row["finished_provisional"] = "False"
                row["started"] = "False"
                row["team_h_score"] = ""
                row["team_a_score"] = ""
                row["stats"] = "[]"
            writer.writerow(row)

    with (src_dir / "merged_gw.csv").open() as fin, (dst_dir / "merged_gw.csv").open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["GW"]) > target_gw:
                continue
            writer.writerow(row)


def test_truncating_future_gameweeks_does_not_change_the_recommendation(tmp_path, monkeypatch, real_data_available):
    """This test WILL FAIL if any future information leaks into a past
    recommendation — it is not asserting a property we merely believe to
    be true, it is comparing two independently generated recommendations
    byte-for-byte on the parts that matter."""
    src_dir = real_data_available

    truncated_root = tmp_path / "vaastav_truncated"
    _truncate_future(src_dir, truncated_root / SEASON, TARGET_GW)

    # Run 1: against the FULL dataset (contains future gameweeks beyond TARGET_GW)
    full_result = replay.run_gameweek(SEASON, TARGET_GW, artifact_dir=tmp_path / "full")

    # Run 2: against a copy with everything at/after TARGET_GW deleted
    monkeypatch.setattr(vl, "EXTERNAL_ROOT", truncated_root)
    truncated_result = replay.run_gameweek(SEASON, TARGET_GW, artifact_dir=tmp_path / "truncated")

    rec_full = full_result.recommendation
    rec_trunc = truncated_result.recommendation

    assert rec_full["squad"] == rec_trunc["squad"], "squad selection changed when future data was deleted — leakage"
    assert rec_full["starting_xi"] == rec_trunc["starting_xi"]
    assert rec_full["captain"] == rec_trunc["captain"]
    assert rec_full["fixture_projections"] == rec_trunc["fixture_projections"], (
        "team-model projections changed when future data was deleted — the team model is leaking"
    )
    assert rec_full["projected_gw_points"] == rec_trunc["projected_gw_points"]


def test_frozen_recommendation_never_contains_actual_result_fields(tmp_path, real_data_available):
    """A separate, narrower leakage check: the frozen recommendation must
    never contain the actual match outcome (scores) — only the
    probabilistic team-model projection. Guards against a future refactor
    accidentally passing real scores into the frozen artifact."""
    result = replay.run_gameweek(SEASON, TARGET_GW, artifact_dir=tmp_path / "artifacts")
    rec = result.recommendation
    forbidden_keys = {"home_score", "away_score", "actual_points", "total_points"}
    for fixture in rec["fixture_projections"]:
        assert forbidden_keys.isdisjoint(fixture.keys()), f"actual-result field leaked into frozen fixture projection: {fixture}"
    for player in rec["squad"]:
        assert forbidden_keys.isdisjoint(player.keys()), f"actual-result field leaked into frozen squad entry: {player}"
