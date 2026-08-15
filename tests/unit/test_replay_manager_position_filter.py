"""Regression test for a real bug found while extending decision-level
replay to 2024-25: from GW23 onward that season, FPL introduced a
selectable "Manager" pseudo-player mechanic (real managers with
position="AM", e.g. Fabian Hurzeler / Brighton, scored via separate mng_*
fields). These aren't part of the classic 15-player squad this optimizer
builds and previously caused a KeyError crash three layers deep (position
lookups in scoring/optimization code that only know GK/DEF/MID/FWD)."""
from __future__ import annotations

import pytest

from apex_fpl.backtesting.replay import run_gameweek
from apex_fpl.backtesting import vaastav_loader as vl


def test_2024_25_gw23_contains_manager_rows_in_raw_data():
    d = vl._season_dir("2024-25")
    if not (d / "merged_gw.csv").exists():
        pytest.skip("2024-25 merged_gw.csv not present; fetch it before running this test")
    rows = vl.load_merged_gw("2024-25")
    am_rows = [r for r in rows if r.get("position") == "AM"]
    assert am_rows, "if this ever fails, re-verify the 2024-25 archive still contains Manager rows before removing this test"


def test_run_gameweek_does_not_crash_on_a_manager_gameweek(tmp_path):
    d = vl._season_dir("2024-25")
    if not (d / "merged_gw.csv").exists() or not (d / "fixtures.csv").exists():
        pytest.skip("2024-25 data not present; fetch it before running this test")
    # GW35 previously crashed with KeyError: 'AM' before the position filter was added.
    result = run_gameweek("2024-25", 35, artifact_dir=tmp_path)
    positions = {p["position"] for p in result.recommendation["squad"]}
    assert positions <= {"GK", "GKP", "DEF", "MID", "FWD"}, f"non-standard position leaked into the squad: {positions}"
