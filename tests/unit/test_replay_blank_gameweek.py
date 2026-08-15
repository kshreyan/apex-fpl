"""Regression test for the real blank-gameweek bug found while running the
Phase 3 replay: 2022-23 GW7 has zero fixtures (postponed following Queen
Elizabeth II's death, redistributed to later gameweeks). run_gameweek()
must raise a specific, informative BlankGameweekError rather than an
opaque failure from deep inside the simulator."""
from __future__ import annotations

import pytest

from apex_fpl.backtesting.replay import BlankGameweekError, run_gameweek
from apex_fpl.backtesting import vaastav_loader as vl


def test_2022_23_gw7_is_a_known_blank_gameweek():
    d = vl._season_dir("2022-23")
    if not (d / "fixtures.csv").exists():
        pytest.skip("historical data not present; fetch it before running this test")
    fx = vl.fixtures_at_gw("2022-23", 7)
    assert fx == [], "if this ever fails, the underlying data changed — re-verify GW7's real status before removing this test"


def test_run_gameweek_raises_blank_gameweek_error_not_a_generic_crash(tmp_path):
    d = vl._season_dir("2022-23")
    if not (d / "fixtures.csv").exists():
        pytest.skip("historical data not present; fetch it before running this test")
    with pytest.raises(BlankGameweekError):
        run_gameweek("2022-23", 7, artifact_dir=tmp_path)
