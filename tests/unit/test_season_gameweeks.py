"""Regression test for a real gap found while extending Phase 3 to more
seasons: 2019-20 (COVID-19 disrupted) is numbered gameweeks 1-29, then
jumps straight to 39-47 after the restart — FPL never reused 30-38. A
loop over range(1, 39) silently skips 9 real, played gameweeks for this
season. vaastav_loader.season_gameweeks() exists specifically so callers
discover the real numbering instead of assuming a contiguous range."""
from __future__ import annotations

import pytest

from apex_fpl.backtesting import vaastav_loader as vl


def test_2019_20_has_the_known_covid_gameweek_gap():
    d = vl._season_dir("2019-20")
    if not (d / "fixtures.csv").exists():
        pytest.skip("2019-20 fixtures.csv not present; fetch it before running this test")
    gws = vl.season_gameweeks("2019-20")
    assert gws[0] == 1
    assert gws[-1] == 47
    assert 30 not in gws and 38 not in gws, "if this ever fails, re-verify 2019-20's real numbering before removing this test"
    assert len(gws) == 38, "38 real gameweeks were played that season despite the numbering gap"


def test_normal_season_is_contiguous_1_to_38():
    d = vl._season_dir("2023-24")
    if not (d / "fixtures.csv").exists():
        pytest.skip("2023-24 fixtures.csv not present; fetch it before running this test")
    gws = vl.season_gameweeks("2023-24")
    assert gws == list(range(1, 39))


def test_2022_23_correctly_omits_the_known_blank_gw7():
    """Cross-check against the Phase 3 blank-gameweek finding
    (tests/unit/test_replay_blank_gameweek.py): GW7's fixtures were
    postponed and rescheduled to other event numbers, so event=7 never
    appears in fixtures.csv at all for this season — season_gameweeks()
    should reflect that as an absence, not a crash or a silent gap."""
    d = vl._season_dir("2022-23")
    if not (d / "fixtures.csv").exists():
        pytest.skip("2022-23 fixtures.csv not present; fetch it before running this test")
    gws = vl.season_gameweeks("2022-23")
    assert 7 not in gws
    assert len(gws) == 37
    assert gws[0] == 1 and gws[-1] == 38
