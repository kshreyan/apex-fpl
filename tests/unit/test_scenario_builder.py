from __future__ import annotations

import pytest

from apex_fpl.backtesting import scenario_builder as sb
from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.backtesting.replay import BlankGameweekError


def _data_available(season: str, needs_merged_gw: bool = True) -> bool:
    d = vl._season_dir(season)
    if not (d / "fixtures.csv").exists():
        return False
    if needs_merged_gw and not (d / "merged_gw.csv").exists():
        return False
    return True


def test_build_gameweek_scenario_data_real_gw():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 data not present; fetch it before running this test")
    data = sb.build_gameweek_scenario_data("2022-23", 20)
    assert len(data.fixture_inputs) == 12  # GW20 2022-23 has 12 fixtures
    assert len(data.players_for_sim) > 400  # full player pool, not a handful
    assert all(m["position"] in sb.CLASSIC_POSITIONS for m in data.candidates_meta.values())
    assert len(data.target_rows) > 0


def test_build_gameweek_scenario_data_raises_on_blank_gameweek():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 data not present; fetch it before running this test")
    with pytest.raises(BlankGameweekError):
        sb.build_gameweek_scenario_data("2022-23", 7)  # known blank GW (Queen Elizabeth II postponement)


def test_build_gameweek_scenario_data_excludes_manager_rows():
    if not _data_available("2024-25"):
        pytest.skip("2024-25 data not present; fetch it before running this test")
    data = sb.build_gameweek_scenario_data("2024-25", 35)  # previously crashed with KeyError: 'AM'
    positions = {m["position"] for m in data.candidates_meta.values()}
    assert positions <= set(sb.CLASSIC_POSITIONS)
    assert "AM" not in positions
