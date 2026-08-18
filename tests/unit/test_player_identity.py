"""Real-data tests for the code-based cross-season identity join (Phase
13, Block 1.3) -- uses Mohamed Salah's real, committed 2022-23 and
2024-25 players_raw.csv rows as the primary case, not synthetic data,
because the real data makes the point the module exists for better than
a fabricated example could: his `code` (118748) is identical in both
seasons, but his `id` changed (283 -> 328) and even his `web_name`
changed ("Salah" -> "M.Salah") -- neither of the two "obvious" join
keys is actually safe, confirmed, not assumed.
"""
from __future__ import annotations

import pytest

from apex_fpl.backtesting import player_identity as pid


def _data_available(season: str) -> bool:
    return (pid.EXTERNAL_ROOT / season / "players_raw.csv").exists()


def test_salah_code_is_stable_across_seasons_but_id_and_name_are_not():
    if not _data_available("2022-23") or not _data_available("2024-25"):
        pytest.skip("players_raw.csv not present for 2022-23/2024-25; fetch it before running this test")
    map_2223 = pid.load_code_map("2022-23")
    map_2425 = pid.load_code_map("2024-25")

    salah_2223 = map_2223[118748]
    salah_2425 = map_2425[118748]

    assert salah_2223["web_name"] == "Salah"
    assert salah_2425["web_name"] == "M.Salah"  # the same real person, a different web_name
    assert salah_2223["id"] != salah_2425["id"]  # 283 vs 328 -- id is not a safe cross-season key
    assert salah_2223["second_name"] == salah_2425["second_name"] == "Salah"  # the actual invariant


def test_resolve_live_code_to_season_id_returns_the_real_2022_23_id():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 players_raw.csv not present; fetch it before running this test")
    assert pid.resolve_live_code_to_season_id(118748, "2022-23") == "283"


def test_resolve_live_code_to_season_id_returns_none_for_an_unknown_code():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 players_raw.csv not present; fetch it before running this test")
    assert pid.resolve_live_code_to_season_id(999999999, "2022-23") is None


def test_build_live_to_season_id_map_omits_players_absent_that_season():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 players_raw.csv not present; fetch it before running this test")
    live_elements = [
        {"id": 999, "code": 118748},  # Salah, real, present in 2022-23
        {"id": 1000, "code": 1},  # a code no real 2022-23 player has
    ]
    out = pid.build_live_to_season_id_map(live_elements, "2022-23")
    assert out == {"999": "283"}


def test_load_code_map_raises_a_clear_error_for_a_season_with_no_players_raw_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(pid, "EXTERNAL_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="isn't committed for every season"):
        pid.load_code_map("1999-00")
