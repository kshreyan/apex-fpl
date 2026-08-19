"""Unit tests for apex_fpl.rules.chip_windows -- the real 2026/27
two-half chip structure (Phase 13 Block 2.7), loaded directly from
configs/seasons/2026_27.yaml, not re-derived here."""
from __future__ import annotations

from apex_fpl.rules import chip_windows as cw


def test_loads_all_8_windows():
    windows = cw.load_chip_windows()
    assert len(windows) == 8
    assert {(w.name, w.half) for w in windows} == {
        ("wildcard", 1), ("freehit", 1), ("bboost", 1), ("3xc", 1),
        ("wildcard", 2), ("freehit", 2), ("bboost", 2), ("3xc", 2),
    }


def test_wildcard_and_freehit_not_usable_in_gw1():
    windows = cw.load_chip_windows()
    assert cw.active_window("wildcard", 1, windows) is None
    assert cw.active_window("freehit", 1, windows) is None


def test_bboost_and_triple_captain_usable_in_gw1():
    windows = cw.load_chip_windows()
    assert cw.active_window("bboost", 1, windows) is not None
    assert cw.active_window("3xc", 1, windows) is not None


def test_first_half_chips_expire_at_gw19_no_carryover_into_gw20():
    windows = cw.load_chip_windows()
    w = cw.active_window("wildcard", 19, windows)
    assert w is not None and w.half == 1
    assert cw.active_window("wildcard", 20, windows).half == 2
    # GW19's window must not still be "active" at GW20 -- a fresh half-2
    # window opens instead, distinct chip inventory, not a carryover.
    assert cw.active_window("wildcard", 19, windows) is not cw.active_window("wildcard", 20, windows)


def test_gameweeks_remaining_counts_inclusive_of_current_gw():
    windows = cw.load_chip_windows()
    w = cw.active_window("bboost", 19, windows)
    assert w.gameweeks_remaining(19) == 1  # last gameweek of the window: exactly 1 left, this one
    w2 = cw.active_window("bboost", 1, windows)
    assert w2.gameweeks_remaining(1) == 19  # full first-half window


def test_gameweeks_remaining_is_zero_outside_the_window():
    windows = cw.load_chip_windows()
    w = cw.active_window("wildcard", 2, windows)
    assert w.gameweeks_remaining(25) == 0  # half-1 window queried against a half-2 gameweek


def test_half_for_gameweek():
    assert cw.half_for_gameweek(1) == 1
    assert cw.half_for_gameweek(19) == 1
    assert cw.half_for_gameweek(20) == 2
    assert cw.half_for_gameweek(38) == 2
