from __future__ import annotations

import pytest

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.field import ownership as own


def _data_available(season: str) -> bool:
    return (vl._season_dir(season) / "merged_gw.csv").exists()


def test_load_ownership_counts_matches_a_known_real_value():
    """Mohamed Salah's real GW1 2022-23 ownership count, read directly
    from the archive during development: 4,848,340. A direct real-data
    check, not a synthetic example."""
    if not _data_available("2022-23"):
        pytest.skip("2022-23 data not present; fetch it before running this test")
    counts = own.load_ownership_counts("2022-23", 1)
    assert counts["283"] == 4848340  # element id 283 == Salah, confirmed against players_raw.csv


def test_estimate_total_managers_converges_to_a_plausible_real_range():
    """Cross-checked by hand against 4 independent high-ownership players
    (Haaland, Trippier, Rashford, Salah) at 2022-23's final gameweek —
    all converged to within ~1% of each other (11.40M-11.50M), consistent
    with real-world knowledge that 2022-23 FPL had roughly 10-11M total
    entrants. A loose range here, not an exact pinned value, since this
    is a genuine estimate from real data, not a constant."""
    if not _data_available("2022-23"):
        pytest.skip("2022-23 data not present; fetch it before running this test")
    total = own.estimate_total_managers("2022-23")
    assert 10_000_000 < total < 13_000_000


def test_load_ownership_fractions_are_plausible_percentages():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 data not present; fetch it before running this test")
    fractions = own.load_ownership_fractions("2022-23", 38)
    values = list(fractions.values())
    assert all(0 <= v < 1.1 for v in values)  # a small allowance above 1.0 for early-season underestimation, not clamped (see module docstring)
    # the most-owned player that gameweek should be a very high, but not literally 100%, fraction
    assert 0.5 < max(values) < 1.0


def test_load_ownership_fractions_reuses_a_supplied_total_managers():
    if not _data_available("2022-23"):
        pytest.skip("2022-23 data not present; fetch it before running this test")
    fractions_a = own.load_ownership_fractions("2022-23", 20, total_managers=10_000_000)
    fractions_b = own.load_ownership_fractions("2022-23", 20, total_managers=20_000_000)
    # halving total_managers should roughly double every fraction
    shared_pid = next(iter(fractions_a))
    assert abs(fractions_a[shared_pid] - 2 * fractions_b[shared_pid]) < 1e-9
