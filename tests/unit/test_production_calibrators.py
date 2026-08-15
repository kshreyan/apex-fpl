from __future__ import annotations

import pytest

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.calibration import production_calibrators as pc
from apex_fpl.models.minutes.baseline import MinutesForecast


def _data_available():
    d = vl._season_dir(pc.CALIBRATION_FIT_SEASON)
    return (d / "merged_gw.csv").exists() and (d / "fixtures.csv").exists()


def test_get_minutes_calibrator_is_cached_singleton():
    if not _data_available():
        pytest.skip(f"{pc.CALIBRATION_FIT_SEASON} data not present; fetch it before running this test")
    c1 = pc.get_minutes_calibrator()
    c2 = pc.get_minutes_calibrator()
    assert c1 is c2, "get_minutes_calibrator() should be cached, not refit every call"
    assert c1.method in ("isotonic", "platt", "none")


def test_apply_minutes_calibration_preserves_appearance_and_minutes_fields():
    if not _data_available():
        pytest.skip(f"{pc.CALIBRATION_FIT_SEASON} data not present; fetch it before running this test")
    mfc = MinutesForecast(p_appearance=0.9, p_60_plus=0.85, expected_minutes_if_played=88.0, n_history_gws=10)
    out = pc.apply_minutes_calibration(mfc)
    assert out.p_appearance == mfc.p_appearance, "p_appearance was never validated for calibration and must be left untouched"
    assert out.expected_minutes_if_played == mfc.expected_minutes_if_played
    assert out.n_history_gws == mfc.n_history_gws
    assert 0.0 <= out.p_60_plus <= 1.0


def test_apply_minutes_calibration_never_exceeds_p_appearance():
    """Regression guard for the logical P(60+) <= P(appearance) constraint
    simulate_gameweek's minutes-bucket sampling relies on — calibration is
    fit independently of this constraint and could violate it without the
    defensive clamp."""
    if not _data_available():
        pytest.skip(f"{pc.CALIBRATION_FIT_SEASON} data not present; fetch it before running this test")
    for p_app in (0.5, 0.7, 0.9, 0.95, 1.0):
        mfc = MinutesForecast(p_appearance=p_app, p_60_plus=p_app, expected_minutes_if_played=90.0, n_history_gws=10)
        out = pc.apply_minutes_calibration(mfc)
        assert out.p_60_plus <= p_app + 1e-9, f"calibrated p_60_plus {out.p_60_plus} exceeded p_appearance {p_app}"


def test_apply_minutes_calibration_pulls_extreme_high_confidence_down():
    """Matches the Phase 5 finding: raw predictions near 1.0 are
    overconfident and should be pulled down by calibration, not left
    unchanged (mean 0.9775 -> 0.9175 in the validation reliability
    table)."""
    if not _data_available():
        pytest.skip(f"{pc.CALIBRATION_FIT_SEASON} data not present; fetch it before running this test")
    mfc = MinutesForecast(p_appearance=1.0, p_60_plus=1.0, expected_minutes_if_played=90.0, n_history_gws=15)
    out = pc.apply_minutes_calibration(mfc)
    assert out.p_60_plus < 1.0
