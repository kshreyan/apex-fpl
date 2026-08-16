from __future__ import annotations

import pytest

from apex_fpl.models.minutes import cold_start as cs


def _rows(prices_minutes: list[tuple[float, int]]) -> list[dict]:
    return [{"position": "MID", "value": str(int(p * 10)), "minutes": str(m)} for p, m in prices_minutes]


def test_fit_produces_a_monotonic_relationship_between_price_and_p60():
    # cheap players mostly don't start, expensive players mostly do
    rows = _rows([(4.0, 0)] * 20 + [(4.5, 10)] * 20 + [(6.0, 70)] * 20 + [(9.0, 90)] * 20)
    model = cs.fit_cold_start_minutes_model(rows)
    cheap = model.predict(4.0)
    expensive = model.predict(9.0)
    assert expensive.p_60_plus > cheap.p_60_plus
    assert expensive.p_appearance > cheap.p_appearance


def test_predict_returns_a_valid_minutes_forecast():
    rows = _rows([(4.0, 0)] * 10 + [(8.0, 85)] * 10)
    model = cs.fit_cold_start_minutes_model(rows)
    forecast = model.predict(6.0)
    assert 0.0 <= forecast.p_appearance <= 1.0
    assert 0.0 <= forecast.p_60_plus <= forecast.p_appearance + 1e-9
    assert forecast.n_history_gws == 0
    assert forecast.expected_minutes_if_played > 0


def test_evaluate_reports_a_real_log_loss_number():
    train_rows = _rows([(4.0, 0)] * 30 + [(9.0, 90)] * 30)
    model = cs.fit_cold_start_minutes_model(train_rows)
    test_rows = _rows([(4.0, 0)] * 10 + [(9.0, 90)] * 10)
    result = cs.evaluate_cold_start_minutes_model(model, test_rows)
    assert result["n"] == 20
    assert result["log_loss"] < 0.5  # should be confident and correct on this clean-separated synthetic case


def test_real_historical_gw1_data_leave_one_season_out_beats_a_flat_baseline():
    """Real-data validation, mirroring every other model's evaluation
    convention in this project: fit on 3 of the 4 independent seasons'
    real GW1 rows, evaluate held-out log loss on the 4th, and confirm it
    beats a flat baseline (predicting the pooled base rate for every
    player regardless of price) -- i.e. price genuinely carries signal,
    not just assumed to."""
    from apex_fpl.backtesting import vaastav_loader as vl

    seasons = ["2020-21", "2022-23", "2023-24", "2024-25"]
    missing = [s for s in seasons if not (vl._season_dir(s) / "merged_gw.csv").exists()]
    if missing:
        pytest.skip(f"{missing} data not present; fetch it before running this test")
    gw1_by_season = {}
    for season in seasons:
        rows = vl.load_merged_gw(season)
        gw1_by_season[season] = [r for r in rows if r["GW"] == "1" and r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]

    held_out = "2024-25"
    train_rows = [r for s in seasons if s != held_out for r in gw1_by_season[s]]
    test_rows = gw1_by_season[held_out]

    model = cs.fit_cold_start_minutes_model(train_rows)
    result = cs.evaluate_cold_start_minutes_model(model, test_rows)

    import numpy as np
    base_rate = np.mean([int(r["minutes"]) >= 60 for r in train_rows])
    test_played60 = np.array([int(r["minutes"]) >= 60 for r in test_rows], dtype=float)
    p_flat = np.clip(base_rate, 1e-6, 1 - 1e-6)
    flat_log_loss = float(-np.mean(test_played60 * np.log(p_flat) + (1 - test_played60) * np.log(1 - p_flat)))

    assert result["log_loss"] < flat_log_loss
