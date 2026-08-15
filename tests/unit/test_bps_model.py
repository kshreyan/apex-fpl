from __future__ import annotations

from apex_fpl.models.bonus import bps_model


def _row(position="MID", minutes=90, goals=0, assists=0, cs=0, saves=0, gc=0, yc=0, rc=0, og=0, ps=0, pm=0, bps=20):
    return {
        "position": position, "minutes": str(minutes), "goals_scored": str(goals), "assists": str(assists),
        "clean_sheets": str(cs), "saves": str(saves), "goals_conceded": str(gc), "yellow_cards": str(yc),
        "red_cards": str(rc), "own_goals": str(og), "penalties_saved": str(ps), "penalties_missed": str(pm),
        "bps": str(bps),
    }


def test_fit_bps_models_produces_a_model_per_position_with_enough_data():
    rows = [_row(position=pos, goals=g % 3, bps=20 + g) for pos in bps_model.POSITIONS for g in range(30)]
    models = bps_model.fit_bps_models(rows)
    assert set(models.keys()) == set(bps_model.POSITIONS)
    for pos, m in models.items():
        assert m.n_train == 30
        assert m.residual_std >= 0


def test_fit_bps_models_skips_positions_with_too_little_data():
    rows = [_row(position="MID", bps=20)] * 5  # below the 20-row threshold
    models = bps_model.fit_bps_models(rows)
    assert "MID" not in models


def test_fit_bps_models_ignores_zero_minute_rows():
    played = [_row(minutes=90, bps=20 + i) for i in range(25)]
    unplayed = [_row(minutes=0, bps=999) for _ in range(25)]  # would corrupt the fit if included
    models = bps_model.fit_bps_models(played + unplayed)
    assert models["MID"].n_train == 25


def test_predict_mean_uses_fitted_coefficients():
    m = bps_model.PositionBPSModel(
        position="MID",
        coefficients={name: 0.0 for name in bps_model.FEATURE_NAMES} | {"goals_scored": 10.0},
        intercept=5.0, residual_std=1.0, n_train=100,
    )
    events = {name: 0.0 for name in bps_model.FEATURE_NAMES} | {"goals_scored": 2.0}
    assert m.predict_mean(events) == 5.0 + 2.0 * 10.0


def test_evaluate_bps_models_reports_mae_and_r2():
    train_rows = [_row(position="MID", goals=g % 3, bps=20 + 10 * (g % 3)) for g in range(50)]
    models = bps_model.fit_bps_models(train_rows)
    test_rows = [_row(position="MID", goals=g % 3, bps=20 + 10 * (g % 3)) for g in range(20)]
    result = bps_model.evaluate_bps_models(models, test_rows)
    assert "MID" in result
    assert result["MID"]["n"] == 20
    assert result["MID"]["mae"] < 1.0  # near-perfect fit on this synthetic deterministic relationship
