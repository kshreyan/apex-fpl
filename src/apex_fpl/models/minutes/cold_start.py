"""Cold-start minutes forecasting (Phase 12) — for a gameweek with NO
prior-season-within-current-season history at all (GW1 of a new season
is the main case; this project's minutes champion,
`apex_fpl.models.minutes.challengers.exponential_decay`, needs recent
gameweek history it structurally cannot have on day one of a season).

Real historical data across 4 independent seasons (2020-21, 2022-23,
2023-24, 2024-25 — the same 4 used for every decision-level replay in
this project) shows a real, consistent, monotonic relationship between a
player's SEASON-OPENING price and their P(60+ minutes) at GW1 (checked
by hand before writing this module: cheapest-20%-by-price players start
60+ minutes only 10-26% of the time at GW1 across these 4 seasons;
priciest-20% players start 60+ minutes 52-58% of the time, in every
single season checked). This is intuitive — clubs price players using
the same information (expected role, reputation, fitness) a manager
would use to guess who starts — but it's verified here on real data, not
assumed.

Explicit limitation, stated rather than hidden: this model only uses
price, ignoring genuinely relevant GW1-specific signals a human manager
would also use (pre-season friendlies, confirmed injuries, new-signing
integration time) that aren't in this dataset. It is a real, validated,
useful FALLBACK for the specific situation where no better information
exists — not a claim to match a fully-informed human's GW1 judgment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from apex_fpl.models.minutes.baseline import MinutesForecast

EXPECTED_MINUTES_IF_PLAYED_DEFAULT = 66.5  # pooled real GW1 mean minutes-if-played across the 4 validation seasons


@dataclass
class ColdStartMinutesModel:
    p_appearance_by_price: IsotonicRegression
    p_60_plus_by_price: IsotonicRegression
    expected_minutes_if_played: float = EXPECTED_MINUTES_IF_PLAYED_DEFAULT

    def predict(self, price: float) -> MinutesForecast:
        p_appearance = float(np.clip(self.p_appearance_by_price.predict([price])[0], 0.0, 1.0))
        p_60_plus = float(np.clip(min(self.p_60_plus_by_price.predict([price])[0], p_appearance), 0.0, 1.0))
        return MinutesForecast(
            p_appearance=p_appearance, p_60_plus=p_60_plus,
            expected_minutes_if_played=self.expected_minutes_if_played, n_history_gws=0,
        )


def fit_cold_start_minutes_model(gw1_rows: list[dict]) -> ColdStartMinutesModel:
    """gw1_rows: real merged_gw.csv-style rows for GW1 of one or more
    seasons (a `position` field, `value` (price, tenths of £m), and
    `minutes`)."""
    prices = np.array([int(r["value"]) / 10.0 for r in gw1_rows], dtype=float)
    minutes = np.array([int(r["minutes"]) for r in gw1_rows], dtype=float)
    appeared = (minutes > 0).astype(float)
    played60 = (minutes >= 60).astype(float)

    p_app_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(prices, appeared)
    p_60_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(prices, played60)

    played_minutes = minutes[minutes > 0]
    expected_if_played = float(played_minutes.mean()) if len(played_minutes) > 0 else EXPECTED_MINUTES_IF_PLAYED_DEFAULT
    return ColdStartMinutesModel(p_app_model, p_60_model, expected_if_played)


def evaluate_cold_start_minutes_model(model: ColdStartMinutesModel, test_gw1_rows: list[dict]) -> dict:
    """Real held-out log loss for P(60+), matching the evaluation
    convention used for every other minutes model in this project
    (docs/phase4b_tournament_report.md)."""
    prices = np.array([int(r["value"]) / 10.0 for r in test_gw1_rows], dtype=float)
    minutes = np.array([int(r["minutes"]) for r in test_gw1_rows], dtype=float)
    played60 = (minutes >= 60).astype(float)

    p_pred = np.clip(model.p_60_plus_by_price.predict(prices), 1e-6, 1 - 1e-6)
    log_loss = float(-np.mean(played60 * np.log(p_pred) + (1 - played60) * np.log(1 - p_pred)))
    return {"n": len(test_gw1_rows), "log_loss": round(log_loss, 4)}
