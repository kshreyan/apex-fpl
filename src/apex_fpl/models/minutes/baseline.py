"""Naive minutes/appearance baseline model (spec Part VIII).

The simplest honest starting point: recent start-rate. This is explicitly
NOT the target architecture — Part VIII calls for hazard/survival/mixture
models conditioned on rotation, injury, fixture congestion, tactical role,
etc. This baseline exists only to make the Phase 2 milestone run
end-to-end on real data before any of that sophistication is justified by
evidence (the spec's own stopping-rule principle: don't add complexity
before a simpler baseline has been tried and measured against it).
"""
from __future__ import annotations

from dataclasses import dataclass

# Neutral priors for players with no observable history (e.g. newly
# promoted/transferred). Deliberately conservative and mid-range, not
# tuned against any backtest — documented placeholders, per the World Cup
# repo's own "sensible neutral priors" pattern.
NEUTRAL_PRIOR_P60 = 0.30
NEUTRAL_PRIOR_P_ANY = 0.50
NEUTRAL_PRIOR_MINUTES_IF_PLAYED = 45.0


@dataclass(frozen=True)
class MinutesForecast:
    p_appearance: float  # P(minutes > 0)
    p_60_plus: float  # P(minutes >= 60)
    expected_minutes_if_played: float
    n_history_gws: int


def forecast_minutes(historical_minutes: list[int], lookback: int | None = 6) -> MinutesForecast:
    """historical_minutes: this player's minutes in recent gameweeks,
    STRICTLY before the target deadline, in chronological order. Only the
    trailing `lookback` entries are used if given (None = use all)."""
    window = historical_minutes[-lookback:] if lookback else historical_minutes
    n = len(window)
    if n == 0:
        return MinutesForecast(NEUTRAL_PRIOR_P_ANY, NEUTRAL_PRIOR_P60, NEUTRAL_PRIOR_MINUTES_IF_PLAYED, 0)

    appearances = [m for m in window if m > 0]
    p_any = len(appearances) / n
    p_60 = sum(1 for m in window if m >= 60) / n
    exp_if_played = float(sum(appearances) / len(appearances)) if appearances else NEUTRAL_PRIOR_MINUTES_IF_PLAYED
    return MinutesForecast(p_any, p_60, exp_if_played, n)
