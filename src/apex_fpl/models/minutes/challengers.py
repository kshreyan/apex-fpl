"""Minutes model challengers (spec Part VIII / XXXVI mandatory baselines)
for the Phase 4 model tournament, competing against
apex_fpl.models.minutes.baseline.forecast_minutes (the flat-window
start-rate champion from Phase 2/3).

- always_90: the exact naive assumption spec Part LXVII explicitly warns
  against ("Do not assume 90 minutes") — kept deliberately as a mandatory
  sanity-check baseline it must be shown to lose to, not a serious
  candidate.
- persistence: predicts purely from the single most recent match — a
  classic simple baseline (spec Part XXXVI: "Recent [...] average" family,
  taken to its simplest extreme of window size 1).
- exponential_decay: a genuinely different mechanism from the champion's
  flat-window average — recency-weighted, so a purple patch or a recent
  injury-return dip is weighted more than an equally-old data point,
  without a hard cutoff.
"""
from __future__ import annotations

import numpy as np

from apex_fpl.models.minutes.baseline import (
    NEUTRAL_PRIOR_MINUTES_IF_PLAYED,
    NEUTRAL_PRIOR_P60,
    NEUTRAL_PRIOR_P_ANY,
    MinutesForecast,
)


def always_90(historical_minutes: list[int]) -> MinutesForecast:
    return MinutesForecast(p_appearance=1.0, p_60_plus=1.0,
                            expected_minutes_if_played=90.0, n_history_gws=len(historical_minutes))


def persistence(historical_minutes: list[int]) -> MinutesForecast:
    if not historical_minutes:
        return MinutesForecast(NEUTRAL_PRIOR_P_ANY, NEUTRAL_PRIOR_P60, NEUTRAL_PRIOR_MINUTES_IF_PLAYED, 0)
    last = historical_minutes[-1]
    p_any = 1.0 if last > 0 else 0.0
    p_60 = 1.0 if last >= 60 else 0.0
    exp_if_played = float(last) if last > 0 else NEUTRAL_PRIOR_MINUTES_IF_PLAYED
    return MinutesForecast(p_any, p_60, exp_if_played, len(historical_minutes))


def exponential_decay(historical_minutes: list[int], half_life_matches: float = 3.0, max_window: int = 15) -> MinutesForecast:
    """historical_minutes: chronological, oldest first. Weights the most
    recent `max_window` entries by exponential recency decay rather than a
    flat average — a genuinely different mechanism from the champion's
    forecast_minutes(), not just a hyperparameter variant of it."""
    window = historical_minutes[-max_window:] if max_window else historical_minutes
    n = len(window)
    if n == 0:
        return MinutesForecast(NEUTRAL_PRIOR_P_ANY, NEUTRAL_PRIOR_P60, NEUTRAL_PRIOR_MINUTES_IF_PLAYED, 0)

    ages = np.arange(n - 1, -1, -1)  # most recent entry (last in list) has age 0
    weights = 0.5 ** (ages / half_life_matches)
    minutes = np.array(window, dtype=float)
    appeared = minutes > 0
    played_60 = minutes >= 60

    w_sum = weights.sum()
    p_any = float((weights * appeared).sum() / w_sum)
    p_60 = float((weights * played_60).sum() / w_sum)

    w_appeared = weights[appeared]
    if w_appeared.sum() > 0:
        exp_if_played = float((weights[appeared] * minutes[appeared]).sum() / w_appeared.sum())
    else:
        exp_if_played = NEUTRAL_PRIOR_MINUTES_IF_PLAYED

    return MinutesForecast(p_any, p_60, exp_if_played, n)
