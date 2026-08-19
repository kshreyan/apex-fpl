"""Defensive Contribution (DefCon) threshold-crossing forecast (Phase 13
Block 2.2) -- a real scoring category introduced 2025/26, live under a
DIFFERENT points structure this season (configs/seasons/2026_27.yaml's
`scoring.defensive_contribution`: DEF needs 10+ qualifying actions
(clearances+blocks+interceptions+tackles) for a flat 2 points; MID/FWD
need 12+ (the same four actions plus recoveries); GK is excluded
entirely). Not yet modeled anywhere in the live path.

**The evidence ceiling, stated once here rather than re-derived per
caller.** DefCon has existed for exactly ONE completed-enough season
(2025/26) as of this module's construction -- there is no cross-season
replication possible for this signal until 2026/27 itself accrues real
results, a calendar constraint, not an effort one. Every validation of
this module is necessarily single-season. Consumers must not describe
results from it as multi-season-confirmed the way this project's other
promoted components are.

**Design, deliberately the simplest defensible mechanism, matching this
project's precedent for count-forecasting (apex_fpl.models.minutes.
challengers.exponential_decay uses the same recency-weighted-mean
structure for a different count).** A player's qualifying-action count
this gameweek is forecast as an exponentially recency-weighted mean of
their own trailing action counts, then treated as a Poisson rate to
estimate P(count >= threshold). No fitted parameters beyond the SAME
half_life default already used for minutes (3.0) -- reused, not
re-tuned on DefCon data, specifically to avoid a second free parameter
chosen by looking at the only season available to validate against.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

DEFCON_THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12, "GK": None}
DEFCON_POINTS = 2
DEFAULT_HALF_LIFE = 3.0  # matches apex_fpl.models.minutes.challengers.exponential_decay's own default
DEFAULT_MAX_WINDOW = 15  # matches the same module


def defcon_action_count(row: dict, position: str) -> int:
    """row: a merged_gw.csv-style dict with clearances_blocks_interceptions,
    tackles, recoveries. DEF excludes recoveries from the qualifying
    count (configs/seasons/2026_27.yaml); MID/FWD include it; GK has no
    qualifying count at all (returns 0, but callers must gate on
    DEFCON_THRESHOLDS[position] being None, not rely on a zero count
    alone, since a real DEF/MID/FWD can also legitimately record zero)."""
    cbi = int(row.get("clearances_blocks_interceptions", 0))
    tackles = int(row.get("tackles", 0))
    if position == "DEF":
        return cbi + tackles
    if position in ("MID", "FWD"):
        return cbi + tackles + int(row.get("recoveries", 0))
    return 0


def forecast_defcon_hit_probability(historical_actions: list[int], position: str,
                                     half_life: float = DEFAULT_HALF_LIFE, max_window: int = DEFAULT_MAX_WINDOW) -> float:
    """historical_actions: chronological (oldest first) qualifying-action
    counts, strictly before the target gameweek. Returns P(this
    gameweek's count >= the position's threshold). 0.0 for GK (no
    threshold exists) or when historical_actions is empty (no history
    to forecast from -- an uninformative-prior 0.0, not a guess, same
    spirit as MinutesForecast's own empty-history neutral priors)."""
    threshold = DEFCON_THRESHOLDS.get(position)
    if threshold is None or not historical_actions:
        return 0.0

    window = historical_actions[-max_window:]
    n = len(window)
    ages = np.arange(n - 1, -1, -1)
    weights = 0.5 ** (ages / half_life)
    forecast_rate = float(np.average(window, weights=weights))
    if forecast_rate <= 0:
        return 0.0
    return float(poisson.sf(threshold - 1, forecast_rate))


def forecast_defcon_expected_points(historical_actions: list[int], position: str,
                                     half_life: float = DEFAULT_HALF_LIFE, max_window: int = DEFAULT_MAX_WINDOW) -> float:
    p_hit = forecast_defcon_hit_probability(historical_actions, position, half_life, max_window)
    return p_hit * DEFCON_POINTS
