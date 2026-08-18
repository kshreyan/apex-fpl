"""Production calibrators — validated in Phase 5 (docs/phase5_calibration_report.md)
and wired into `apex_fpl.backtesting.replay`/`scenario_builder` (see
docs/phase5_production_wiring_report.md).

NOT currently wired into `scripts/run_production_recommendation.py` —
the module that actually generates the live, published site prediction.
This isn't an oversight to silently fix by importing it there: this
calibrator's isotonic map (see `MINUTES_HALFLIFE`) is fit specifically
against `apex_fpl.models.minutes.challengers.exponential_decay`'s raw
P(60+) distribution, but the live path currently runs
`apex_fpl.models.minutes.cold_start` instead (no in-season history
exists yet to fit `exponential_decay` on) — and `cold_start`'s own
isotonic-regression-on-real-GW1-outcomes IS already its calibration.
Applying this calibrator on top of `cold_start`'s output would apply a
correction fit for a different model's miscalibration pattern to the
wrong distribution. This module becomes a live-wiring task once the
live path transitions to `exponential_decay` in-season (the promotion
schedule's minutes-transition item), not before. Found and left
correctly unwired, not silently used, during a Block 1.1 unwired-module
audit that initially mis-flagged this as dead code — worth spelling out
here so the next read of this file doesn't repeat that mistake.

Fitted ONCE on a dedicated calibration-fitting season (2020-21, chosen
specifically because it is NOT used anywhere else in this project — not
the Phase 4b hyperparameter-tuning season, not any of the Phase 3
decision-level significance seasons) and cached, since — unlike the team
model, which is refit per-call on growing history — the calibrator does
not depend on which gameweek is currently being predicted.

Only the minutes model's P(60+) is calibrated here. Phase 5 explicitly
tested and REJECTED calibrating the team model's clean-sheet probability
(it was already well-calibrated; a naive isotonic fit overfit a small
644-observation sample and made it worse on held-out data) — that
rejection is a finding to respect, not something to quietly redo here.
P(appearance) was never calibrated in Phase 5 either (only P(60+) was
validated) and is intentionally left raw.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from functools import lru_cache

import numpy as np

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.calibration import calibrator as cal
from apex_fpl.models.minutes import challengers as minutes_challengers
from apex_fpl.models.minutes.baseline import MinutesForecast

CALIBRATION_FIT_SEASON = "2020-21"
MINUTES_HALFLIFE = 3.0  # must match the promoted Phase 4b champion's config
START_GW = 7


def _collect_minutes_calibration_data(season: str = CALIBRATION_FIT_SEASON) -> tuple[np.ndarray, np.ndarray]:
    all_rows = vl.load_merged_gw(season)
    all_rows = [r for r in all_rows if r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]
    gameweeks = [g for g in vl.season_gameweeks(season) if g >= START_GW]

    p_raw, y = [], []
    for gw in gameweeks:
        if not vl.fixtures_at_gw(season, gw):
            continue
        history_rows = [r for r in all_rows if int(r["GW"]) < gw]
        target_rows = [r for r in all_rows if int(r["GW"]) == gw]
        if not target_rows:
            continue

        minutes_by_player: dict[str, list[int]] = defaultdict(list)
        for r in sorted(history_rows, key=lambda r: int(r["GW"])):
            minutes_by_player[r["element"]].append(int(r["minutes"]))

        for r in target_rows:
            hist = minutes_by_player.get(r["element"], [])
            fc = minutes_challengers.exponential_decay(hist, half_life_matches=MINUTES_HALFLIFE)
            p_raw.append(fc.p_60_plus)
            y.append(1 if int(r["minutes"]) >= 60 else 0)
    return np.array(p_raw), np.array(y)


@lru_cache(maxsize=1)
def get_minutes_calibrator() -> cal.BinaryCalibrator:
    p_raw, y = _collect_minutes_calibration_data()
    return cal.fit_calibrator(p_raw, y)


def apply_minutes_calibration(mfc: MinutesForecast) -> MinutesForecast:
    """Returns a new MinutesForecast with p_60_plus calibrated. p_appearance
    is left untouched (never validated for calibration in Phase 5).
    Calibration is fit independently of the P(60+) <= P(appearance) logical
    constraint the rest of the pipeline relies on (simulate_gameweek's
    minutes-bucket sampling assumes it) — empirically, calibration pulls
    near-1.0 raw predictions down (e.g. mean 0.9775 -> 0.9175 in the
    validation reliability table), which can occasionally push a
    calibrated p_60_plus below the player's own p_appearance if the two
    were very close. Clamped defensively to preserve the logical subset
    relationship rather than let it silently break."""
    calibrator = get_minutes_calibrator()
    calibrated_p60 = float(calibrator.transform(np.array([mfc.p_60_plus]))[0])
    calibrated_p60 = min(calibrated_p60, mfc.p_appearance)
    return replace(mfc, p_60_plus=calibrated_p60)
