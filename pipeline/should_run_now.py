#!/usr/bin/env python3
"""Cheap pre-deadline proximity check (Phase 13, Block 1.6).

pipeline.yml's daily anchor schedule and manual workflow_dispatch always
run the full pipeline, unconditionally -- they cover settlement
detection and calibration freshness, which don't depend on deadline
proximity at all. A second, hourly schedule entry exists purely to give
predict.py more chances to catch late-breaking team news (press
conferences, fitness tests) close to a deadline -- but running the full
pipeline (install deps, test suite, model) every single hour, all day,
every day, for that one purpose would be wasteful and would flood the
commit history. This script is the cheap gate the hourly tick checks
first: a single uncached probe fetch (predict.py's own pattern, see
pipeline/predict.py), not the full pipeline, deciding whether this
particular hour is worth the expensive steps at all.

Prints exactly "true" or "false" to stdout (nothing else), so the
workflow step can capture it directly as a step output.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from apex_fpl.data import bronze

from pipeline import gw_state as gs

CLOSING_WINDOW_HOURS = 24.0


def should_run_now() -> bool:
    bootstrap = json.loads(bronze.fetch_raw("bootstrap_static")[0])
    now = datetime.now(timezone.utc)
    target_gw = gs.next_prediction_gameweek(bootstrap, now)
    if target_gw is None:
        return False  # season over -- nothing an hourly check could catch
    fixtures = json.loads(bronze.fetch_raw("fixtures")[0])
    phase_info = gs.gameweek_phase(bootstrap, fixtures, target_gw, now)
    return phase_info.phase == gs.Phase.PRE_DEADLINE and phase_info.hours_until_deadline <= CLOSING_WINDOW_HOURS


if __name__ == "__main__":
    print("true" if should_run_now() else "false")
