# Phase 5 → Production: Wiring Minutes Calibration into `replay.py`

**Run:** 2026-08-15. **Reproduce:** the calibrator is now on by default in `run_gameweek()`; re-run `PYTHONPATH=src python scripts/run_phase3_replay.py <season>` for any of the 4 independent seasons.

## What was wired

`src/apex_fpl/calibration/production_calibrators.py`: the Phase 5-validated isotonic calibrator for minutes P(60+) (fit once on 2020-21, cached — not refit per gameweek, since unlike the team model it doesn't depend on growing in-season history) is now applied by default in `run_gameweek()` (`apply_minutes_calibration=True`). Only P(60+) is calibrated — P(appearance) was never validated for calibration in Phase 5 and is deliberately left raw. A defensive clamp (`calibrated_p60 = min(calibrated_p60, p_appearance)`) preserves the logical P(60+) ≤ P(appearance) constraint the simulator's minutes-bucket sampling relies on, since calibration is fit independently of that constraint and empirically does pull some near-1.0 predictions down (Phase 5's own finding). Clean-sheet calibration was **not** wired in — Phase 5 explicitly rejected it (overfit on a small fitting sample); that rejection is respected here, not silently redone.

## Re-ran the Phase 3 decision-level replay across all 4 independent seasons

| Season | Pre-calibration mean diff/GW | Post-calibration mean diff/GW | Delta |
|---|---|---|---|
| 2020-21 | +5.81 | +5.00 | −0.81 |
| 2022-23 | +2.00 | +2.35 | +0.35 |
| 2023-24 | +2.69 | +4.03 | **+1.34** |
| 2024-25 | +1.94 | +1.84 | −0.10 |
| **Pooled (127 GWs)** | **+3.118**, CI [+0.85, +5.27] | **+3.315**, CI [+1.09, +5.42] | **+0.197** |

**The overall pipeline (Phase 4b promotions + Phase 5 calibration combined) remains solidly decision-significant** — pooled 95% CI [1.09, 5.42] excludes zero, sign test p=0.0157 (75 wins, 50 losses, 2 ties).

## Isolating calibration's *specific* incremental effect — honest, not rounded either direction

Two seasons improved after calibration (2022-23, 2023-24 — the latter by a full +1.34/GW), two got slightly worse (2020-21, 2024-25). A season-level paired bootstrap on the 4 deltas gives **mean +0.195/GW, 95% CI [-0.52, +0.98]** — includes zero. **This specific increment is not statistically distinguishable from noise at the evidence available.**

This comparison is honest but coarser than it should be: the correct test would be a per-gameweek block bootstrap on (post − pre) exactly like every other significance test in this project, but the pre-calibration per-gameweek artifact files were overwritten when this rerun regenerated `artifacts/phase3_replay/<season>/` in place. **This is a real process gap** — the pre-calibration season summaries survive as reported numbers in `docs/phase3_extended_replay_report.md`, but the underlying per-gameweek pairs needed for a fine-grained bootstrap do not. Noted here so it isn't repeated: future "before/after" production changes should copy the prior artifact directory aside before re-running, not just cite its summary.

## Interpretation — why this result is plausible, not just "inconclusive and shrug"

Calibration reshapes probability *spread and confidence* (making overconfident near-certain predictions more honest); it does not necessarily reshuffle the *mean-point ranking* a greedy expected-value-maximizing squad optimizer selects on. If a player's calibrated P(60+) drops from 0.98 to 0.92 but he's still comfortably the highest-expected-points option at his price, the optimizer picks him either way — calibration's honesty gain doesn't change that decision. Its value is more likely to matter for uncertainty-*sensitive* decisions this project hasn't built yet: captaincy risk tolerance (spec Part XXVIII), robust/distributionally-robust optimization (Part XXVI), and rank-probability field simulation (Part XXXII) — all cases where *how* uncertain a projection is matters, not just its mean. That the current squad optimizer is largely insensitive to this specific improvement is informative, not a failure of the calibration work.

## Decision: kept wired in as default

Calibration stays on (`apply_minutes_calibration=True` default) because: (1) it is strongly validated and significant at the component level on its own terms (Phase 5), (2) there is no evidence it harms decision-level outcomes (the pooled result with calibration remains significant, matching or exceeding the pre-calibration result in 2 of 4 seasons), and (3) it is a prerequisite for future uncertainty-sensitive work (Phases 8/28/32) that cannot use honest probabilities it doesn't have. The inconclusive *incremental* decision-level test is reported plainly rather than used to justify reverting a change that's independently well-supported.

## Registry and tracking updates

`artifacts/model_registry.json`: minutes model entry updated to note calibration is now wired into production, with this report's honest incremental-effect finding, not an oversold "confirmed to help decisions" claim.
