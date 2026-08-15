# Phase 5 Report — Calibration & Uncertainty Decomposition

**Run:** 2026-08-15. **Reproduce:** `PYTHONPATH=src python scripts/run_phase5_calibration.py`. Full results: `artifacts/phase5_calibration/phase5_summary.json`.

## Setup

**Calibration (spec Part XX).** Two real champion-model probability outputs were calibrated: the Phase 4b-promoted minutes model's P(60+ minutes), and the champion team model's clean-sheet probability (derived from the Dixon-Coles scoreline matrix). Isotonic and Platt calibrators were fit on a **dedicated calibration-fitting season, 2020-21** — genuinely separate from every other use in this project (not the Phase 4b tuning season, not any of the 4 decision-level significance seasons) — then evaluated on 3 held-out test seasons (2022-23, 2023-24, 2024-25), with a block-bootstrap (block = season+gameweek) significance test on whether calibration actually helps, exactly mirroring the promotion-decision methodology used throughout this project.

**Uncertainty decomposition (spec Part XVIII).** The Monte Carlo simulator was extended to expose per-sample minutes alongside points, enabling a law-of-total-variance split of each player's simulated point variance into "selection/minutes" (between played-state: no-appearance / sub / 60+) and "aleatoric" (within-state scoring randomness) components — computed from real simulated data for 2022-23 GW20, not asserted. A third uncertainty source, model (parameter) disagreement, is reported directly from Phase 4a's two actually-evaluated team models rather than invented.

## Result 1: minutes calibration — massive, highly significant improvement

| | log loss | Brier | ECE | slope | intercept |
|---|---|---|---|---|---|
| Raw | 0.7277 | 0.0970 | 0.0213 | 0.316 | -0.151 |
| Calibrated (isotonic) | **0.3072** | 0.0951 | **0.0063** | 1.029 | -0.030 |

Block-bootstrapped (calibrated − raw) log-loss difference: **-0.4367, 95% CI [-0.4913, -0.3860]** — decisively excludes zero. **PROMOTE.**

**Why this makes sense mechanistically, not just numerically:** the raw calibration slope of 0.316 (should be 1.0) means predictions are far more extreme than warranted. This is a direct consequence of how the Phase 4b-promoted `exponential_decay` model computes P(60+) — a weighted average of recent binary appearance outcomes. A player who started their last several matches gets a weighted average extremely close to (often exactly) 1.0, because the estimator has no mechanism to leave room for the unpredictable (a knock in training, a tactical rest, an unexpected red card next match). The reliability table confirms this concretely: **54% of all predictions fall in the [0, 0.1] bin**, with mean predicted probability 0.0062 against an empirical frequency of 0.0276 — a 4.4x understatement, in the single largest bin, for a metric (log loss) that punishes confident-and-wrong predictions severely. This single bin is very likely responsible for most of the raw log-loss inflation.

## Result 2: clean-sheet calibration — correctly rejected, and *why* is itself instructive

| | log loss | Brier | ECE | slope | intercept |
|---|---|---|---|---|---|
| Raw | 0.5282 | 0.1747 | 0.0159 | **1.001** | **0.015** |
| Calibrated (isotonic) | 0.5835 | 0.1772 | 0.0380 | 0.176 | -0.898 |

Block-bootstrapped difference: **+0.0535, 95% CI [+0.0178, +0.0966]** — entirely positive, calibration makes it **worse**. **DO NOT PROMOTE.**

The raw team-model clean-sheet probability was already excellently calibrated (slope 1.001, intercept 0.015 — almost exactly the ideal 1.0/0.0) — a genuine validation of the Phase 4a-tournament-winning team model's outputs. Applying isotonic regression on only 644 calibration-fitting observations (team-level events are far rarer than player-level ones — 20 per gameweek vs hundreds) let it overfit noise in that small sample; the fitted correction then generalized poorly and actively damaged an already-good signal on the test seasons. **This is a real, known weakness in `fit_calibrator`**, worth stating plainly: it selects between isotonic/Platt/none by log loss *on the fitting set itself*, which structurally favors isotonic (a more flexible, lower-bias-on-training-data method) even when it will generalize worse — a more rigorous design would use a further inner split purely for method selection. The reason this didn't cause harm here is that the **downstream promotion gate uses a genuinely held-out test set**, which caught the overfit and correctly blocked it. This is exactly what that safeguard is for, and it worked — but it's a stronger argument for fixing the method-selection step than a reason to ignore it.

## Result 3: uncertainty decomposition — a real, decomposed picture, and it corroborates Result 1

For 2022-23 GW20's top projected players (all real simulated data, not illustrative examples):

| Player | EP | Variance | Selection/minutes share | Aleatoric share | State probs |
|---|---|---|---|---|---|
| Erling Haaland | 5.40 | 16.35 | 0.14 | 0.86 | 7% none / 8% sub / 85% full |
| Ivan Toney | 4.84 | 23.36 | **0.43** | 0.57 | **30% none** / 70% full |
| Mohamed Salah | 4.31 | 9.69 | **0.00** | 1.00 | **100% full** |
| Harry Kane | 4.20 | 8.27 | 0.00 | 1.00 | 100% full |

Ivan Toney's high selection/minutes share (43%) correctly reflects his real rotation/return-from-suspension risk at that point in the season — the decomposition is doing real, sensible work distinguishing "uncertain whether he plays" from "uncertain how he'll perform given he plays." But **several nailed-on players (Salah, Kane, Trippier, Iwobi, Martinelli) show literally 0% selection/minutes uncertainty** — the minutes model assigns them P(60+) = exactly 1.0, collapsing an entire real source of uncertainty spec Part XVIII explicitly warns against collapsing. This is the *same overconfidence Result 1 just measured and fixed* showing up concretely in a downstream simulation: the raw (uncalibrated) minutes probabilities feeding the simulator understate exactly the uncertainty category they're supposed to represent. Results 1 and 3 aren't two separate findings — they're the same problem observed twice, once statistically and once in a live decomposition.

## Result 4: team-model disagreement as a model-uncertainty signal

Per-fixture disagreement between the Phase 4a champion and its tuned-but-not-promoted challenger (docs/phase4_tournament_report.md) ranged from 0.008 (Nottm Forest's away expected goals — near-total agreement) to 0.334 (Brentford's home expected goals) across the 12 fixtures in 2022-23 GW20 — a genuine, evidence-grounded signal for which fixtures carry more model uncertainty than others, rather than a single global number pretending every projection is equally trustworthy.

## What Phase XVIII's other named uncertainty sources are NOT covered here, honestly

Role, data, and schedule uncertainty are not computed — there is no evidence-backed way to derive them from what this project has built yet (no role-prediction model, no data-freshness scoring, no fixture-change-probability model). Listing them as "done" would be exactly the kind of unearned precision the spec warns against; they remain open gaps, tracked in `docs/fpl_gap_analysis.md`.

## Recommended next step (not done in this pass — flagged, not started)

The clear implication of Results 1 and 3 together is that **wiring the fitted minutes calibrator into the production simulation pipeline** (`src/apex_fpl/backtesting/replay.py`, currently using raw `exponential_decay` output) should materially improve both the simulator's honesty about selection uncertainty and — plausibly — decision-level FPL points, the same way Phase 4b's promotions did. This wasn't done automatically in this pass because it requires persisting the fitted calibrator as a reusable artifact (currently only fit transiently inside this script) and a fresh decision-level significance re-test (the same multi-season replay methodology as `docs/phase3_extended_replay_report.md`) to confirm it — real engineering and compute, not a one-line change, and worth a deliberate go-ahead rather than folding into this report silently.

## Test coverage added

11 new tests: `tests/unit/test_calibrator.py` (5, including a constructed-overconfidence recovery test and calibration-slope sanity checks) and `tests/unit/test_uncertainty.py` (6, including a law-of-total-variance identity check and two constructed edge cases — all-selection-variance and all-aleatoric-variance — that prove the decomposition behaves correctly at its extremes, not just on typical data). `src/apex_fpl/simulation/monte_carlo.py` was extended to carry per-sample minutes (needed for the decomposition); the full existing test suite, including the vectorized-scoring-matches-engine property test, still passes unchanged.
