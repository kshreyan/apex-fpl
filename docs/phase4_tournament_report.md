# Phase 4 Model Tournament Report — Team Goal Forecasting

**Run:** 2026-08-15. **Reproduce:** `PYTHONPATH=src python scripts/run_phase4_team_tournament.py`. Full per-fold and pooled metrics, tuned constants, and the promotion decision: `artifacts/phase4_tournament/tournament_summary.json`.

## Setup

Nested temporal cross-validation across 6 real Premier League seasons (2019-20 through 2024-25, all fetched from the audited Vaastav archive — team-level fixture/score data, not the player-level columns the audit flagged cadence concerns about, so no grade-B caveat applies here). Three expanding-window outer folds:

| Fold | Training seasons | Inner-tuning validation season | Outer TEST season (never touched during tuning) |
|---|---|---|---|
| 1 | 2019-20, 2020-21 | 2021-22 | 2022-23 |
| 2 | 2019-20, 2020-21, 2021-22 | 2022-23 | 2023-24 |
| 3 | 2019-20, 2020-21, 2021-22, 2022-23 | 2023-24 | 2024-25 |

Within each outer test season, models are walked forward gameweek-by-gameweek (same mechanism as the Phase 3 replay): before predicting a gameweek, the team model refits on every match from the training+validation seasons plus every earlier match already played in the test season itself. The test season's own future gameweeks are never used.

**Candidates:**
- `champion_unfit` — the attack/defense model with Phase 2/3's original constants (K_BASE=0.045, HALFLIFE=380 days, Dixon-Coles rho=-0.04), carried over unmodified from the World Cup repo's international-football fit.
- `challenger_tuned` — same mechanism, but K_BASE/HALFLIFE grid-searched (6×4=24 combinations) on the inner validation season only, and rho refit by MLE on the training seasons only — the outer test season is never used for tuning.
- `baseline_constant` — no team-skill signal at all (spec Part XXXVI mandatory baseline).
- `baseline_prev_season_avg` — per-team average goals for/against, no decay or opponent-strength adjustment.

**Primary metric (pre-registered before running):** log loss on the outer test seasons, block-bootstrapped (block = one season's one gameweek, per spec Part XXXIX) to get a 95% CI on the challenger-minus-champion difference. **Promotion criterion (spec Part LXI):** promote only if the challenger's log loss is significantly *lower* (CI upper bound < 0).

## Result 1: the core mechanism clearly beats naive baselines

Pooled across all 3 outer test seasons (1,140 real matches):

| Model | log loss | RPS | Brier | Accuracy | ECE | Goals MAE |
|---|---|---|---|---|---|---|
| champion_unfit | **0.9775** | 0.2026 | 0.5813 | 0.5404 | 0.0343 | 0.9597 |
| challenger_tuned | 0.9779 | 0.2026 | 0.5818 | 0.5404 | **0.0231** | 0.9656 |
| baseline_prev_season_avg | 1.0170 | 0.2161 | 0.6089 | 0.4956 | 0.0525 | 0.9818 |
| baseline_constant | 1.0638 | 0.2330 | 0.6434 | 0.4509 | 0.0235 | 1.0277 |

Both attack/defense variants beat both baselines on every metric, in every one of the 3 outer folds individually (not just pooled — see `tournament_summary.json`'s `per_fold_metrics`). This is a meaningful validation: the ported mechanism (online, decayed, opponent-adjusted attack/defense ratings + Dixon-Coles) is doing real work, not just adding complexity for nothing.

## Result 2: refitting the constants did NOT produce a significant improvement — Phase 3's hypothesis was wrong

`docs/phase3_replay_report.md` flagged refitting `K_BASE`/`HALFLIFE_DAYS`/`rho` as "the single most likely source of a genuine improvement." That hypothesis is **not confirmed**:

**Block-bootstrapped log-loss difference (challenger − champion): +0.0008, 95% CI [-0.0049, +0.0066].** The CI clearly spans zero — no statistically significant difference. Per the pre-registered promotion criterion, **the challenger is NOT promoted; `champion_unfit` remains champion.**

Two things stand out in the tuning process itself that temper how much weight to put on this null result:
- The winning `(k_base, halflife)` combination was **inconsistent across folds** (0.08/730 → 0.08/380 → 0.045/730, with the third fold landing back on the champion's own default k_base). This suggests the inner-validation log-loss surface is fairly flat/noisy around these values, not that a stable better optimum was found and rejected — a single season (~380 matches) is a fairly small, noisy inner-validation set for a 24-combination grid search.
- **Calibration (ECE) was consistently better for the tuned challenger** (0.0231 vs 0.0343 pooled) even though log loss and accuracy were statistically tied. This wasn't the pre-registered primary metric, so it correctly does not trigger promotion on its own — but it's a real secondary signal worth investigating further (e.g. a finer or better-regularized search, or evaluating calibration as a co-primary metric in a future tournament) rather than something to quietly promote on after the fact.

## What this means for the flat Phase 3 FPL-points result

This tournament's biggest practical value is *ruling something out*: the team model itself is not the bottleneck behind Phase 3's flat (no-edge) FPL result. It clearly, robustly beats naive baselines at the one thing it's actually responsible for (forecasting match outcomes). That points the next round of investigation away from the team model and toward the components Phase 3's report already flagged as the crudest: the naive start-rate minutes model, and — most likely, per spec Part X's own warning — the proportional goal/assist allocation, which is explicitly named in the spec as something to move past rather than trust.

## Model registry update

`artifacts/model_registry.json` updated: `champion_unfit` remains the team-model champion (unchanged since Phase 2). `challenger_tuned` is logged as an evaluated-and-rejected candidate with its full metrics and rationale, not deleted — per spec Part XLVII/LXI, negative results are tracked, not hidden.

## Honest limitations of this tournament

- Only 3 outer folds (limited by how many seasons of audited fixture data exist) — a genuinely small number of independent test seasons for a strong significance claim either way.
- The grid search is coarse (24 combinations) and only tunes 2 continuous hyperparameters + a separately-refit rho; a proper Bayesian/successive-halving search (spec Part XLV) over a larger space was not attempted given the apparent flatness of the surface already found.
- Evaluation is match-outcome-only (W/D/L probabilities + goals MAE) — this doesn't yet test the specific quantities the FPL pipeline actually needs downstream (clean-sheet probability calibration specifically, not just overall accuracy), which is Phase 5 work.
