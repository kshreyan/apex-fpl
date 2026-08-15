# Phase 4b Model Tournament Report — Minutes & Attacking-Allocation Models

**Run:** 2026-08-15. **Reproduce:** `PYTHONPATH=src python scripts/run_phase4b_player_tournament.py`. Full results: `artifacts/phase4b_tournament/tournament_summary.json`.

## Motivation

`docs/phase4_tournament_report.md` concluded the team model was not the bottleneck behind Phase 3's flat FPL-points result, and pointed at the naive minutes model and the proportional attacking-allocation model (explicitly named in spec Part X as something to move past) as the more likely culprits. This tournament tests that directly.

## Setup

Nested split across 3 real seasons of player-level data (2021-22, 2022-23, 2023-24, fetched from the audited Vaastav archive): **2021-22 used only for inner hyperparameter tuning**, **2022-23 and 2023-24 are true held-out outer test seasons** (48,575 pooled real player-gameweek observations), walked forward gameweek-by-gameweek (GW7-38) exactly like the Phase 3 replay — history strictly before each gameweek, never that gameweek's own results. Team-model expected goals use the Phase 4-validated champion (default constants) fit on the full 6-season fixture history.

**Minutes candidates:** `champion_flat_lookback6` (Phase 2/3's baseline), `challenger_flat_tuned` (same mechanism, lookback grid-searched), `challenger_exp_decay` (recency-weighted, a genuinely different mechanism), `baseline_always_90` (the exact mistake spec Part LXVII warns against), `baseline_persistence` (last-match-only).

**Attacking candidates:** `champion_proportional_lookback6` (Phase 2/3's baseline), `challenger_proportional_tuned` (lookback grid-searched), `challenger_shrinkage` (Dirichlet/empirical-Bayes shrinkage toward a uniform prior — addresses the champion's known small-sample-noise weakness), `baseline_equal_split` (spec Part XXXVI mandatory "no information" baseline).

**Primary metrics (pre-registered):** minutes — binary log loss of P(60+ minutes) against actual outcome. Attacking — Poisson negative log-likelihood of allocated expected-goals against actual goals scored. **Promotion criterion:** block-bootstrapped (block = one season's one gameweek) challenger-minus-champion difference, promote only if the 95% CI is entirely negative (challenger strictly better).

## Result 1: both challengers promoted, with large and highly significant effects

| Component | Metric | Champion | Best challenger | Block-bootstrap diff (95% CI) | Decision |
|---|---|---|---|---|---|
| Minutes | log loss (P(60+)) | 1.2711 | **0.7208** (`challenger_exp_decay`, half_life=3.0 matches) | −0.537 [−0.594, −0.479] | **PROMOTE** |
| Attacking | goal Poisson NLL | 0.4099 | **0.1385** (`challenger_shrinkage`, α=10.0, lookback=15) | −0.275 [−0.295, −0.255] | **PROMOTE** |

Both are large effect sizes with confidence intervals nowhere near zero — a much clearer result than Phase 4a's team-model tournament (which found no significant improvement). `baseline_always_90` behaved exactly as the metric design predicts it should: catastrophically (log loss 25.1), confirming the evaluation setup correctly penalizes the naive mistake spec Part LXVII explicitly warns against. `challenger_shrinkage` also clearly beat `baseline_equal_split` (0.1385 vs 0.1664), confirming it's doing genuine player-specific work, not just collapsing to "no information."

**Why shrinkage wins so decisively, mechanistically:** Poisson NLL penalizes a near-zero predicted rate very harshly when the rare event happens anyway (a defender who's allocated ~0% goal share from a small sample, then scores). The raw proportional champion assigns exactly this kind of near-zero share to most defenders/low-minutes players; shrinkage's whole purpose is avoiding that failure mode. This is the expected, well-understood behavior of a shrinkage estimator, not a surprising artifact.

**Caveat on the attacking tournament's scope:** for tractability, a single tuned lookback value (15, chosen via the minutes flat-window tuning) was reused for the attacking model's history window rather than tuning it independently — a genuine simplification, not an oversight. A dedicated attacking-lookback grid search is a reasonable next refinement, though the shrinkage effect is large enough that it's very unlikely to be an artifact of this shared parameter.

## Result 2: closing the loop — do these component wins translate to better FPL decisions?

Component-level metrics (log loss, NLL) are not what a manager actually cares about — final FPL points are. Spec Part XXVII explicitly warns that prediction-quality improvements don't automatically imply decision-quality improvements. This was tested directly, not assumed: both champions were wired into `src/apex_fpl/backtesting/replay.py` as the new defaults, and Phase 3's season replay was re-run.

| Season | Pre-promotion (Phase 3 report) | Post-promotion | 95% bootstrap CI (post) |
|---|---|---|---|
| 2022-23 | Model 1,667 = Baseline 1,667 (tied, diff +0.00) | Model **1,780** vs Baseline 1,718 (diff +2.00/GW) | [−2.84, +6.48] |
| 2023-24 | (not previously run) | Model **1,749** vs Baseline 1,663 (diff +2.69/GW) | [−2.41, +7.81] |
| **Pooled (63 gameweeks)** | — | **mean diff +2.35/GW, model won 37, baseline won 24, tied 2** | **[−1.30, +5.78]** |

**Honest verdict: directionally consistent improvement across both seasons, not statistically significant by the pre-registered bootstrap criterion.** The pooled 95% CI still includes zero. A complementary one-sided sign test on the win/loss record (37 wins vs 24 losses, excluding 2 ties) gives p=0.0619 — suggestive, marginal, and explicitly **not** below the conventional 0.05 threshold this project has used consistently. This result is reported exactly as computed, not rounded up to "significant" because the point estimate happens to look good.

This is not a failure — it is the expected, honest shape of this kind of result: the component-level tournaments had ~48,575 independent observations and could detect even a moderate true effect with a tight CI; a season replay has only ~31-32 independent gameweek-level observations, which is a much lower-powered test for the same underlying improvement. The correct interpretation is "consistent with a real but modest decision-level benefit, not yet confirmed at the season-count available" — not "no effect" and not "confirmed effect." More seasons of replay (a natural Phase 3 extension) would sharpen this estimate.

## Model registry update

`artifacts/model_registry.json`: `minutes_model` and `attacking_allocation_model` champions updated to `challenger_exp_decay` and `challenger_shrinkage` respectively, with full evidence and rejection/promotion rationale for every candidate tested, not just the winner. The original Phase 2/3 artifacts (`artifacts/phase2_milestone/`, and the original `artifacts/phase3_replay/2022-23/` numbers as reported in `docs/phase3_replay_report.md`) are left untouched as the frozen historical record of what those specific model versions actually produced — re-running the same scripts now produces different numbers because the defaults changed, which is the correct and expected behavior of a champion/challenger system, not a contradiction to reconcile away.

## Honest limitations

- The attacking model's lookback wasn't independently tuned (see caveat above).
- Only 2 seasons of decision-level (FPL-points) replay exist post-promotion; the "not yet significant" verdict could change with more data in either direction.
- Bonus points and defensive contributions remain entirely unmodeled in the simulation (documented since Phase 2) — the FPL-points gap between projected and realized totals partly reflects this, not just forecast error in the promoted components.
- The minutes and attacking tournaments were run independently of each other (holding the other component fixed at its own champion) — a joint ablation (Part XL) isolating interaction effects between the two hasn't been done.
