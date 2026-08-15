# Phase 6 Report — Joint Simulation & Reduced-Form BPS Model

**Run:** 2026-08-15. **Reproduce:** `PYTHONPATH=src python scripts/run_phase6_joint_simulation_demo.py` (2022-23 GW20, ~9s). Full results: `artifacts/phase6_joint_simulation/demo_summary.json`.

## Scoping against real constraints

The spec's vision for BPS (Part XIV) is bottom-up simulation from underlying match actions (tackles, blocks, interceptions, shot locations). This project has no source for that event-level data — already flagged in `docs/fpl_gap_analysis.md` and `configs/seasons/2026_27.yaml`'s `unresolved_gaps`. The honest, buildable alternative, and exactly what that config file's own `resolution_plan` suggested: fit a **reduced-form model** of `bps` from the outcome-level events we do have (goals, assists, clean sheets, minutes, cards) on real historical data, and use it to drive a genuinely *joint* (within-match, ranked) bonus simulation — not a claim to have reverse-engineered the official action-level formula, but an evidence-based approximation of its aggregate effect.

## What was built

1. **`src/apex_fpl/models/bonus/bps_model.py`** — per-position linear regression of `bps` on 12 real event features, fit on 2022-23, validated on **held-out 2023-24**: R² 0.81–0.93, MAE 1.9–3.4 BPS points across all four positions. Genuinely strong out-of-sample predictive power, and the fitted coefficients are sensible and interpretable (e.g. FWD goals worth +24.6 BPS, MID +18.6, DEF +12.8 — a believable gradient; red cards −9 to −12.6; penalty saves +14.8 for GK). This closes the `unresolved_gaps` item with real evidence rather than an assumed number.

2. **`src/apex_fpl/simulation/joint_simulator.py`** — a challenger to the Phase 2 baseline simulator, kept separate rather than modifying the production module in place, with two structural improvements:
   - **Goals/assists are now a genuine multinomial allocation of the team's own simulated goal total**, not independent per-player Poisson draws. `sum(player_goals) == team_goals` in every single scenario, by construction — verified directly (`test_goal_allocation_matches_team_scoreline_exactly` re-derives the exact scoreline draw with the same seed and checks bit-for-bit agreement). This is precisely spec Part XV's reconciliation requirement, which the baseline simulator does not satisfy.
   - **Bonus points are now simulated at all** (the baseline always returns 0 — a documented gap since Phase 2). Each player's BPS is predicted by the model above plus scenario-level noise, then **ranked jointly** among real match participants using the official tie-break rule — verified against `premierleague.com/en/news/106533` rather than assumed from memory (a 2-way tie for 1st gives both players 3 and the *next* player 1, not 2 — confirmed by a direct web search before implementing, and locked in by 4 dedicated tie-break tests).

3. **A real bug found and fixed while testing, not after**: the bonus-ranking function initially assigned a real bonus point to a non-participant when fewer than 3 players had actually played in a (small, synthetic-test) match — the sentinel value for "didn't play" was sorting last but still receiving a rank and a bonus point. Fixed by excluding non-participants from ranking entirely rather than relying on sentinel values to sort correctly; caught by `test_non_appearing_player_never_gets_bonus`, which failed before the fix and passes after.

15 new tests (10 joint simulator + 5 BPS model), all passing; 116/116 project-wide.

## Real-data result: genuinely mixed, and the naive top-line number is misleading

Comparing per-player expected-points MAE against real GW20 outcomes, baseline (no bonus) vs joint (with bonus), across all 699 candidate players:

| | MAE |
|---|---|
| Baseline simulator (no bonus) | 0.979 |
| Joint simulator (with bonus) | 1.043 |
| Difference | **+0.064 (worse)** |

**Taken at face value, this says adding bonus simulation slightly hurts.** But that aggregate number is dominated by the 666 of 699 players (95.3%) who won zero bonus that gameweek, for whom any attempt to predict bonus can only add error, never reduce it (baseline predicts exactly 0, which is exactly right for non-winners). Breaking the same data down by whether a player actually won bonus tells a different, more informative story:

| | n | Baseline MAE | Joint MAE | Diff |
|---|---|---|---|---|
| Actual bonus-winners | 33 | 7.003 | 6.774 | **−0.229 (better)** |
| Non-winners | 666 | 0.681 | 0.759 | +0.079 (worse) |

**For the players who actually won bonus — the case the whole exercise exists to help — joint simulation is more accurate, not less.** The correlation between simulated mean bonus and actual bonus across all 699 players is 0.249: weak, but clearly positive and non-spurious, not noise. The aggregate MAE comparison above is the wrong primary metric for judging whether this is working — it's swamped by a large population where the correct answer is trivially "zero," masking real, correct-direction signal in the much smaller population that matters.

**Even so, the model badly under-predicts the actual scale of bonus for real winners** — the 10 actual bonus-winners' simulated mean bonus ranged only 0.01–0.88 against an actual value of 3 each. The likely cause: the reduced-form model's features (goals, assists, clean sheets, cards, minutes) don't capture the defensive/creative-action volume (tackles, interceptions, key passes, etc.) that plausibly drives a meaningful share of real BPS variance, especially for defenders and midfielders — exactly the event-level data this project doesn't have a source for, the same gap that motivated the reduced-form approach in the first place. Bonus prediction is correspondingly a harder target than raw BPS magnitude prediction (which had R² 0.81–0.93): ranking is a relative, high-variance quantity that small feature-set errors can flip.

## One more honest flag, not fully resolved this session

`fit_p_goal_assisted()` measured **P(a goal is assisted) = 0.893** from real 2022-23 league-wide data — notably higher than conventional football's commonly-cited ~65-70% assist rate. This may be a real, FPL-specific fact (spec Part XI explicitly warns "official FPL assists are not identical to conventional football assists," citing more generous crediting for rebounds/deflections/touch sequences), or it may indicate a measurement issue in how the ratio was computed. Used as measured, since it comes from real data and the spec's own framing makes it plausible rather than clearly wrong — but flagged here as worth independent verification before relying on it further, not asserted with more confidence than is warranted.

## Decision: NOT PROMOTED, correctly scoped as build-and-validate work

This is one gameweek's evidence — exactly the sample size this project has repeatedly warned isn't sufficient to draw a decision-level conclusion from (`docs/phase2_milestone_report.md` onward). The joint-simulation *machinery* (goal/assist reconciliation, tie-break-correct bonus ranking) is verified correct by direct tests, independent of any one gameweek's outcome, and is a genuine capability improvement over the baseline. The *BPS prediction quality*, however, is a real, measured limitation: decent aggregate accuracy, weak-but-real bonus-ranking signal, clearly not yet strong enough to claim as an unqualified improvement. `champion_unfit`/baseline-simulator-derived production behavior is unchanged; this stays a challenger, not wired into `replay.py`.

## Concrete next steps (not started, left for direction)

1. ~~A multi-gameweek validation~~ **DONE 2026-08-15 — see addendum below.**
2. ~~A walk-forward BPS model fit~~ **DONE — resolved in the same addendum.**
3. **Richer BPS features**, if a defensive-action data source is ever found (closing the root cause of the under-prediction for real bonus-winners) — or, short of that, exploring whether adding `ict_index`/`influence`/`creativity`/`threat` (already present in the historical archive, unused here) improves the reduced-form model's ranking accuracy without needing event-level data at all.
4. **A more targeted use of the confirmed signal** — see the addendum's "what this changes" section: a blanket swap of the baseline simulator is now decisively ruled out, but the bonus-winner-specific effect is real and stable enough to be worth exploiting more surgically (e.g. as a secondary signal restricted to a small high-bonus-probability subset, rather than a full point-estimate replacement for all 700+ candidates).

## Addendum 2026-08-15 — multi-gameweek validation: the effect is real, not a lucky gameweek, and now decisively confirmed in *both* directions

**Run:** `scripts/run_phase6_multi_gw_validation.py`, following the exact same 4-season × 8-gameweek design (every 4th gameweek from GW7) already established for the CVaR multi-gameweek replay (`docs/robust_captaincy_report.md`) — 2020-21, 2022-23, 2023-24, 2024-25 (2021-22 held out as the designated tuning season, 2019-20 excluded for its incompatible schema). 32 gameweek observations, 31 valid (2022-23 GW7 is a genuine blank gameweek, correctly skipped, matching every earlier replay in this project). Full results: `artifacts/phase6_multi_gw_validation/validation_results.json`.

This run also fixes the walk-forward leakage flagged as unresolved above: the BPS model and `p_goal_assisted` are now fit **once, strictly on 2021-22 only** — a season that never overlaps with any of the 4 test seasons, the same train/test season separation already used for the production calibrator. This barely moved the held-out (2023-24) metrics at all (R² 0.8828/0.8163/0.8127/0.9306 for GK/DEF/MID/FWD, vs 0.8866/0.8161/0.812/0.9328 when trained on all of 2022-23) — the single-gameweek demo's leakage simplification turns out not to have been doing meaningful work, which is itself informative.

Pooled, block-bootstrapped (5,000 resamples across the 31 gameweek observations — the same "block = gameweek" convention used everywhere else in this project):

| | mean diff (joint − baseline) | 95% CI |
|---|---|---|
| All players | +0.071 | **[+0.067, +0.076]** |
| Bonus-winners only | −0.229 | **[−0.241, −0.216]** |
| Non-winners only | +0.085 | **[+0.080, +0.091]** |

Pooled correlation(predicted mean bonus, actual bonus) = **0.257** — essentially identical to the single-gameweek estimate (0.249).

**Both single-gameweek findings replicate almost exactly, and both are now statistically decisive.** Every one of the 31 individual gameweeks shows the same sign in both directions — the bonus-winner diff ranges narrowly from −0.136 to −0.317, the non-winner diff from +0.065 to +0.123 — this is not noise averaging out to a coincidental mean; it's a stable, structural effect present in every single gameweek tested.

**What this changes:** the single-gameweek report's "not enough evidence" caveat is now resolved — with all 3 CIs excluding zero, this is no longer an open question. Two firm conclusions follow:

1. **A blanket swap of the joint simulator for the baseline (using its per-player point estimate for all ~700 candidates, e.g. as the default input to squad selection) is decisively NOT justified** — the aggregate MAE degradation is real and confirmed, not a sampling artifact. This closes that door rather than leaving it open pending more data.
2. **The bonus-ranking mechanism's signal for actual bonus-winners is also decisively real**, not a lucky draw — confirmed independently and consistently across 4 different seasons and 31 different real gameweeks. The magnitude is genuinely small in absolute point terms (a fraction of a point either way, on players who typically score 2-15), so this is a real-but-low-stakes effect, not a large one — worth stating plainly rather than oversold.

**Decision, updated:** still **NOT PROMOTED** as a replacement for the baseline simulator — that's now a confirmed, not just suspected, negative result. The joint-simulation machinery and its underlying bonus-ranking signal remain validated as *real* (both the reconciliation/tie-break correctness and the weak-but-genuine predictive signal for actual winners), just not in a form that improves the metric that would actually gate its use in squad selection. The next step listed above (#4, exploiting the signal more surgically rather than as a blanket replacement) is the honest path forward if this is revisited, not a re-run of the same aggregate comparison with more data — more data has now already resolved that question.
