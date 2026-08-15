# Phase 8 Report — Robust and Stochastic Optimization (EV vs CVaR vs Mean-Variance/MAD)

**Run:** 2026-08-15/16. **Reproduce:** `PYTHONPATH=src python scripts/run_phase8_robust_tournament.py` (32 gameweeks, ~1hr — most of the time is CVaR's known slow MILP solves, not MAD's). Full results: `artifacts/phase8_robust_tournament/tournament_results.json`.

## Scope

`docs/robust_captaincy_report.md` already took CVaR (tail-risk-averse squad selection) through a full evidence chain to a final NOT-PROMOTED decision. Phase 8's job, per `research/research_plan.md` ("benchmark deterministic-EV against robust/stochastic/distributionally-robust variants; let evidence decide, per spec Part XXVI"), is to add a genuinely different risk formulation rather than re-litigate CVaR — a mean-variance / Mean Absolute Deviation (MAD) optimizer, `select_squad_mean_variance` in `src/apex_fpl/optimization/robust.py`, alongside the existing CVaR code.

**Why MAD, not true variance:** true portfolio variance is quadratic in the selection variables (would need a MIQP solver this project doesn't have). MAD is the classic linear proxy (Konno & Yamazaki 1991) — `dev_s >= points_s(x) - mean(x)` and `dev_s >= mean(x) - points_s(x)` are both genuinely linear in `x` because a squad's per-scenario total and its mean are both linear combinations of the same fixed per-player scenario data, so this stays an EXACT MILP on the same `scipy.optimize.milp`/HiGHS backend as everything else, not an approximation.

**Why this is a structurally different question from CVaR, not a repeat:** CVaR cares only about the worst alpha-fraction of scenarios — it is blind to dispersion everywhere else. MAD penalizes deviation from the mean symmetrically, on both the upside and the downside, everywhere in the distribution. A player who is highly inconsistent but never actually blanks (all the variance is big-upside spikes) would be penalized by MAD but not meaningfully by a low-alpha CVaR objective, and vice versa a player who is rock-steady except for one specific disaster tail would concern CVaR much more than MAD. Confirming these two behave differently on real data (not just in theory) is itself part of what this tournament tests.

`lambda_risk=0.5` and `alpha=0.2` (CVaR, unchanged from the earlier work) are both single, reasonable starting choices, not deeply grid-searched — an explicit limitation, matching how CVaR's own alpha was originally chosen.

## Correctness

7 new tests (`tests/optimization/test_robust_mad.py`), all passing, mirroring the CVaR test suite's structure: legality, the direct-definition check for `compute_mad`, a `lambda_risk=0` sanity check (must reduce EXACTLY to plain EV selection — a real correctness property of the linearization, not just "looks plausible"), the weak-dominance property (the MAD optimizer's squad must score at least as well as the EV squad on the (mean − λ·MAD) objective it directly optimizes), a disaster-prone-player sensitivity check, and the same time-limit/diagnostics regression coverage CVaR needed after its own real bug. 134/134 project-wide.

## Real-data result: 2022-23/2020-21/2023-24/2024-25, 31 gameweeks (same design as the CVaR replay)

| | EV | CVaR (α=0.2) | MAD (λ=0.5) |
|---|---|---|---|
| Mean realized points | 55.45 | 54.45 | 55.90 |
| Worst single gameweek | 23 | **33** | 27 |
| Bottom-25%-of-gameweeks average | 37.62 | **38.25** | 37.00 |
| Dispersion of REALIZED scores (MAD, across all 32 gameweeks) | **11.53** | 12.00 | 12.55 |

Block-bootstrapped (5,000 resamples):

| Comparison | Mean diff | 95% CI |
|---|---|---|
| CVaR vs EV | −1.00 | [−4.32, +2.45] |
| MAD vs EV | +0.45 | [−1.29, +2.42] |
| MAD vs CVaR | +1.45 | [−1.90, +4.68] |

**All three CIs include zero — neither robust variant shows a statistically significant mean-points advantage over EV, or over each other, on this 31-gameweek sample.** (The CVaR-vs-EV row exactly reproduces the original CVaR replay's own figure, `-1.00, [-4.32,+2.45]`, from `docs/robust_captaincy_report.md` — same seed, same design, a useful determinism/reproducibility cross-check rather than a new finding.)

## The interesting part: CVaR and MAD behave exactly as their math predicts, and differently from each other

This is the honest, non-obvious finding worth reporting even though neither cleared the significance bar. **CVaR shows real tail protection** — best worst-single-gameweek (33 vs EV's 23) and best bottom-25%-average (38.25) — consistent with its earlier-documented "real disaster-recovery pattern, just not yet significant" characterization. **MAD shows NO equivalent tail protection** — its worst gameweek (27) and bottom-25%-average (37.00) are both worse than CVaR's, and its bottom-25%-average is even slightly worse than plain EV's. This is exactly what the module's own docstring predicts: MAD penalizes dispersion everywhere, not specifically the downside tail, so there's no reason to expect it to protect worst-case outcomes the way a tail-focused objective does.

**A genuinely counter-intuitive result:** the MAD optimizer's squad has the HIGHEST dispersion of realized scores across the 32 real gameweeks (12.55), higher than even plain EV (11.53) — despite being built specifically to MINIMIZE dispersion. This is not a contradiction once the two notions of "dispersion" are separated: MAD minimizes dispersion WITHIN each single gameweek's simulated scenario distribution (a property of that one week's forecast uncertainty), which is a different quantity from dispersion of realized outcomes ACROSS different real gameweeks over a season (driven by genuine week-to-week variation in fixtures, form, and forecast accuracy — sources of variability the single-gameweek optimizer has no visibility into). Reported plainly rather than smoothed over, since it's a real, measured result and a useful caution against assuming a within-gameweek risk objective automatically produces season-level consistency.

## Decision: neither CVaR nor MAD promoted; both validated as doing exactly what they're designed to do

`select_squad` (EV) remains the production squad optimizer — this is now the SECOND risk-aware variant evaluated to a clear, evidence-based NOT-PROMOTED conclusion on real multi-season data, joining CVaR. Both are correctly implemented (verified independent of any one season's outcome) and behave exactly as their respective mathematical objectives predict on real data — this is a successful use of the research process (spec Part XXVI: let evidence decide) producing two honest negative results, not two failures to "finish" the phase. CVaR's real, if not-yet-significant, tail-protection signal remains the more promising thread of the two if either is revisited (more gameweeks, or combining the tail-protection idea with the now-validated Phase 7 multi-gameweek machinery, are both live options, left for direction).

## Concrete next steps (left for direction)

1. **More gameweeks / seasons**, the same lever that resolved Phase 4b's original inconclusive finding and Phase 7's lookahead-vs-myopic question — untested here given this report's scope, not assumed unnecessary.
2. **A lambda/alpha grid search** rather than the single starting values used here, now that the machinery for both is proven correct and reasonably fast.
3. **A combined CVaR-aware multi-gameweek transfer plan**, extending `transfers.py`'s objective beyond pure EV using the tail-protection idea that showed real (if not significant) signal here and in the original CVaR work — a genuinely new research question neither Phase 7 nor Phase 8 tested on its own.
