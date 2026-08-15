# Uncertainty-Sensitive Decision Machinery — Captaincy Risk & Robust (CVaR) Optimization

**Run:** 2026-08-15. **Reproduce:** `PYTHONPATH=src python scripts/run_robust_captaincy_demo.py` (2022-23 GW20, ~140s, dominated by the CVaR MILP solve). Full results: `artifacts/robust_captaincy_demo/demo_summary.json`.

## What was built

**Captaincy risk (`src/apex_fpl/optimization/captaincy.py`, spec Part XXVIII):** distributional captain metrics computed from the real per-scenario samples the Monte Carlo simulator already produces — mean, median, std, P(blank), P(2+/6+/10+/15+/20+), P(no appearance) — plus three captain-selection modes (EV / risk-averse-25th-percentile / ceiling-90th-percentile) and a joint captain+vice-captain simulation that applies the real auto-vice-captain rule (vice's score doubles only if the captain gets 0 minutes; if both blank, the armband bonus is wasted). Explicitly does **not** compute effective ownership or expected rank gain — no ownership/field model exists yet (spec Parts XXIII/XXXII), and fabricating those numbers would violate the project's own standard.

**Robust optimization (`src/apex_fpl/optimization/robust.py`, spec Part XXVI):** a CVaR (Conditional Value at Risk) squad optimizer using the Rockafellar-Uryasev linear formulation — an exact MILP, not a heuristic, that maximizes the average squad-points outcome across the worst alpha-fraction of real correlated Monte Carlo scenarios. Proven correct with a direct mathematical test (`test_cvar_squad_weakly_dominates_ev_squad_on_cvar`): on the scenarios it's optimized against, the CVaR squad's CVaR must be at least as good as any other feasible squad's, including the EV-optimal one — this held on real random test data, which is meaningful evidence the LP formulation is implemented correctly, not just "looks plausible."

12 new tests, all passing (96/96 project-wide).

## Real-data result — corrected after a multi-seed scenario-count sweep (2026-08-15)

The first run (below, kept for the record) used a single fixed scenario subsample (seed 2026, n=400) and found the CVaR squad looking *worse* than the EV squad on every downside metric. That finding **does not replicate** under a proper multi-seed test and is now understood to have been an unlucky draw, not representative behavior — see "Corrected finding" immediately below before reading the original numbers.

### Corrected finding: scenario-count sweep (`scripts/run_cvar_scenario_sensitivity.py`)

Same fixed 2022-23 GW20 simulation (15,000 draws, identical across every trial), varying only the CVaR optimizer's scenario-subsample size and random seed:

| n_scenarios | seed | solve time | mean diff (CVaR − EV) | CVaR₀.₀₅ diff | CVaR₀.₁₀ diff | CVaR₀.₂₀ diff |
|---|---|---|---|---|---|---|
| 400 | 3000 | 129.8s | −0.18 | **+0.40** | **+0.26** | **+0.15** |
| 400 | 3001 | 59.6s | −0.24 | **+0.59** | **+0.46** | **+0.28** |
| 800 | 3000 | 187.2s | −0.10 | **+0.73** | **+0.64** | **+0.46** |
| 800 | 3001 | 102.9s | −0.42 | **+0.66** | **+0.53** | **+0.29** |
| 1600 | 3000 | 382.0s | −0.10 | **+0.73** | **+0.64** | **+0.46** |
| 1600 | 3001 | 110.9s | −0.10 | **+0.73** | **+0.64** | **+0.46** |

**All 6 trials show the CVaR squad beating the EV squad on every downside metric tested**, for a small, consistent mean-point cost (−0.10 to −0.42, i.e. 0.2%-0.8% of the ~51-point mean) — exactly the textbook robust-optimization tradeoff, and the opposite of the original single-seed result.

**A second, unplanned finding: the optimizer's answer converges.** n=800/seed=3000, n=1600/seed=3000, and n=1600/seed=3001 land on **bit-identical** results (same mean, same all three CVaR values to 10+ significant figures) — verified not to be a caching artifact, since their solve times differ substantially (187s, 382s, 111s), which would be an unlikely coincidence if the solver were silently reusing a cached result. The natural reading: by ~800-1600 scenarios (out of the full 15,000-draw distribution), different random subsamples of that size already converge to the *same* optimal squad for this real 699-player pool. Variance across trials also visibly shrinks with n (400: mean diffs range −0.18 to −0.24, a 0.06 spread; 800-1600: −0.10 to −0.42 for the two non-converged points, but 3 of 4 large-n trials agree exactly).

**Answering the original question directly: yes, more scenarios closes the gap** — not by gradually shrinking a persistent bias, but by revealing that the n=400/seed=2026 result was noise, and that the optimizer stabilizes to a real, sensible downside-protection answer once given enough scenarios (≥800 for this pool). Solve time is the real cost: 382s for the slowest n=1600 trial, growing (noisily, not cleanly linearly with n) — pushing to n=3000+ was not attempted given the sweep already found clear convergence by 1600, and diminishing returns don't justify open-ended compute here.

**Promotion status unchanged for now:** this is still evidence from one gameweek's fixed simulation, not the multi-gameweek decision-level replay the project's own promotion bar has required for every other change (Phases 4a/4b/the Phase 3 extension). The CVaR optimizer is not wired into `replay.py` — that remains gated on the multi-gameweek downside replay in the "next steps" section below, now on noticeably stronger footing than before this sweep.

### Original single-seed result (2026-08-15, superseded above — kept for the record, not deleted)

Ran on 2022-23 GW20 (699 candidate players, 15,000 simulations, CVaR optimized against a 400-scenario subsample at alpha=0.2, seed=2026):

| | EV squad | CVaR squad | Diff |
|---|---|---|---|
| Mean points (full 15,000-sim eval) | 50.97 | 50.08 | −0.89 |
| CVaR at alpha=0.05 | 26.83 | 26.59 | −0.24 |
| CVaR at alpha=0.10 | 29.76 | 29.45 | −0.30 |
| CVaR at alpha=0.20 | 33.42 | 33.03 | −0.39 |
| Worst single simulated outcome | 11.0 | 14.0 | +3.0 |

This looked like a real in-sample/out-of-sample gap at the time and was reported as the headline finding. The multi-seed sweep above shows it was specific to that one seed's particular 400-scenario draw, not a general property of the CVaR optimizer or of n=400. The lesson worth keeping: **a single scenario-subsample trial is not enough evidence to characterize a stochastic optimizer's out-of-sample behavior** — exactly the kind of single-sample overconfidence this project has flagged elsewhere (e.g. the Phase 2 milestone's n=1 gameweek caveat), just encountered in a new place.

## The multi-gameweek downside replay (2026-08-15) — the actual promotion gate, and the honest, mixed result

`scripts/run_cvar_multi_gw_replay.py`: 8 gameweeks per season (every 4th from GW7) across the same 4 independent seasons used everywhere else in this project (2020-21, 2022-23, 2023-24, 2024-25), 31 non-blank gameweek observations (2022-23 GW7 correctly skipped as the known blank gameweek), n=400 CVaR scenarios per gameweek, both squads' starting XI + captain scored against real outcomes.

**A real engineering problem surfaced immediately and was fixed first.** The first attempt (unbounded solve time) was killed after 2 gameweeks had consumed 33 minutes combined — some real problem instances (disproportionately early-season gameweeks with less differentiated player values, a known MILP branch-and-bound phenomenon) take 15+ minutes to prove optimal, not a bug. Two things were fixed: (1) a `time_limit` parameter added to `select_squad_cvar` bounding each solve to 90s, and (2) a real correctness bug caught in the process — the code checked `res.success`, which scipy only sets `True` when optimality is *proven*; a time-limited-but-feasible solution (status 1) would have been wrongly treated as failure and crashed, discarding a perfectly usable squad. Fixed to check status correctly and report the MIP gap transparently. Regression-tested (`test_time_limit_returns_a_legal_squad_even_if_not_proven_optimal`).

**Result:** 18 of 31 gameweeks hit the 90s time limit (mean reported MIP gap 2.86% when time-limited — the returned squad is provably within ~3% of optimal, not an arbitrary guess) and 13 solved to proven optimality.

| Metric | EV | CVaR | Diff |
|---|---|---|---|
| Mean realized points | 55.45 | 54.45 | −1.00 (95% bootstrap CI **[−4.32, +2.45]** — includes zero) |
| Worst single gameweek | 23 | 33 | **+10** (CVaR's floor was notably higher) |
| Bottom-25%-of-gameweeks average (decision-level empirical CVaR, k=8) | 37.62 | 38.25 | +0.62 (small) |
| Gameweeks won / lost / tied (CVaR vs EV) | — | — | **12 / 16 / 3** (sign test p=0.83 — not significant, mildly favors EV) |

**This is a genuinely mixed result, not a clean win for either side, and is reported as such.** The tail-protection story has real support: CVaR's single worst gameweek (33) was clearly better than EV's single worst (23) — a 10-point gap — and in 4 of EV's 5 worst gameweeks, CVaR scored noticeably higher the same week (e.g. +13, +10, +5), exactly the "protects the disaster scenario" behavior a CVaR objective targets. But the simple per-gameweek win/loss count actually leans the other way (CVaR won fewer gameweeks than it lost, though not significantly), and the bottom-25%-tail-average edge (+0.62) is small relative to what eight tail observations can reliably resolve. Splitting by solve quality (proven-optimal: mean diff +0.23, 13 obs; time-limited: mean diff −1.89, 18 obs) hints that time-limiting may be diluting the CVaR advantage somewhat, but even the proven-optimal subset still lost more gameweeks than it won (4 vs 7) — not a rescue, just a partial mitigating factor worth naming.

**Decision: NOT PROMOTED.** By the same bar every other change in this project has been held to (a block-bootstrap CI that excludes zero on the pre-committed primary metric), this does not clear it — the mean-points CI spans zero, and even the tail metrics, while directionally interesting, were never going to be resolvable to statistical significance from only 31 gameweeks (8 in the tail). This is not a rejection of the CVaR approach — the worst-single-gameweek and EV-disaster-recovery patterns are real and worth taking seriously — it's a statement that the current evidence doesn't meet this project's own bar for wiring it into `replay.py` by default. `champion_unfit`-derived EV selection remains the production squad optimizer.

**What would actually resolve this:** more gameweeks (31 is thin for tail statistics — spec Part XXXIX's own caution about research-overfitting on small samples applies symmetrically to "don't promote" as much as "don't reject"), and solving without the 90s time bound where compute allows, to remove the confound the proven-optimal-vs-time-limited split hints at. Both are real compute investments, not free — left for explicit future direction, consistent with how every other open item in this report has been handled.

## Captaincy findings

For GW20's actual top candidates, all three captain-selection modes (EV, risk-averse, ceiling) agreed on the same player (Erling Haaland — mean 5.35, median 6.00, P(blank)=0.12, P(no appearance)=0.07), because he dominated at every quantile tested for this particular gameweek's pool, not because the modes are equivalent — the synthetic tests (`tests/unit/test_captaincy.py`) directly construct cases where they diverge (a high-mean "boom-bust" player loses to a lower-mean "safe" player under the risk-averse mode, and wins under the ceiling mode), proving the mechanism works correctly even though this specific real gameweek didn't exercise the divergence.

The joint captain+vice-captain simulation showed a genuine, computed effect: mean captaincy bonus of 5.69 versus 5.35 for the captain considered alone — the vice-captain fallback (activating with probability 0.069, i.e. whenever Haaland got 0 minutes in a simulation) recovers some value in exactly the scenarios where blind captain-only accounting would show a total loss.

## Illustrative-only real outcome (n=1, not evidence)

The CVaR squad realized 77 actual points in GW20 versus the EV squad's 68 — a nice-looking result, but exactly the kind of single-gameweek comparison `docs/phase2_milestone_report.md` and every subsequent phase report in this project has explicitly warned against treating as evidence. It is not cited as support for either approach.

## Honest limitations and next steps

1. **The scenario-count question is resolved** (2026-08-15 sweep, above): n≥800 gives consistent, converged downside protection on real data; n=400 is usable but noisier between seeds.
2. **The multi-gameweek downside replay is done** (2026-08-15, above): 31 real gameweeks, mixed evidence, NOT PROMOTED. A real tail-protection signal exists (worst-single-gameweek, EV-disaster-recovery pattern) but doesn't clear this project's significance bar at n=31 with a 90s-bounded solve. More gameweeks and/or unbounded solve time are the concrete paths to a sharper answer, both real compute investments left for explicit direction.
3. Captaincy risk analysis has no such caveat — it's cheap to compute (no MILP, just percentiles of existing samples) and ready to use as-is.
