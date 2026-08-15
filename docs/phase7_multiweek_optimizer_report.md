# Phase 7 Report — Multiweek Transfer Optimizer

**Run:** 2026-08-15. **Reproduce:** `PYTHONPATH=src python scripts/run_phase7_rolling_horizon_replay.py` (2022-23 GW2-38, ~a few minutes). Full results: `artifacts/phase7_rolling_horizon_replay/replay_results.json`.

## What this is, and how it differs from everything built so far

Every prior phase's replay (`docs/phase3_replay_report.md` onward) deliberately picks "best XV from scratch" independently each gameweek — a clean way to test whether the underlying forecasting pipeline works, but not a real transfer strategy (`replay.py`'s own docstring says as much). Phase 7 builds the actual missing piece: `src/apex_fpl/optimization/transfers.py`, an exact MILP that evolves ONE persistent squad through real transfer decisions across a horizon of forecasted gameweeks, correctly modeling the 2026/27 rules confirmed in `configs/seasons/2026_27.yaml` — 1 free transfer/gameweek, banking up to 5, -4 points per transfer beyond the free allowance.

Two players (mean/EV) squad selection, existing formation/quota constraints, and a 2-week-old CVaR robust-optimization module are all reused/imported (`apex_fpl.optimization.squad`), not duplicated.

## The model

A single flat MILP per horizon call jointly decides, for every gameweek in the horizon: squad membership, starting XI, captain, AND the transfers needed to get there from the previous gameweek — with free-transfer banking modeled via a linear "sandwich" relaxation (`ft[t] <= max_free_transfers`, `ft[t] <= ft[t-1] - made[t-1] + paid[t-1] + 1`) that's provably exact at the optimum (higher `ft` only ever relaxes a future point-costing constraint, so the solver always pushes it to the true `min()` value — standard MILP practice, not an approximation glossed over). Every `plan_transfers()` call cross-checks its own internal LP bookkeeping (`paid`, bank) against an independent recomputation from squad-membership diffs and asserts they match — this caught a real sign error in the free-transfer recurrence during testing (see below).

Two explicit, honestly-flagged simplifications, both consequences of gaps already tracked in `docs/fpl_gap_analysis.md`, not new omissions:
- **Prices held fixed across a single optimization horizon** — no price-change model exists yet (Part XXII). In the intended receding-horizon usage pattern (re-solve every real gameweek, commit only that gameweek's decision), this is low-cost: the committed decision always uses that week's real, current price, never a stale or (worse) leaked future price.
- **The transfer-in candidate pool is shortlisted** (current squad + top 15/position by best-horizon EP) purely for MILP tractability — not a claim of global optimality over the full ~700-player pool.

`rolling_horizon_transfers()` is the receding-horizon driver: `horizon=1` gives a purely myopic rolling policy, `horizon>1` gives genuine multi-gameweek lookahead — both go through the exact same code path, so a myopic-vs-lookahead comparison differs only in one argument, not two implementations that could silently diverge.

## Two real bugs found and fixed during testing, not after

1. **A sign error in the free-transfer recurrence constraint.** The intended inequality `ft(t) <= ft(t-1) - made(t-1) + paid(t-1) + 1` was built with the `made(t-1)` term's sign flipped, which under-constrained `ft(t)` (letting it be too generous) whenever `t-2 >= 0`. Caught immediately by the module's own internal LP-vs-recomputed-paid assertion firing during a constructed test (`test_lookahead_beats_myopic_by_banking_a_free_transfer_for_a_double_opportunity`) — the exact kind of "assert your own internal accounting, don't just trust the search found something feasible" discipline this project has used since the CVaR `res.success`-vs-status bug.
2. **A currently-owned player can be legitimately absent from a later gameweek's candidate universe** (their team has no fixture that gameweek — a postponement, not a full blank gameweek) — `rolling_horizon_transfers` was silently dropping them from the optimization window entirely, which made `plan_transfers` reject the call (`current squad player(s) not present in players`). This surfaced on the FIRST real-data run, not in synthetic tests, because real 2022-23 fixture data actually has this pattern; synthetic tests hadn't covered it. Fixed by keeping every currently-owned player in the window even when absent from all of that window's universes, defaulting their EP to 0.0 (correct: they're still a squad member, they just don't score that week) using their most-recently-known position/team/price. A dedicated regression test (`test_rolling_horizon_keeps_currently_owned_player_absent_from_a_later_universe`) locks this in.

11 new tests, all passing (127/127 project-wide), including a flagship constructed scenario proving genuine lookahead value: a real FPL strategic pattern — banking this week's free transfer instead of a small +2 upgrade, to afford two simultaneous signings for a bigger future opportunity without a hit — where the multi-gameweek plan scores 502 against a myopic rolling policy's 500 (exactly the predicted +2 gap, derived by hand before running).

## Real-data result: 2022-23, GW2-38, 36 gameweeks

One evolving squad per policy, all three sharing the identical initial squad (chosen via the existing EV squad optimizer on GW2's forecast), realized against REAL historical points (hit costs are real point deductions applied to the actual score, matching true FPL rules — not simulated EP):

| Policy | Total points | Mean/GW | Transfers | Hits taken |
|---|---|---|---|---|
| Buy-and-hold (never transfer) | 1720 | 47.78 | 0 | 0 |
| Myopic rolling (horizon=1) | 1890 | 52.50 | 43 | 7 |
| Lookahead rolling (horizon=4) | 1899 | 52.75 | 36 | 0 |

Block-bootstrapped (5,000 resamples, block = gameweek):

| Comparison | Mean diff | 95% CI |
|---|---|---|
| Myopic vs buy-and-hold | +4.72 | **[+1.11, +8.25]** — excludes zero |
| Lookahead vs myopic | +0.25 | [-3.25, +3.58] — includes zero |
| Lookahead vs buy-and-hold | +4.97 | **[+0.78, +9.22]** — excludes zero |

**Making transfers at all — even a purely myopic, single-gameweek-lookahead policy — is a real, statistically significant improvement over a static squad.** This is an important sanity check as much as a finding: it confirms the whole transfer machinery (not just the optimizer, but the underlying EP forecasts feeding it) is doing genuinely useful work, not just moving points around.

**Lookahead vs myopic is NOT statistically distinguishable on this single season** — 36 gameweeks is the same order of sample size that left Phase 4b's original decision-level finding inconclusive before it was extended to 4 seasons (`docs/phase3_extended_replay_report.md`), so this is not a surprising place to land on a first pass.

**But there's a real structural difference the point-total CI doesn't capture: lookahead achieved a statistically tied score while taking ZERO hits (vs myopic's 7, totaling -28 points spent) and making 7 fewer transfers overall (36 vs 43).** This is exactly the mechanism the flagship unit test isolates: lookahead can see an upcoming multi-transfer opportunity coming and bank free transfers for it instead of spending them piecemeal and eating hits later — myopic, unable to see beyond the current week, has no way to know a hit is avoidable until it's already necessary. That this shows up as a real, verifiable pattern in the real 2022-23 season (not just the constructed test) is worth taking seriously even though the raw point-total CI doesn't clear significance yet.

## Decision: real capability built and validated; promotion to production judgment held pending more seasons

The optimizer itself is correct and tested independently of any one season's outcome (11 tests, including two real bugs caught before they reached real data). The "transfers help at all" finding is decisively confirmed. The "lookahead beats myopic" finding is genuinely promising — a real avoided-hits mechanism, not noise — but one season is consistent with this project's own repeated standard: not enough to claim decision-level significance on its own (see the CVaR and Phase 4b→3-extension precedents). This is NOT wired into a "production" recommendation path (no such path exists yet for multi-gameweek planning), so there is no champion/challenger promotion decision to make yet in the `artifacts/model_registry.json` sense — this phase's deliverable is the capability itself, validated once, with a clear next step.

## Addendum 2026-08-15 — extended to 4 independent seasons: lookahead vs myopic is now decisively confirmed

**Run:** `scripts/run_phase7_multi_season_replay.py`, following the exact same 4-independent-season design used throughout this project (2020-21, 2022-23, 2023-24, 2024-25 — 2021-22 held out as the tuning season, 2019-20 excluded for its incompatible schema). Each season gets its own fresh initial squad (chosen via the EV optimizer on that season's own GW2 forecast) and its own fully independent buy-hold/myopic/lookahead run — seasons are never mixed mid-replay. Full results: `artifacts/phase7_multi_season_replay/replay_results.json`.

Per-season totals (myopic / lookahead, out of ~37 gameweeks each):

| Season | Myopic total | Myopic hits | Lookahead total | Lookahead hits |
|---|---|---|---|---|
| 2020-21 | 1936 | 0 | 2081 | 0 |
| 2022-23 | 1879 | 4 | 1899 | 0 |
| 2023-24 | 1940 | 1 | 2171 | 0 |
| 2024-25 | 2113 | 1 | 2276 | 0 |

**Lookahead took zero hits in all 4 seasons.** Myopic took at least one hit in 3 of 4.

Pooled across all 147 gameweek observations (block-bootstrapped, 5,000 resamples, block = season+gameweek — the same convention as every other multi-season replay in this project):

| Comparison | Mean diff | 95% CI |
|---|---|---|
| Myopic vs buy-and-hold | +9.52 | **[+7.18, +12.01]** |
| Lookahead vs myopic | +3.80 | **[+2.05, +5.58]** — now excludes zero |
| Lookahead vs buy-and-hold | +13.32 | **[+10.63, +16.21]** |

Lookahead beat myopic in 87 of the 138 non-tied gameweeks (sign test p=0.0028, corroborating the bootstrap CI). Total transfers: myopic 153 (6 hits), lookahead 147 (0 hits).

**This resolves the single-season report's open question the same way Phase 4b's original 2-season decision-level finding resolved when extended to 4 seasons (`docs/phase3_extended_replay_report.md`) — more independent gameweek observations turned a real-but-statistically-invisible effect into a confirmed one.** The mechanism is exactly what the single-season report and the flagship unit test both pointed to: lookahead can see a future multi-transfer opportunity coming and bank free transfers for it instead of spending them piecemeal and eating hits later. Across 147 real gameweeks, this shows up as a complete absence of hits (0 vs myopic's 6) alongside a genuinely higher point total, not just a "same points, fewer hits" wash — the CI on raw points excludes zero on its own.

## Decision, updated: lookahead is the validated, recommended multi-gameweek transfer policy

`transfers.plan_transfers`/`rolling_horizon_transfers` with `horizon>1` (validated at horizon=4) is now confirmed, across 4 independent seasons and 147 real gameweeks, to significantly outperform both a static squad and a myopic one-gameweek-at-a-time policy. There is still no production multi-gameweek recommendation path for this to be "promoted into" in the champion/challenger sense `artifacts/model_registry.json` normally tracks — but the open question from the single-season report (is this real or noise?) is now closed: it's real. If/when a multi-gameweek recommendation feature is built, this result is the evidence base for defaulting it to lookahead (horizon>=4) rather than a myopic rolling policy.

## Concrete next steps (left for direction)

1. ~~Extend to the same 4 independent seasons~~ **DONE — see addendum above.**
2. **A genuine price-change model** (Part XXII, still NONE) would let a multi-gameweek plan reason about "buy before a price rise" — currently invisible to this optimizer by design, since prices are held fixed within a horizon.
3. **Widen the horizon** beyond 4 gameweeks and/or the shortlist beyond 15/position — the full run (4 seasons × 2 policies × ~37 gameweeks, ~290 MILP solves, 60s time limit each) completed without incident, suggesting there's room to go further, though this run didn't separately log per-gameweek solve time or `proven_optimal` status (`TransferPlan` carries both, but `rolling_horizon_transfers`'s per-step `GameweekPlan` doesn't yet surface them) — that instrumentation is a small, worthwhile addition before scaling up further, not something to assume from this run's fast wall-clock completion alone.
4. **Build an actual production multi-gameweek recommendation feature** using this now-validated policy — the natural next concrete deliverable, distinct from further research validation.
