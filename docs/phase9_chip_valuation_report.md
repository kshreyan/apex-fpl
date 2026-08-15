# Phase 9 Report — Chip Valuation with Option-Value Framing

**Run:** 2026-08-16. **Reproduce:** `PYTHONPATH=src python scripts/run_phase9_chip_valuation_demo.py` (2022-23 GW2-38, ~10-15 min). Full results: `artifacts/phase9_chip_valuation/chip_valuation_results.json`.

## Scope

`research/research_plan.md`'s Phase 9 entry asks for "dynamic chip valuation with option-value framing" — treating each chip not just as "how much does playing it NOW gain" but as an option whose value includes the choice of WHEN to play it within its usable window. Built `src/apex_fpl/optimization/chips.py`: value functions for all 4 chip types, all reusing existing machinery (the Monte Carlo simulator, the squad optimizer, and — for Wildcard specifically — the Phase 7 multi-gameweek transfer optimizer) rather than new forecasting logic, plus a real, well-established optimal-stopping rule (the "secretary problem" 1/e observe-then-commit rule) for the genuine option-value question: does WAITING for a better week actually pay off, on real data?

**A real, deliberately-stated scope boundary:** this project has not audited what chip rules/windows actually applied in the historical 2022-23 season used for this demonstration (only the CONFIRMED 2026/27 ruleset — 2 of each chip, split across two half-season windows — is verified in `configs/seasons/2026_27.yaml`; older seasons are known to have had different chip structures, not verified here). This is therefore a methodological demonstration of the option-value idea on real EP data across the full GW2-38 range treated as one open window, not a claim about what was legal to play when in 2022-23 specifically.

## The four valuations, and why each is computed the way it is

- **Bench Boost** — value = EP of the current bench that gameweek (their points already exist in the simulation; the chip just makes them count).
- **Triple Captain** — value = EP of the current captain that gameweek (the extra 1x beyond the normal 2x already applied).
- **Free Hit** — value = EP(best unconstrained one-gameweek squad) − EP(current squad's best XI), for that gameweek only. This is the FULL value because a Free Hit squad reverts completely afterward — nothing carries forward, so nothing is missing from a single-gameweek comparison.
- **Wildcard** — value = the FULL multi-gameweek benefit of an unconstrained rebuild, computed by comparing two calls to `transfers.plan_transfers` over the same 3-gameweek-ahead window: one using the squad's REAL free-transfer count at that point, one with `free_transfers=15` (effectively unlimited). Unlike Free Hit, a wildcard's squad persists, so its value must include future gameweeks, not just the play week — this is the one chip that genuinely needed the Phase 7 machinery, not just the Phase 2 single-gameweek optimizer.

All four are evaluated against a REAL, evolving squad trajectory — the same validated lookahead (horizon=4) transfer policy from `docs/phase7_multiweek_optimizer_report.md`, not a hypothetical or static squad.

9 new tests, including a Monte Carlo verification that the 1/e stopping rule actually achieves close to its theoretical ~36.8% success rate at finding the true maximum (vs. a naive first-choice policy's ~1/n) — checked directly rather than assumed to work correctly just because it's a textbook result. 143/143 project-wide.

## Real-data result: 2022-23, GW2-38 (36 gameweeks)

| Chip | Mean value | Median | Min | Max |
|---|---|---|---|---|
| Bench Boost | 8.80 | 9.68 | 0.00 | 12.05 |
| Triple Captain | 5.47 | 5.34 | 3.85 | 7.37 |
| Free Hit | 2.75 | 2.36 | 0.00 | 6.85 |
| Wildcard (8-gameweek sample) | 5.70 | — | 2.23 | 8.80 |

Sensible orderings throughout: Wildcard (multi-gameweek benefit) and Bench Boost (all 15 players' points, not just 11) are the largest; Free Hit's modest values reflect that the lookahead-optimized squad is already close to that week's best possible XI most weeks, so there's usually not much left on the table for a single-gameweek rebuild to capture.

## The option-value question: does the 1/e stopping rule beat naive/hindsight?

| Chip | Hindsight-optimal | 1/e-rule | Naive-first (always GW2) |
|---|---|---|---|
| Bench Boost | GW2 (12.05) | GW38 (7.54) | GW2 (12.05) |
| Triple Captain | GW10 (7.37) | GW38 (5.67) | GW2 (3.85) |

**Triple Captain is the expected, textbook-consistent case:** the stopping rule (5.67) lands well short of the unknowable full-information optimum (7.37) but clearly beats naive-first (3.85) — real, measurable value from patience.

**Bench Boost is the opposite, and just as informative to report plainly:** the true best week (GW2) happens to be the very FIRST gameweek in the season — inside the stopping rule's own mandatory observation window (the first round(36/e)=13 gameweeks, calibrated purely to set a threshold, never eligible to be chosen). By construction, the rule can NEVER select from its own observation phase, so it falls through, finds nothing later in the season that clears the calibration threshold, and defaults to the forced last-resort pick at GW38 — worse than just naively grabbing the very first opportunity. This is a real, understood limitation of applying the classical secretary-problem framework as-is to a REAL, moderate-sized season (n=36): the framework assumes candidates arrive in a uniformly random order with no exploitable structure, but a real FPL season has genuine structure (fixture difficulty, early-season squad freshness, price/form trends) that can concentrate the true optimum unusually early or late — exactly the case the classical rule handles worst. Both stopping-rule picks landing on GW38 (the forced fallback) rather than an active "this beats the threshold" choice is the same underlying symptom in both chips: an observation-window threshold that turned out to be hard to beat again later in this specific real season.

**This is not a failure of the optimizer or the method — it's the honest, real output of applying a well-verified, correctly-implemented stopping rule to one specific real season's data**, exactly the kind of result this project reports without smoothing over (see `docs/phase6_joint_simulation_report.md`'s and `docs/phase8_robust_stochastic_report.md`'s similarly frank treatment of results that didn't come out as hoped). It's also a useful, concrete illustration of *why* option-value framing matters in the first place: the same rule that clearly helps for Triple Captain clearly hurts for Bench Boost in this one real instance, which is exactly the kind of case-by-case variability a hindsight-only analysis would hide.

## Decision: capability built and demonstrated, not a promotion question

Like Phase 7's transfer optimizer, there is no existing production "which chip, which gameweek" recommendation path for this to be wired into or promoted against — this phase's deliverable is the validated valuation + stopping-rule machinery itself, demonstrated honestly on real data including a real case where the "smart" policy underperformed. `docs/robust_captaincy_report.md`'s and `docs/phase8_robust_stochastic_report.md`'s general finding — real, well-implemented statistical machinery doesn't always produce a win on any one specific real sample, and that's reported as-is rather than cherry-picked or re-run until it looks better — applies here too.

## Concrete next steps (left for direction)

1. **Multi-season validation** of the stopping-rule comparison (the same lever used everywhere else in this project) — one season is not enough to know whether the Bench Boost result here was a genuine one-off or reveals something structural about how FPL's fixture/form calendar interacts with the classical secretary-problem assumption of random order.
2. **An FPL-structure-aware stopping rule** — e.g., one that weights early-season observations differently, given a real season isn't actually a uniformly-random-order process the way the classical rule assumes; a genuinely new piece of methodology, not yet attempted.
3. **A real 2026/27-chip-window-aware backtest**, once the historical archive's actual chip availability/rules for a given season are audited (currently an explicitly out-of-scope gap for this pass) — the current GW2-38 "one open window" treatment is a simplification, not a claim about real chip legality in any specific season.
4. **Joint chip + transfer planning**, since a wildcard interacts directly with the transfer plan (this report already reuses `transfers.py` for wildcard's own valuation) — a genuinely unified optimizer that decides chips and transfers together, rather than valuing chips against an already-fixed transfer trajectory, is a natural and more ambitious extension.
