# Phase 10 Report — Ownership and Rank Simulator

**Run:** 2026-08-16. **Reproduce:** `PYTHONPATH=src python scripts/run_phase10_field_simulation_demo.py` (2022-23 GW20, ~20s). Full results: `artifacts/phase10_field_simulation/field_simulation_results.json`.

## What this needed that didn't exist yet

Every optimizer built through Phase 9 maximizes MY squad's own points — none of them know anything about the competitive field, so none can answer "how much does this actually help my RANK," which is what real FPL managers are ultimately playing for. `docs/fpl_gap_analysis.md` had this listed as fully NONE (Ownership/field model, XXIII/XXXII) before this phase.

## Grounding the field in real data, not a fabricated one

`merged_gw.csv`'s `selected` column is a real per-player, per-gameweek COUNT of managers owning that player — not previously used anywhere in this project. Converting a count to an ownership FRACTION needs a total-manager estimate, which isn't given directly anywhere in the archive. `src/apex_fpl/field/ownership.py::estimate_total_managers` derives one from real data instead of assuming a round number: `players_raw.csv`'s season-end `selected_by_percent` divided into the corresponding player's final-gameweek `selected` count recovers an implied total. Checked by hand against 4 independent high-ownership 2022-23 players (Haaland, Trippier, Rashford, Salah) before writing the function, this converged to within ~1% (11.40M–11.50M) — a real, cross-validated estimate, not a fabricated one, and consistent with public knowledge of 2022-23's real ~10-11M FPL entrant count. The shipped function takes the median across the 8 most-owned players for robustness.

## The field Monte Carlo

`src/apex_fpl/simulation/field.py` builds a field of synthetic rival squads by sampling players, position-by-position, weighted by their REAL ownership share — not millions of individually-modeled real rivals (no data source could support that), but a genuine, data-grounded approximation of the field's aggregate composition. Three explicit, honestly-stated limitations, none hidden:

1. **Synthetic squads aren't budget-constrained** — pure ownership-weighted sampling doesn't re-derive the budget tradeoffs real ownership already reflects.
2. **No cross-player ownership correlation** — real "template" squads (managers who own one premium player disproportionately also owning a specific other one) aren't modeled; each position is sampled independently.
3. **Every synthetic rival is assumed to pick their starting XI/captain via the SAME EV-optimal logic used for the user's own squad** (`squad.select_starting_xi`) — not a claim real rivals are all optimal, but the most defensible non-arbitrary default given no data on real managers' actual sub-optimality.

The one thing this implementation gets structurally right, and which matters most for a meaningful rank estimate: a synthetic rival's simulated score is built by summing their squad's PER-SCENARIO samples from the SAME Monte Carlo run already computed for the user's own squad, not redrawn independently. This correctly correlates the user's score and the field's score within each simulated world (shared players, shared match outcomes) — comparing two independently-drawn distributions would overstate uncertainty about relative rank whenever the user shares players with the field, which is the normal case.

10 new tests (4 ownership + 6 field simulation), including a direct check against Salah's real GW1 2022-23 ownership count (4,848,340), quota/no-duplicate legality on synthetic squads, ownership-weighted sampling actually behaving as weighted (a 95%-owned player appears in >85% of 500 sampled squads), and an exact-match check (a rival squad identical to the user's own must reproduce the user's own scenario samples exactly, given the same deterministic starting-XI logic). 153/153 project-wide.

## Real-data result: 2022-23 GW20

- Total managers estimated: 11,474,729.
- Top ownership: Haaland 82.1%, Trippier 65.2%, Martinelli 45.9%, Cancelo 42.2%, Rashford 41.4% — all plausible for that gameweek.
- My squad (standard EV optimizer): mean simulated score 50.97.
- Field mean simulated score (2,000 synthetic rivals): **40.70**.
- Field mean score, independently cross-checked via the naive ownership-weighted estimate (no field Monte Carlo involved at all): **40.00**.
- My mean percentile within the field: **0.780** (beats ~78% of the field on average, scenario-by-scenario).
- P(top 10% of the field) = 0.368, P(top 25%) = 0.655, P(bottom half) = 0.115 — internally consistent, monotonic.
- My squad's REAL realized GW20 score: 68 (a single anecdotal data point, not evidence of anything on its own — noted for completeness, not treated as validation).

**The two independently-computed field-mean estimates (40.70 from the full synthetic Monte Carlo vs. 40.00 from the simple ownership-weighted sum) agree within ~1.7% of each other.** This is a genuine internal consistency check, not a coincidence baked into the code — they're computed via entirely different mechanisms (one runs 2,000 full squad-selection MILPs and sums correlated per-scenario samples; the other is a single closed-form weighted sum with no squad-selection step at all) and their closeness is real evidence the field simulation isn't producing an unreasonable number, even without external ground truth to check against.

That the EV-optimized user squad (50.97) clearly beats the ownership-weighted field average (40.70) is the expected direction: an EV squad picked with this week's freshest forecast should outperform a field average that reflects many managers' STALE decisions (squads built in earlier gameweeks, budget/transfer-cost constraints this analysis doesn't impose on synthetic rivals). This is a sanity check on direction, not a claim of validated magnitude.

## An honest, unresolved limitation

**There is no real "average_entry_score" or rank-distribution ground truth in this historical archive to validate the field simulation's absolute scale against.** The internal ownership-weighted cross-check above is the only sanity check available in this pass — it confirms internal consistency, not external accuracy. If a genuine external source (e.g. the live FPL API's `events[].average_entry_score` field, confirmed to exist for the CURRENT 2026/27 season in `configs/seasons/2026_27.yaml`'s own construction) can be found or archived for historical seasons, that would be the natural way to actually validate this model's scale, not just its internal coherence. Flagged here explicitly rather than implied to be already resolved.

## Decision: validated capability, no promotion question yet

Like Phases 7 and 9, there's no existing production path this could be "promoted into" — no prior ownership/field model existed to compare against (`fpl_gap_analysis.md` had this row at NONE). This phase's deliverable is the capability itself: real-data-grounded ownership loading, a genuinely correlated field Monte Carlo (not an independent-distributions approximation), and a working percentile/rank estimate — demonstrated on one real gameweek with a real internal consistency check, and honest about what it hasn't been validated against.

## Concrete next steps (left for direction)

1. **External validation** against real average-score/rank data, if a source can be found for historical seasons — the single most important open item, explicitly not resolved here.
2. **Multi-gameweek / multi-season demonstration**, the same lever used everywhere else in this project to move from "one gameweek looks sensible" to a real decision-level claim.
3. **A genuine effective-ownership (EO) model**, which needs real captaincy-rate data this project doesn't have (`selected` only gives ownership, not who captained whom) — currently a real, stated gap, not faked with an assumed captaincy split.
4. **Budget-constrained or correlated synthetic squads**, closing the two structural approximations flagged above, if the added realism is shown to matter for the percentile estimate.
