# Phase 11 Report — Decision-Focused Research

**Run:** 2026-08-16. **Reproduce:** `PYTHONPATH=src python scripts/run_phase11_decision_focused_tournament.py` (~5 min). Full results: `artifacts/phase11_decision_focused_tournament/tournament_results.json`.

## Scope

`research/research_plan.md`'s Phase 11 entry (spec Part XXVII) asks for Track A (prediction-focused) vs Track B (decision-focused) vs Track C (hybrid), "compared on held-out decision regret without sacrificing calibration." A full decision-focused-learning implementation would differentiate through the squad-selection MILP itself (SPO+ and similar techniques) — a substantial, fragile undertaking on its own, distinct from everything else built this project. This phase implements a smaller, well-understood, genuinely decision-focused technique instead: **shrinkage**, tuned directly against realized decision quality rather than prediction accuracy.

- **Track A** — the existing champion pipeline's raw EP, unmodified, fed straight to the squad optimizer (exactly what every prior phase already does).
- **Track B** — each player's EP shrunk toward their position's median (`adjusted = median + shrinkage × (raw − median)`), with `shrinkage` TUNED on 2021-22 (the project's standing tuning-only season) to directly maximize TOTAL REALIZED squad points across 8 tuning gameweeks — a genuinely decision-focused objective, not a prediction-loss metric.
- **Track C** — a simple 50/50 blend of Track A and Track B's EP.

`src/apex_fpl/models/decision_focused.py`, 5 new tests (158/158 project-wide), including a constructed scenario proving the mechanism works as intended: with perfect predictions, tuning correctly picks shrinkage=1.0 (no adjustment, since there's nothing to correct); with a persistently overestimated player across every tuning gameweek, tuning correctly picks shrinkage<1.0 (pulling the noisy star toward the median reduces the resulting regret).

## Real-data result: tune on 2021-22, test on the same 4 independent seasons used throughout this project

Tuning chose **shrinkage = 0.3** (a fairly aggressive pull toward the position median) on 2021-22's 8 gameweeks. Evaluated on 31 held-out gameweeks across 2020-21/2022-23/2023-24/2024-25, block-bootstrapped:

| | Mean realized | vs Track A |
|---|---|---|
| Track A (prediction-focused) | 55.45 | — |
| Track B (decision-focused) | 54.55 | −0.90, 95% CI [−2.45, +0.39] |
| Track C (hybrid) | 55.61 | +0.16, 95% CI [−0.52, +0.87] |

**Both CIs include zero — decision-focused shrinkage does not improve held-out decision regret, and if anything trends slightly negative.** Breaking this down further: of the 31 test gameweeks, shrinkage changed the actual squad/XI/captain decision in only 6 of them (25 were IDENTICAL to Track A) — 2 improved, 4 got worse. This pattern — a small, mostly-inert intervention that occasionally moves the needle in either direction, netting to something indistinguishable from zero — is consistent with the shrinkage value chosen on 8 tuning gameweeks having overfit to that small sample's idiosyncrasies rather than capturing a real, generalizable pattern. Given this project has repeatedly found that decision-level tests need many more than 8 observations to resolve real effects (Phase 4b's original 2-season inconclusive finding, Phase 7's lookahead-vs-myopic gap), an 8-gameweek TUNING set is an even smaller, noisier basis to select a hyperparameter from than the ~30-gameweek TEST sets this project uses to evaluate one.

## The more important finding: this specific implementation fails the spec's OWN joint bar, on calibration grounds alone

Spec Part XXVII explicitly requires decision-focused gains "without sacrificing calibration" — a genuine methodological caution about exactly the failure mode this run demonstrates. Checking the aggregate ratio of predicted EP to actual realized points (1.0 = unbiased) for each track's actually-selected starters, summed across all 31 test gameweeks:

| | Predicted EP / Actual |
|---|---|
| Track A | 0.930 (a modest, pre-existing, reasonable underestimate) |
| Track B | **0.375** — badly miscalibrated |
| Track C | 0.648 — also notably miscalibrated |

**Track B's tuning process was structurally blind to this damage.** Because squad membership within a position is a fixed quota (exactly 2 GK/5 DEF/5 MID/3 FWD, not flexible), and shrinkage is a monotonic affine transform applied uniformly within each position, it preserves relative ranking WITHIN a position perfectly — so most decisions (which players fill a position's quota) are literally untouched by shrinkage, exactly matching the 25/31 "identical decision" finding above. The tuning objective (total realized squad points) only ever sees the rare cases where shrinkage changes something (captaincy, starting-XI flex, and cross-position budget tradeoffs during squad selection) — it has no way to notice that the underlying EP values it's optimizing over have become nearly 3x too low in aggregate, because that damage is largely invisible to a metric that only cares about final squad composition, not the numbers used to get there. **This is exactly why the spec pairs decision regret with a calibration requirement rather than trusting decision regret alone** — a real, concrete demonstration of the risk, not a hypothetical one.

## Decision: NOT PROMOTED — a clean, informative negative result on both criteria

Track B fails BOTH of spec Part XXVII's joint criteria: it doesn't improve held-out decision regret (CI includes zero, point estimate slightly negative) AND it substantially sacrifices calibration (ratio 0.375 vs Track A's 0.930) — a decisive rejection, not an ambiguous one. Track C (hybrid) is a genuinely safer middle ground — its decision-regret effect is also not significant but centers much closer to zero (+0.16 vs Track A's baseline), and its calibration damage, while real, is roughly half of Track B's — but it still isn't a positive result on its own, just a less bad one. `select_squad`'s existing raw-EP pipeline (Track A) remains production. This is the kind of clean, honest negative result spec Part XXVI explicitly anticipates as a successful outcome of the research process, not a failure to "finish" the phase — and it surfaces a real, general lesson (decision-regret-only tuning can silently damage calibration) rather than just rejecting one specific hyperparameter choice.

## Concrete next steps (left for direction)

1. **More tuning gameweeks** — 8 is a small basis for selecting a hyperparameter; whether a shrinkage value tuned on, say, 30+ gameweeks generalizes any better than this run's is untested.
2. **A joint objective that penalizes calibration loss directly** during tuning (e.g., maximize realized points subject to a calibration-ratio constraint, or a weighted combination) — the natural fix for the exact failure mode this report identifies, not yet implemented.
3. **A genuine differentiable-optimization approach** (SPO+ or similar), if this simpler shrinkage-based proxy is judged worth escalating from — a substantially larger undertaking, explicitly out of scope for this pass.
