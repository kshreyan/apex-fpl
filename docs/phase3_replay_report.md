# Phase 3 Replay Report — 2022-23 Season, GW7-GW38

**Run:** 2026-08-14. **Reproduce:** `PYTHONPATH=src python scripts/run_phase3_replay.py`. Full per-gameweek artifacts (frozen recommendation + evaluation for all 31 replayed gameweeks) in `artifacts/phase3_replay/2022-23/`, aggregate in `season_summary.json`.

> **Superseded numbers, kept as historical record.** On 2026-08-15, Phase 4b promoted new minutes and attacking-allocation champions (`docs/phase4b_tournament_report.md`) and re-ran this exact replay — the `run_phase3_replay.py` script's *defaults* changed accordingly (it now targets whichever season is passed as an argument, and `replay.run_gameweek()`'s model defaults are the promoted Phase 4b champions, not the models described below). This document is left unedited as the frozen record of what the Phase 2/3-era models actually produced; see the Phase 4b report for the post-promotion re-run and honest significance testing of whether the improvement is confirmed at the season-decision level.

## Headline result

| | Total | Mean / GW |
|---|---|---|
| Model-driven squad | 1,667 | 53.8 |
| Recent-form baseline (no model, just last-6-GW actual points) | 1,667 | 53.8 |

**Exactly tied on total points across 31 real gameweeks.** Mean per-gameweek difference: **+0.00**, 95% bootstrap CI **[-5.07, +5.23]** — comfortably spans zero. Gameweek-by-gameweek: the model won 14, the baseline won 17, no ties in individual gameweeks (the season-long tie is a coincidence of the two totals, not a pattern of draws).

## Why this matters more than the Phase 2 result

`docs/phase2_milestone_report.md` reported a single gameweek (GW20) where the model beat the baseline by +31 points, and explicitly warned: *"n = 1 gameweek is not statistical evidence of anything... one gameweek beating one baseline could easily be variance."* This replay is the direct test of that warning, and **it was right to warn** — GW20's +31 was an outlier (visible in the per-GW log: differences range from -27 to +31 across the season), not a sign of forecasting skill. Averaged properly with a block bootstrap, there is **no detectable edge** for this baseline model over a naive recent-form heuristic in this season.

This is the correct and expected outcome for a Phase 2 baseline that spec Part X explicitly named as something to move past (proportional goal allocation), running team-strength constants ported unmodified from an entirely different sport competition (international football → club football, never refit), with no defensive contributions, no BPS/bonus simulation, and a minutes model that is literally just "how often did they play recently." The honest conclusion is **not** "the approach doesn't work" — it's "this specific baseline, as built, shows no measurable advantage yet, and the real research work (Phase 4 model tournaments, refitting team-model constants, moving past proportional allocation) hasn't started." Reporting this plainly is the point of doing Phase 3 at all — the spec's stopping-rule and anti-overfitting principles (Parts XXXIX, LX) exist precisely so a lucky single gameweek doesn't get mistaken for validated skill.

## A real bug this run caught: 2022-23 GW7 is a genuine blank gameweek

The first replay attempt crashed on GW7 with an opaque `max() iterable argument is empty` error from deep inside the Monte Carlo simulator. Investigation confirmed: **GW7 of the 2022-23 season genuinely has zero fixtures** in the data — the real-world Gameweek 7 (September 2022) was postponed following the death of Queen Elizabeth II and its fixtures were redistributed to later gameweeks. This is not a data error; it's a real blank gameweek the pipeline needs to handle explicitly (spec Part XXXIII references blank/double gameweeks by name as something the system must model, not silently mishandle).

Fixed: `run_gameweek()` now detects zero target-GW fixtures immediately and raises a specific `BlankGameweekError`, rather than failing obscurely three layers deep once the simulator discovers it has no players to simulate. The replay loop treats this as an expected, informative skip (reported separately from real failures), and a regression test (`tests/unit/test_replay_blank_gameweek.py`) locks in both the historical fact (GW7 has zero fixtures) and the correct exception type.

## What this replay does and doesn't establish

**Does establish:**
- The full pipeline runs unattended across an entire season's worth of real deadlines without crashing (after the GW7 fix), each gameweek's recommendation frozen strictly before its own results are used.
- A concrete, reproducible, statistically-honest baseline number (1,667 points / 53.8 per GW, tied with a naive heuristic) that any future model — refit team constants, a real minutes model, a non-proportional attacking model — must be measured against and beat with a bootstrap CI that excludes zero before being called an improvement, per the champion/challenger promotion criteria in spec Part LXI.
- The leakage-testing infrastructure (`tests/leakage/test_replay_no_future_leakage.py`) actually holds up under a full season of real use, not just a synthetic example.

**Does not establish:**
- Anything about calibration, distributional accuracy, or decision quality beyond total realized points — this replay used a single point-estimate comparison (total/mean points, bootstrap CI on the difference), not the proper scoring rules (log loss, CRPS, calibration) spec Parts XX-XXI require for a real model evaluation. That needs Phase 5.
- Anything about a different season, a different baseline, or a different target gameweek range — this is one season against one baseline.
- Whether any *specific* component (team model, minutes model, attacking allocation) is the weak link — that requires the ablation studies in spec Part XL, not yet built.

## Next steps this result motivates

1. **Phase 4 model tournament**, specifically: refit the team model's constants (`K_BASE`, `HALFLIFE_DAYS`, Dixon-Coles `rho`) on real Premier League data instead of carrying over the World Cup repo's international-football fit — the single most likely source of a free improvement, since those constants were never claimed to transfer.
2. An ablation (Part XL) isolating whether the team model, minutes model, or attacking allocation is responsible for the flat result — right now they're only evaluated bundled together.
3. Replaying additional seasons (2021-22, 2023-24, 2024-25 are all grade-A per the Vaastav audit) before drawing any conclusion about whether 2022-23's flat result generalizes.
