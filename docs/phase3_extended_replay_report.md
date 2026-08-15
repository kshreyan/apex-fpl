# Phase 3 Extended Replay — Decision-Level Significance Across 4 Independent Seasons

**Run:** 2026-08-15. **Motivation:** `docs/phase4b_tournament_report.md`'s "Result 2" found the Phase 4b-promoted minutes/attacking champions produced a directionally positive but *not statistically significant* FPL-points improvement, pooled across only 2 seasons (63 gameweeks, 95% CI [-1.30, +5.78]). This extends that test to 4 genuinely independent seasons for a properly powered answer, rather than accepting "inconclusive" as final on a low-powered sample.

## Season selection — and why 2 available seasons were excluded, with reasons

Data exists for 6 seasons (2019-20 through 2024-25). Two are deliberately excluded from this significance test, each for a specific, stated reason — not because they were inconvenient:

- **2021-22 excluded:** this season's player-gameweek data was used to *tune* the promoted models' hyperparameters (minutes half-life, attacking alpha/lookback) in Phase 4b. Including it in a decision-level significance test would be circular — the champion configuration was partly selected using this season's data.
- **2019-20 excluded:** real bug found — this season's `merged_gw.csv` uses an older schema entirely missing the `position` and `team` columns the pipeline depends on (confirmed by inspection: the header has `name, assists, bonus, ...` with no `position`/`team` fields present at all, unlike every other fetched season). Reconstructing them would require joining through `element` IDs against `players_raw.csv`/`teams.csv`, in the same way the Vaastav collector itself does — real, buildable work, but out of scope for this pass. Documented here rather than silently dropped.

**The 4 remaining seasons — 2020-21, 2022-23, 2023-24, 2024-25 — are genuinely independent of any tuning decision** and are used for the primary test.

## Two more real bugs found and fixed while extending

1. **2019-20's gameweek numbering isn't contiguous.** Real gameweeks run 1-29, then jump to 39-47 — FPL never reused 30-38 after the COVID-19 restart (confirmed identically in both `fixtures.csv` and `merged_gw.csv`). `scripts/run_phase3_replay.py` previously hardcoded `range(START_GW, 39)`, which would have silently skipped 9 real post-restart gameweeks for this season. Fixed by moving gameweek discovery into a proper library function, `vaastav_loader.season_gameweeks()`, that reads the real event numbers from the data rather than assuming a fixed range — with a regression test (`tests/unit/test_season_gameweeks.py`) locking in both this season's actual gap and a cross-check against the already-known 2022-23 GW7 blank gameweek.
2. **2024-25 introduced a "Manager" pseudo-player mechanic from GW23 onward** — real Premier League managers (e.g. Fabian Hürzeler / Brighton) appear as selectable rows with `position="AM"`, scored via separate `mng_*` fields. This crashed `run_gameweek()` with `KeyError: 'AM'` three layers deep in position-lookup code that only knows GK/DEF/MID/FWD. This is the same `mng_*` mechanic first noticed (but not investigated) in Phase 0's live API capture. Fixed by filtering any row whose position isn't one of the four classic-squad positions out of the roster entirely — correct behavior, since Manager picks aren't part of the 15-player squad this optimizer builds, not a workaround. Regression test: `tests/unit/test_replay_manager_position_filter.py`.

Both fixes are in `src/apex_fpl/backtesting/replay.py` and `src/apex_fpl/backtesting/vaastav_loader.py`; the full test suite (69 tests before this session's additions) still passes.

## Result: the improvement is real and statistically significant

| Season | Model total | Baseline total | Mean diff/GW | 95% CI | W-L-T |
|---|---|---|---|---|---|
| 2020-21 | 1,644 | 1,458 | **+5.81** | **[+2.56, +9.25]** (significant alone) | 24-7-1 |
| 2022-23 | 1,780 | 1,718 | +2.00 | [-2.84, +6.48] | 18-11-2 |
| 2023-24 | 1,749 | 1,663 | +2.69 | [-2.41, +7.81] | 19-13-0 |
| 2024-25 | 1,900 | 1,838 | +1.94 | [-2.09, +6.19] | 16-14-2 |
| **Pooled (127 GWs)** | — | — | **+3.12** | **[+0.85, +5.27]** | **77-45-5** |

**The pooled 95% CI excludes zero.** By this project's own pre-registered promotion-style criterion (used consistently since Phase 4a), the Phase 4b-promoted models produce a **statistically significant improvement in realized FPL points**, not just component-level proper-scoring-rule metrics. A complementary one-sided sign test on the pooled win/loss record (77 wins, 45 losses, 5 ties) gives **p = 0.0024** — strongly corroborating, not just a coincidence of the bootstrap's resampling.

**Every one of the 4 seasons was directionally positive** (mean diff > 0, more wins than losses) even though only 2020-21 cleared significance on its own — exactly the pattern a real, moderate, consistent effect that individual-season sample sizes were too small to confirm reliably would produce, and exactly why Phase 4b's report declined to call the 2-season result significant rather than rounding up on a favorable-looking but inconclusive CI.

## Honest limitations

- Still only 4 seasons — a real number, but not enormous; a 5th and 6th (2019-20, 2021-22) exist in principle but are excluded for the stated methodological reasons above, not fetched-and-hidden.
- The 4 seasons span genuinely different conditions (2020-21: pandemic, closed stadiums, midweek fixture congestion; 2022-23: World Cup mid-season break; 2023-24/2024-25: normal seasons) — this is a point in favor of generalizability (the effect held across quite different conditions), but 4 seasons is not enough to make a strong claim about *which* conditions it's robust to.
- This remains a "best XV from scratch every gameweek" evaluation, not a real season-long squad with transfers/chips (Phase 7 scope) — the significant result says the underlying forecasting pipeline is doing real work, not that a manager using it end-to-end all season would realize exactly +3.12 points/GW once transfer costs, chip timing, and price changes enter the picture.
- Bonus points and defensive contributions remain entirely unmodeled (documented since Phase 2); the gap between projected and realized points still partly reflects this.

## Registry and documentation updates

`artifacts/model_registry.json`'s `decision_level_followup` fields updated on both `minutes_model` and `attacking_allocation_model` to point here with the significant result, superseding the "not yet confirmed" language from Phase 4b (which is left unedited as the honest record of what was known at that time — this is a new finding from new evidence, not a correction of an error).
