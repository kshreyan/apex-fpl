# Vaastav `Fantasy-Premier-League` Archive — Point-in-Time Correctness Audit

**Audited:** 2026-08-14. **Method:** direct inspection of the live repository (README, `DATA_DICTIONARY.md`, `LICENSE`, `collector.py`), plus empirical verification against the GitHub REST API (commit history for specific data files) and the raw CSV data itself — not inference from the README's claims alone.

**Repository:** [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League), MIT-licensed code (confirmed by reading `LICENSE` directly — GitHub's UI reports the ambiguous "Other" license because the file appends a data-ownership disclaimer after the MIT text, not because the license itself is non-standard). The disclaimer states plainly: "The data provided is property of http://fantasy.premierleague.com and https://understat.com — I don't own any of the data." Safe to use for non-commercial research with attribution.

## Verdict

**Conditionally trustworthy for historical backtesting, with one confirmed exclusion and one confidence downgrade for the most recent seasons.** Not a single monolithic yes/no — the archive's different columns have different point-in-time guarantees, and treating them uniformly would be a mistake.

## How the data actually gets built (not just what the README claims)

Read `collector.py` directly: per-gameweek files (`data/<season>/gws/gwN.csv`) are assembled from each player's own history file, which is populated from FPL's **`element-summary/{id}/` `history` array** — FPL's own immutable, per-gameweek ledger (one row per completed gameweek, containing that gameweek's goals/minutes/bps/**price**/**ownership**/transfers as FPL itself recorded them at the time). This matters enormously: it means most columns are **not** vaastav re-deriving "current state" and mislabeling it as historical — they're FPL's own frozen record, which by construction can't drift just because vaastav's scraper ran late.

**Empirical confirmation, not just code-reading:** compared `value` (price) between `data/2022-23/gws/gw1.csv` and `gw38.csv` for the same season. **461 of 572 common players show a different price** between the two files (e.g. Bednarek 45→41, Stones 55→56, Garnacho 45→41). This is exactly the pattern real in-season price drift produces, and is strong evidence `value` is a genuine per-gameweek historical value, not a static season figure copy-pasted across files.

## Confirmed leakage risk: the `xP` column

The maintainer's own `README.md` and `DATA_DICTIONARY.md` document this directly — I did not need to discover it myself, but I verified the claim is actually present and specific rather than a vague caveat:

> `xP` is scraped from FPL's `ep_this` field *after* each gameweek has ended... Empirical comparisons suggest the scraped `xP` values diverge from live pre-match `ep_this`: live API `ep_this` vs `form` correlation ≈0.98, scraped `xP` vs `form` correlation ≈0.75, `xP` rolling-3 vs same-GW `total_points` correlation ≈0.40 (unusually high for a genuinely pre-match feature).

Confirmed in code: `get_expected_points()` in `collector.py` reads `xP` from a **separate** file (`xP{gw}.csv`), not from the `history` ledger — this is the one column in the dataset that is genuinely a live/current-state scrape rather than FPL's frozen per-gameweek record, and it is the one column proven to leak.

**Rule for APEX FPL: `xP`/`ep_this`/`ep_next` from this archive must never be used as a gameweek feature.** Either drop the column entirely or shift it `+1` gameweek per-player as the maintainer suggests — and even then, treat it as low-confidence given the maintainer's own correlation evidence.

## Cadence regime change — empirically verified via commit history, not just the README's notice

The README states weekly updates "stopped at the end of the 2024-25 season," replaced by three bulk updates/season (start, end of January window, end of season). I verified this against actual commit timestamps via the GitHub API rather than trusting the notice at face value:

| Season | File | Commit dates found | Interpretation |
|---|---|---|---|
| 2022-23 | `gws/gw1.csv` | 2022-08-08, 2022-11-23 | Captured 2 days after GW1 (season started 2022-08-06), later corrected. Genuine near-real-time capture. |
| 2022-23 | `gws/gw38.csv` | 2023-05-29 | Captured right after the season's final gameweek. Progressive, season-long capture confirmed. |
| 2025-26 | `gws/gw1.csv` | 2025-08-20, **2026-02-05** | Initially captured close to real-time (4 days after kickoff), but **rewritten 5.5 months later** in the "end of January window" bulk update. |
| 2025-26 | `gws/gw20.csv` | **2026-02-05 only** | Never captured close to when GW20 actually happened (~mid-December 2025) — first appears 6-8 weeks late, only in the bulk update. |
| 2025-26 | `gws/gw38.csv` | **2026-06-17 only** | Only captured once, in the end-of-season bulk update. |

This is a real, structural change, not a false alarm: for the 2025-26 season (and by extension however 2026-27 gets handled, since `data/2026-27/` already exists in the repo with pre-season scaffolding but no `gws/` directory yet), most gameweeks were **never captured shortly after they happened** — only in a delayed bulk sweep.

**Why this mostly doesn't corrupt the ledger-sourced columns:** because `value`/`selected`/`goals_scored`/`minutes`/`bps`/etc. come from FPL's own frozen `history` array (see above), a late scrape still pulls the correct historical row for that gameweek — FPL's API doesn't retroactively rewrite a past gameweek's recorded price or stats when you query it later. This is a reasoned inference from reading the collection code plus documented FPL API behavior, **not an independently verified byte-level guarantee** — I had no second ground-truth source to diff against in this session. Treat 2025-26-onward ledger columns as **B (reusable, moderate confidence)** rather than **A (fully validated)** until a spot-check against another source is done.

**Why this makes the `xP` risk strictly worse for 2025-26-onward:** an `ep_this` value scraped 6+ months after a gameweek has no plausible relationship to what was knowable before that gameweek's deadline — the already-documented leakage risk is not just "uncertain timing" for these seasons, it's essentially guaranteed contamination if used unshifted.

## Files that must never be treated as gameweek-dated

`cleaned_players.csv` (explicitly "the overview stats for the **season**" per the README) and `players_raw.csv` (a snapshot of `bootstrap-static`'s current `elements`, structurally identical to what our own `players.csv`/`player_stats.csv` Silver tables capture) are season-aggregate or current-snapshot files, not per-gameweek historical records. Only `gws/gwN.csv` and `merged_gw.csv` carry genuine per-gameweek dating.

## Net usage policy for APEX FPL

1. **Usable now** for Dataset A/B historical training: `gws/gwN.csv` (or `merged_gw.csv`) match-event, price, and ownership columns for seasons through 2024-25 — genuine weekly capture, corroborated by commit history and the GW1-vs-GW38 price-divergence check.
2. **Usable with a confidence downgrade** for 2025-26 (and presumptively 2026-27's eventual archive once it exists): same columns, same reasoning (FPL's own frozen ledger), but not independently spot-checked against a second source this session — flag as B-grade evidence in any model card that trains on it.
3. **Never use** the `xP` column from this archive, at any season, unshifted. Either drop it or shift `+1` per player — and even shifted, don't trust it without independent recalculation from our own captured `ep_this`/`ep_next` snapshots going forward.
4. **Never use** `cleaned_players.csv`/`players_raw.csv` as if they were gameweek-dated.
5. Do not rely on this archive for **current-season (2026/27) live data** — its 3x/season bulk cadence means it will always lag our own daily Bronze capture by weeks to months during the season. It is a historical-backtesting source only, not an operational one.
