# Data Sources

Machine-readable registry: `configs/sources.yaml`. This document is the human-readable summary — see the YAML for full field-by-field detail per source.

## Wired up and capturing (as of 2026-08-14)

- **`fpl_bootstrap_static`** — the FPL API's players/clubs/positions/gameweeks/scoring-config endpoint. Free, unauthenticated, live. No historical point-in-time mode exists, which is why `src/apex_fpl/data/bronze.py` snapshots it ourselves daily via a `launchd` job (`com.apexfpl.snapshot`, 08:00 local time). First snapshot captured 2026-08-14, seven days before the GW1 deadline (2026-08-21T17:30:00Z).
- **`fpl_fixtures`** — the FPL API's fixture list (kickoff times, FDR difficulty, scores once played). Same capture mechanism, same cadence.

Both sources feed `src/apex_fpl/entities/silver.py`, which builds append-only canonical tables (`data/canonical/*.csv`) idempotently keyed by payload hash, so re-running the build after new captures land never duplicates rows.

## Evaluated and explicitly out of scope

- **`martj42_international_results`** — the historical dataset the *World Cup* predictor project uses. International football only; structurally inapplicable to club-level Premier League FPL. Not used here.

## Flagged, not yet resolved

- **`vaastav_fantasy_premier_league_archive`** — the standard community historical-FPL dataset, and likely the only realistic path to pre-2026/27 training/backtesting data. **Not yet audited for point-in-time correctness.** Using it before that audit risks the exact leakage failure mode Part V exists to prevent (an archive that stores final-season aggregates rather than true per-gameweek pre-deadline snapshots would silently contaminate every historical backtest built on it).
- **Event-level match data** (tackles/blocks/interceptions/recoveries/shot locations) — required for defensive contributions (Part XII), goalkeeper saves (Part XIII), and BPS reconstruction (Part XIV). Neither `bootstrap_static` nor `fixtures` provides this. This is the single largest open data gap blocking a meaningful fraction of the spec's Layer-A player models. Needs evaluation of licensed feeds (Opta/StatsBomb) versus free proxies (understat, FBref) with an explicit ToS review before any adapter is built.
- **Betting odds feed** — needed to activate the market-consensus model (de-vig + logit-consensus math already exists, ported from the World Cup repo's `market.py`, and just needs a real feed wired in). No feed sourced yet.

## Principle governing this registry

A source only moves from "flagged" to "wired up" after its licence/ToS status and point-in-time reliability have actually been checked — not because it appears convenient or is commonly used by other public FPL projects. `configs/sources.yaml`'s `status` field is the single source of truth for what's actually active.
