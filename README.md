# APEX FPL

A research-grade Fantasy Premier League forecasting system, running
unattended, with every prediction and every outcome public.

**Live site:** https://kshreyan.github.io/apex-fpl/ — the running
calibration record (hits and misses alike), this week's picks, and the
methodology behind them.

## What this actually is

Two things, built in two phases:

1. **The model** (`src/`) — Monte Carlo simulation over modelled team
   goal expectations, per-player minutes and attacking-involvement
   shares, and a reduced-form bonus-points model, tournament-selected
   against simpler baselines on multi-season historical replay. See
   `docs/methodology/` on the live site for what it does and doesn't
   model, and `docs/phase*_report.md` for the research history that
   built it — those are pre-launch, hindsight backtests, clearly not the
   live record (see `docs/README.md`).
2. **The automation** (`pipeline/`) — wraps that model in a zero-cost,
   zero-manual-input pipeline: a daily GitHub Actions workflow decides
   whether to publish a prediction (never a hardcoded schedule), commits
   it to git *before* the deadline it's for, scores it automatically
   once the gameweek settles, and republishes a public static site with
   the full record — including every miss, with the same prominence as
   every hit. `CLAUDE.md` has the full reasoning behind how this is
   built; `SETUP.md` is the one-time manual setup; `RUNBOOK.md` is what
   to do when something breaks.

## Running it locally

```
pip install -e ".[dev]"
python -m pytest -q                 # full test suite
python -m pipeline.predict --dry-run    # exercise the pipeline without writing anything
python -m pipeline.healthcheck --no-gh  # local integrity checks (skips the two that need `gh`)
```

## Non-affiliation

APEX FPL is an independent research project. It is not affiliated with,
endorsed by, or connected to the Premier League or Fantasy Premier
League. Fantasy Premier League, Premier League, and associated marks are
trademarks of their respective owners.
