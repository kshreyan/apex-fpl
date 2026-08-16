# What's in this directory

This folder is two things that happen to share a path, not one:

## The live site

`index.html`, `current/`, `gameweek/`, `methodology/`, `assets/` are the
published site — https://kshreyan.github.io/apex-fpl/ — generated
entirely by `pipeline/build_site.py` from the append-only ledgers under
`data/predictions/` and `data/results/`, and fully rebuilt on every
automated pipeline run. Never hand-edit these; they're overwritten the
next time the pipeline runs. These are the project's **live, real
predictions**, each one committed to git before its gameweek deadline.

## Pre-launch research reports (everything else here)

`phase2_milestone_report.md` through `phase12_production_system_report.md`,
`robust_captaincy_report.md`, `data_sources.md`, `fpl_gap_analysis.md`,
`vaastav_archive_audit.md`, `world_cup_transfer_audit.md` — these are
**hindsight backtests and research notes from before this system ever
made a live prediction**, written while building and validating the
model on historical data. They describe how the model performed when
replayed against seasons that had already finished, with the benefit of
knowing the outcome in advance. That's a normal, necessary part of
building a forecasting model — and a fundamentally different kind of
claim than a live prediction made before its deadline.

**These are never combined with, or presented near, the live calibration
record** — that's a standing rule (see `CLAUDE.md`), not an oversight.
If you're trying to judge whether this model is actually any good, the
live site above is the only place that answers that question honestly;
these files are how it got there.
