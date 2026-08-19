# CLAUDE.md

Project rules for APEX FPL's automation and public site (Phase 13) —
*why* things are built the way they are, for whoever (human or agent)
touches this next. `README.md` is the project's front door; `SETUP.md`
is the one-time manual repo setup; `RUNBOOK.md` is what to do when
something breaks. This file is none of those — it's the accumulated
reasoning behind decisions that aren't obvious from reading the code,
each one written down at the point it was actually made or corrected,
not reconstructed after the fact.

## System overview

One workflow, `pipeline.yml`, runs daily (`06:23 UTC`, the anchor tick)
and hourly (`:37`, a cheap-gated tick that only does anything inside the
24h closing window before a deadline — see `pipeline/should_run_now.py`
and Block 1.6), plus on manual dispatch. It always checks in; whether it
*acts* is decided by `pipeline/predict.py`'s own phase/deadline logic
(`pipeline/gw_state.py`), never by the cron schedule itself. In order:
`predict.py` (writes at most one new line to
`data/predictions/gw{n}.jsonl`, commits and pushes immediately) →
`predict_transfers.py` (Block 2.5 — writes at most one new line to
`data/transfer_recommendations/gw{n}.jsonl`; structurally separate from
`predict.py`, wired as a genuinely best-effort step that logs a warning
and never fails the job, since it depends on the real FPL entry's own
squad state existing, which `predict.py`'s core guarantee does not) →
`score.py` (scores any settled gameweek with no result yet) →
`metrics.py` (fully rebuilds `data/calibration.json` from the ledgers)
→ `build_site.py` (fully rebuilds `/docs`) → a second commit and push
(`data`, `docs`) → deploy to GitHub Pages via `actions/deploy-pages`. A
separate weekly workflow, `healthcheck.yml`, runs
`pipeline/healthcheck.py` to catch the failure modes a green daily run
can't see itself failing (see that module's own docstring). Both open
or update a GitHub issue on failure; `RUNBOOK.md` covers what to do
about each one.

**Not yet built:** the standings-capture job (`leagues-classic/314`,
paginated, extract-only per the Stage 1 privacy decision below) that
would populate the top-10k baseline. Flagged repeatedly since Stage 6
rather than silently skipped — `score.py` and `metrics.py` both already
handle its absence correctly (`top_10k_average` reports `null`, not a
wrong number), so nothing is broken by its absence, but it's real,
not-yet-started work, not a design gap.

## The real FPL entry: execution is human-in-the-loop, and failure is not reconciled away

A real FPL entry plays this model's picks, starting GW1. Its rank, picks,
chips, and transfers are third-party-verifiable via the public FPL API —
a stronger claim than anything this project asserts about itself.

Two standing rules, decided before GW1, not after:

- **Execution is manual by design and stays that way.** The model
  computes a recommendation; a human enters it in the real account
  before each deadline. Automating the actual team-setting would require
  storing FPL session credentials as a secret this project has
  deliberately gone out of its way not to need anywhere else in its
  design — that trade was rejected on purpose, not overlooked. Do not
  wire real-account execution into `pipeline.yml` or any other automated
  path.
- **A missed deadline is a permanent, disclosed fact, never a quiet
  fix.** If the real entry's actual picks ever diverge from what the
  model recommended because the manual entry step didn't happen in time,
  that is recorded as a manual execution failure — not corrected,
  not backdated, not silently reconciled to match the model's ledger.
  It's shown on the site and the entry's real rank carries the
  consequence permanently, the same way a missing prediction already
  does (see the data taxonomy below: `coverage.gameweeks_missing_
  prediction`'s rendering is the existing precedent this follows). This
  rule was written down before the squad-state reader existed,
  specifically so it couldn't be quietly softened once that piece
  landed. **Status as of the multi-gameweek/chip/divergence build:**
  `apex_fpl.serving.entry_state` reads the real entry's actual squad
  each settled gameweek (public API, no credentials) and feeds it into
  the transfer recommendation (`pipeline/predict_transfers.py`).
  `pipeline/check_execution_divergence.py` now closes the other half:
  once a gameweek settles, it compares the real entry's actual picks
  (squad as a set of 15 IDs + captain ID) against what `predict.py`
  published for that same gameweek, checked exactly once per gameweek
  and never re-checked (`data/execution_divergence/gw{n:02d}.jsonl`),
  and `pipeline/build_site.py::build_gameweek_page` renders the result
  — a quiet "Execution matched" note, or a permanent critical notice on
  divergence. Wired into `pipeline.yml` as a best-effort step (same
  never-blocks-the-pipeline discipline as `predict_transfers.py` and
  `predict_chips.py`), alongside `score.py` since both act only on
  already-settled gameweeks.

## Data taxonomy

Everything under `data/` is exactly one of three kinds. Knowing which
kind a file is tells you whether it's safe to touch:

- **Source-of-truth ledgers** — `data/predictions/gw{n:02d}.jsonl`,
  `data/results/gw{n:02d}.jsonl`. Append-only. A line is never edited,
  reordered, or deleted, ever, by any script, including this one. A
  correction is a new line with a `supersedes` field pointing at the
  record it replaces (results additionally require a `supersede_reason`
  from a fixed enum). git history over these files is the actual
  integrity mechanism the whole project rests on — treat it as such.
- **Derived projections** — `data/calibration.json`, everything under
  `/docs/` that `pipeline/build_site.py` writes. Fully rebuilt from the
  ledgers on every run. Always safe to delete and regenerate; never
  hand-edited.
- **Immutable raw captures** — `data/raw/gw{n}/*.json`. Written once, on
  a successful fetch. Absent on failure (nothing partial is ever
  written). No supersede semantics — a bad capture is just missing, not
  corrected in place.
- **A fourth, narrower kind: committed model inputs.**
  `data/external/vaastav/2024-25/{fixtures.csv,teams.csv}` — carved out
  of the otherwise-gitignored `data/external/` (Part A's historical
  archive, real dev-machine-local data, everywhere else genuinely
  test-only). Found the hard way, from the first live workflow run:
  `apex_fpl.serving.live_data.build_team_model_fixtures()` reads these
  at actual prediction time, not just in backtests — it's the fallback
  team-strength prior used until a new season has enough of its own
  completed fixtures. CI has no access to anything not committed, so
  this stopped being a test fixture and became a real runtime
  dependency the moment the pipeline had to run somewhere other than a
  dev machine. Kept deliberately narrow (~753KB: just those two files,
  for just the one season `TEAM_MODEL_FALLBACK_SEASONS` actually names)
  — not `merged_gw.csv` (5MB, backtesting-only), not any other season.
  If `TEAM_MODEL_FALLBACK_SEASONS` in `scripts/run_production_
  recommendation.py` ever changes, this `.gitignore` carve-out needs to
  change with it, or the same crash recurs.

## `/current/` must never show picks without the season record attached

This week's live picks and the season's track record are never shown as
two separable pages where a visitor could see one without the other in
the same view. `pipeline/build_site.py::build_current_page` renders the
record section (or its honest empty state, "no gameweeks scored yet")
unconditionally, before the picks section, on the same page — not as a
nav link elsewhere. The homepage carries the full record already, per
its own design ("the calibration record is the homepage, not a buried
page"); `/current/` is a secondary page in navigation, which is an
information-architecture decision about what leads, not a rule that
picks are ever gated behind or hidden pending a record existing.

## Determinism means "same ledgers + calibration.json + git state"

The original site requirement was "same input data → byte-identical
HTML." That's accurate but incomplete: a gameweek page also embeds a
link to the git commit that recorded its prediction, resolved via `git
blame` (see `pipeline/site/git_commits.py`). Blame output legitimately
changes the moment a previously-uncommitted line's commit lands — the
record just became independently verifiable, which is real information,
not noise. A healthcheck (Stage 7) comparing two builds must account for
git state changing between them, not just treat any output diff as a
regression.

Client-side staleness detection (the "pipeline may have stopped running"
banner) is deliberately *not* part of this determinism claim — it reads
the browser's clock at load time via `assets/staleness.js`, not
anything computed at build time, specifically because a build-time check
can never detect the failure mode that matters most: the scheduled
workflow silently not running at all.

A gameweek whose deadline passed with no prediction ever recorded is a
permanent fact, not a transient staleness condition — it's rendered at
build time, straight from `calibration.json`'s
`coverage.gameweeks_missing_prediction`, on that gameweek's own page and
in the homepage history, and stays visible for as long as it's true.

## Every successful run commits — even one that changed nothing real

`pipeline.yml`'s commit step is not "commit if there's something to
commit" in the sense of "skip quiet days." `data/calibration.json`'s
`rebuilt_at_utc` advances on every run that reaches `pipeline/metrics.py`
successfully, which means `git diff --cached --quiet` is essentially
never true in practice — a genuinely empty commit (nothing at all
changed, including that timestamp) is the theoretical edge case the
guard exists for, not the expected path. This was a deliberate choice,
not an oversight the guard failed to catch:

- The client-side staleness banner (`assets/staleness.js`) compares
  `rebuilt_at_utc` against the browser's clock. If quiet runs didn't
  commit, that timestamp would stop advancing on a *healthy* pipeline
  the moment a week went by with nothing new to predict or score,
  producing a false staleness warning on a system that's working fine.
- It incidentally keeps GitHub's 60-day scheduled-workflow
  auto-disable timer permanently reset, since a real commit lands on
  `main` on every successful run. Free, not the reason this exists, but
  worth knowing it covers that failure mode too (see Stage 7's
  healthcheck item on scheduled-workflow health).

## The prediction commit is separate from, and precedes, the scoring/site commit

`pipeline.yml` pushes twice per run, not once: immediately after
`predict.py` succeeds (`data/predictions`, `data/raw`), then again after
`score.py` / `metrics.py` / `build_site.py` all finish (`data`, `docs`).
Found in Stage 7 review: with a single end-of-run commit, an exception
raised while scoring some unrelated, already-settled gameweek would
abort the job before this run's freshly-computed prediction — for a
DIFFERENT, still-upcoming gameweek — ever reached origin. A scoring bug
three gameweeks away from the live one could silently cost that live
gameweek's deadline. Splitting the commit means the highest-stakes
artifact this project produces is safe on `main` the moment it exists,
independent of what happens to the rest of the run. If more commit
points are ever added, keep this ordering: whatever is closest to a
deadline commits first.

## GitHub Pages source is "GitHub Actions," not a branch

`pipeline.yml` deploys via `actions/upload-pages-artifact` +
`actions/deploy-pages`, in a job separate from the one that commits.
This was a real, caught-in-review mistake, not a style choice: commits
pushed by a workflow using the default `GITHUB_TOKEN` do not trigger a
branch-based Pages build (documented GitHub behaviour, to prevent a
workflow from recursively triggering itself). Under a "Deploy from a
branch" Pages source, this pipeline would commit correctly, go green
every day, and the published site would simply never move — a failure
mode nothing in the pipeline itself would ever surface, since every
step it can see would report success. If Pages settings are ever
touched again, the source must stay "GitHub Actions" (see `SETUP.md`
§3) or this regresses silently.

## HTML escaping is structural, not a per-call habit

Every dynamic value that reaches a page in `pipeline/build_site.py` goes
through `pipeline.site.htmlgen.esc()` (directly, or via `render()`,
which escapes every keyword argument automatically). The only way to put
unescaped markup on a page is `pipeline.site.htmlgen.raw()`, which should
only ever wrap HTML this codebase assembled itself from other
`esc()`/`render()` calls — never a value that came from the FPL API or a
ledger record. `grep -rn "raw(" pipeline/site pipeline/build_site.py`
should find every deliberate exception; if a new one doesn't look like
"HTML we built ourselves," it's wrong.

## Backfilled backtests are never shown beside the live record

Any backtested or historical-replay result referenced in this project's
research reports (`docs/phase*_report.md`) is hindsight simulation. It
is never combined with, or presented near, the live calibration record
on the public site. The site's record is exclusively real predictions,
committed before their deadlines.
