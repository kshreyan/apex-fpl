# RUNBOOK — diagnosing and recovering from a failure

For each failure mode: how you'll find out, how to confirm it, and what
to actually do. Cross-references `CLAUDE.md` where a fix touches a rule
documented there, so the reasoning behind a fix isn't only in this file.

## How you'll generally find out

- A failed `pipeline.yml` run opens or comments on a GitHub issue titled
  **"FPL Pipeline run failed"** (see `.github/workflows/pipeline.yml`'s
  `notify-on-failure` job).
- A missed deadline (predict.py exit code 3) opens or comments on
  **"Missed a gameweek prediction deadline"** — this is a *successful*
  (green) run, so it would otherwise be silent; see CLAUDE.md's note on
  why exit 3 gets its own issue rather than just a log annotation.
- A failed weekly healthcheck opens or comments on **"Healthcheck
  failed"** (`.github/workflows/healthcheck.yml`).
- The site itself: a client-side staleness banner (36h warn / 72h
  critical, comparing `data-rebuilt-at` against the browser's clock —
  see `assets/staleness.js`) or a permanent "no prediction recorded"
  notice on a gameweek page / the homepage history.

## `predict.py` step fails (red `pipeline` job, "FPL Pipeline run failed" issue)

1. Open the failed run's log, find the "Run predict.py" step. The
   exception is printed in full — this is the model itself erroring,
   not a partial/fallback prediction (see CLAUDE.md's fail-loud rule:
   nothing was written).
2. Common causes: the live FPL API changed shape under
   `pipeline/fpl_client.py`'s schema validators (look for a
   `SchemaValidationError` naming the exact dotted field), or a genuine
   model bug.
3. Reproduce locally: `python -m pipeline.predict --dry-run`. Fix, add a
   regression test, `python -m pytest -q`, then re-run the workflow via
   **Actions → FPL Pipeline → Run workflow** (`workflow_dispatch`) —
   don't wait for tomorrow's schedule if a deadline is close.
4. If the deadline passed before this got fixed: see "A gameweek shows
   as missing a prediction," below — this is now that case.

## `score.py` step fails

1. Same log-reading approach as above. Note: **this run's prediction
   commit already landed** before `score.py` even runs (see CLAUDE.md's
   two-commit-point note) — nothing time-sensitive was lost.
2. Common cause: the "actual points" staleness limitation documented in
   `pipeline/score.py`'s own docstring — if the pipeline was down across
   more than one full gameweek cycle, `bootstrap-static`'s
   `event_points` may no longer reflect the gameweek being scored.
3. Fix, test, re-run via `workflow_dispatch`. `score.py`'s normal path
   never re-scores an already-scored gameweek, so a re-run is safe to
   repeat.
4. If a result was scored *incorrectly* and already committed (not this
   failure mode, but related): use the correction path, not a manual
   edit — `python -m pipeline.score --correct <gw> --reason
   <bonus_revised|scoring_bug|data_correction|schema_migration>`. This
   appends a new result with `supersedes` set; the original line is
   never touched (CLAUDE.md's ledger rule).

## `build_site.py` step fails, or the `deploy` job fails

1. `build_site.py` failing usually means `data/calibration.json` is
   missing or malformed (it raises loudly rather than rendering a
   misleading empty site) — check whether the "Rebuild
   data/calibration.json" step above it actually succeeded.
2. `deploy` failing (not `pipeline`) with the artifact successfully
   built: check Settings → Pages is still sourced from **"GitHub
   Actions"**, not a branch (`SETUP.md` §3, `CLAUDE.md`'s Pages-source
   rule). The weekly healthcheck's `pages_source` check exists
   specifically to catch this before you'd notice it any other way —
   check whether it's been failing quietly for longer than one week.
3. Re-run via `workflow_dispatch` after fixing.

## A gameweek shows as missing a prediction (site shows a permanent gap)

This means `data/calibration.json`'s `coverage.gameweeks_missing_
prediction` includes it — a deadline passed with no prediction ever
recorded. This is rendered as a permanent, honest fact (CLAUDE.md), not
something to hide or backfill:

1. **Do not** write a late prediction and backdate it, and don't try to
   make the gap disappear by editing `calibration.json` by hand — it's
   a derived projection, rebuilt every run; a hand-edit would just be
   overwritten (or worse, mask a real problem) on the next run anyway.
2. Diagnose *why* the deadline was missed (workflow disabled? predict.py
   erroring silently-looking, i.e. exit 3 every day because the
   schedule runs too close to deadline? see "cron" below) and fix the
   root cause so it doesn't recur.
3. The gap stays visible for that gameweek, permanently. That's the
   design, not a bug to route around — see the closing note in
   `docs/methodology`'s prose and CLAUDE.md's backtests rule: the
   record's credibility depends on this never being quietly patched
   over.

## Healthcheck: `blame_cache_integrity` fails

A cached commit SHA in `data/site/commit_sha_cache.json` no longer
matches a fresh `git blame` for that line. Per
`pipeline/site/git_commits.py`'s docstring, this should never
legitimately happen — treat it as a real incident, not routine drift:

1. Check whether `data/site/commit_sha_cache.json` was hand-edited (`git
   log -p -- data/site/commit_sha_cache.json`) or whether repo history
   was rewritten (force-push, rebase of `main`) around the affected
   commit.
2. If the cache was corrupted (not history): the cached value is wrong;
   delete the specific bad entry (or the whole file — it fully
   regenerates from a fresh blame the next time `build_site.py` runs)
   and let the next run rebuild it.
3. If history was genuinely rewritten: this is a bigger problem than the
   cache — the "proof this preceded the deadline" claim for every commit
   after the rewrite point needs re-examination, not just the cache
   entry. Treat as a possible integrity incident on the whole record,
   not a one-line fix.

## Healthcheck: `raw_capture_hashes` or `raw_not_gitignored` fails

- `raw_not_gitignored`: someone (or some tool) re-added `data/raw/` to
  `.gitignore`. Revert it — see CLAUDE.md's data taxonomy for why this
  tier is committed on purpose (an unverifiable `sha256` is worse than
  no hash at all).
- `raw_capture_hashes` (missing file or mismatch): a raw capture
  referenced by a prediction's `data_sources` isn't on disk, or doesn't
  match its recorded hash. Check whether it was ever actually committed
  (`git log -- <path>`) — if it simply never landed (e.g. this failure
  predates the Stage 6 `.gitignore` fix), that prediction's evidence
  trail has a permanent gap; note it, don't fabricate a replacement.

## Healthcheck: `workflow_sha_pinning` fails

A `uses:` line in a workflow file was changed to a floating tag (e.g.
`@v4`) instead of a commit SHA — possibly during an update. Re-pin it:
`git ls-remote --tags https://github.com/<owner>/<repo>.git` to find the
real commit SHA for the tag you want, `uses: owner/repo@<sha> # vX.Y.Z`.
Never guess a SHA.

## Healthcheck: `workflow_recency` fails, or `pipeline.yml` just stops running with no failed-run issue at all

The worst case: nothing is failing, because nothing is running.

1. Actions tab → confirm `pipeline.yml` is listed and not shown as
   "disabled." GitHub auto-disables a scheduled workflow after **60 days
   with no commits to the repository** — check CLAUDE.md's note on why
   this is normally self-preventing (every successful run commits, even
   on a quiet day) — if it triggered anyway, something upstream has been
   silently failing for a long time already.
2. If disabled: Actions tab → the workflow → **"Enable workflow"**.
3. If enabled but not firing: check for a YAML syntax error introduced
   in a recent edit (`python -c "import yaml;
   yaml.safe_load(open('.github/workflows/pipeline.yml'))"` catches
   structural breaks; GitHub's own Actions tab surfaces parse errors
   directly on the workflow's page too).
4. Either way, trigger a manual run via `workflow_dispatch` once fixed,
   and don't wait for the next scheduled tick to confirm the fix worked.

## The standings-capture job (top-10k baseline)

Not yet built — flagged, not silently skipped, since Stage 6. If it's
still missing when this matters: `points_vs_baselines.top_10k_average`
and `cumulative_top_10k_points` will correctly show as `null` rather
than a wrong number (see `pipeline/score.py`'s own docstring on this
being an honest gap, not an error). No RUNBOOK action needed unless this
job has been built and then stops working, which isn't yet a real
scenario.
