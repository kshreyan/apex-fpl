# One-time repo setup for the automated pipeline (Phase 13, Stage 6)

These are manual steps only you can perform (repo settings, not code).
Do them once, in order, after this stage's commit is pushed.

## 1. Make the repository public

Settings → General → Danger Zone → Change repository visibility → Public.

Two independent reasons this is required, not optional:
- GitHub Pages is free for public repos; private repos need a paid plan
  to publish Pages at all.
- Actions minutes are unlimited for public repos; private repos get a
  capped free tier. Either would silently violate the zero-dollar
  constraint this project is built on.

Checked before recommending this — properly this time. A first pass only
grepped the 123 files at HEAD, which checks the wrong thing: making a
repo public exposes every commit ever made, including anything added and
later removed. Re-checked with `gitleaks detect --no-git=false` across
full history (not just HEAD): 1 commit scanned, no leaks found. This
repo genuinely has only ever had one commit, so there's no deleted-file
history to worry about here — but re-run this check before any future
"make public" decision on a repo with real history; a HEAD-only grep
would have missed exactly this. Public is the one step in this checklist
that isn't really reversible: you can re-privatise the repo afterward,
but anything already cloned or indexed by then is gone. The
standings-extract design (Stage 1) already deliberately avoids storing
real manager names for the same reason, in anticipation of this.

## 2. Give the workflow permission to push

Settings → Actions → General → Workflow permissions → select
**"Read and write permissions"** → Save.

This is a real, necessary step, not a formality: `.github/workflows/
pipeline.yml` requests `contents: write` at the workflow level, but that
request is capped by this repo-level setting. Left on the default
("Read repository contents permission" only), the workflow's own
`git push` step will fail with a 403 every single run.

## 3. Enable GitHub Pages, sourced from Actions

Settings → Pages → Build and deployment → Source: **"GitHub Actions"**
(not "Deploy from a branch").

This is a correction, not the original design: a branch-based Pages
source only rebuilds on commits pushed by a *user*, not commits pushed
by a workflow using the default `GITHUB_TOKEN` — that's documented
GitHub behaviour, specifically to stop a workflow from recursively
triggering more Pages builds. Under the original branch-based setup,
the pipeline would have committed correctly, gone green every day, and
the published site would never have moved past whatever a human last
pushed by hand. Everything about that failure looks healthy except the
one thing that matters.

The fix restores the two permissions the first draft dropped
(`pages: write`, `id-token: write`, both scoped to the `deploy` job
only) and adds two steps already in `pipeline.yml`:
`actions/upload-pages-artifact` packages `/docs` after `build_site.py`
runs; `actions/deploy-pages`, in its own job with a `github-pages`
environment, publishes that artifact. Selecting this Pages source is
what creates that environment for you to grant the job access to — do
this step before the first workflow run, or the `deploy` job has
nothing to deploy to.

The site will be live at `https://<your-username>.github.io/apex-fpl/`
a minute or two after the first successful `deploy` job.

## 4. Secrets: none needed

The workflow authenticates `git push` and the Pages deployment using the
default, automatically-provisioned `GITHUB_TOKEN` (scoped by step 2,
above, and by the job-level `permissions:` blocks in `pipeline.yml`) —
no personal access token, no repository secret, nothing to create or
rotate. A PAT would also work for the push, but would reintroduce a
secret this design doesn't otherwise need; the Actions-native deploy
route was chosen specifically to avoid that trade. If a future stage
(e.g. the standings-capture job) ever needs real credentials, add them
under Settings → Secrets and variables → Actions at that point, not
before.

## 4b. The weekly healthcheck needs nothing extra

`.github/workflows/healthcheck.yml` (Stage 7) runs on its own weekly
schedule and uses the same default `GITHUB_TOKEN` already covered by
step 2 — no separate setup. See its own header comment, `pipeline/
healthcheck.py`'s module docstring, and `RUNBOOK.md` for what it checks
and how to act on a failure.

## 5. Verify

Actions tab → "FPL Pipeline" → **Run workflow** (this exercises the
`workflow_dispatch` trigger manually, without waiting for the daily
schedule). Confirm:
- the `pipeline` job goes green
- the `deploy` job goes green and reports a page URL
- `data/` and `docs/` show a new commit on `main` afterward (in practice
  this happens on every successful run, quiet day or not — see
  CLAUDE.md's determinism note on why `rebuilt_at_utc` always advances)
- the site at the Pages URL above actually shows the new
  `rebuilt_at_utc`, not a stale one — this is the one thing a green
  workflow can't prove by itself

A failed run should also open (or comment on) a GitHub issue titled "FPL
Pipeline run failed" automatically — worth confirming once, deliberately
breaking something trivial (e.g. a bad branch push) to see it happen,
then reverting.

The daily schedule (`06:23 UTC`, see the workflow file's own comment for
the reasoning) then runs unattended from here.
