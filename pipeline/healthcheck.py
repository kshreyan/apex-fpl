#!/usr/bin/env python3
"""System health checks (Phase 13, Stage 7).

A green `pipeline.yml` run only proves that the steps it can see
succeeded. It cannot tell you GitHub Pages quietly stopped rebuilding,
that a persisted commit-SHA cache has drifted from what `git blame`
would say today, or that the daily schedule itself stopped firing --
these are exactly the failure modes that look healthy from inside a
single run (see CLAUDE.md's Pages-source note for the real example that
motivated this module). Each check here is independent, read-only (no
check ever writes to data/, matching pipeline/score.py's own
discipline), and reports pass/fail with a reason.

Run manually: `python -m pipeline.healthcheck`. Also run on a schedule
by .github/workflows/healthcheck.yml, separately from the daily pipeline
-- deliberately separate, so a bug in the pipeline workflow doesn't take
the thing checking it down at the same time. That said: if GitHub ever
disables ALL scheduled workflows for this repo at once, both go dark
together -- this catches pipeline.yml specifically breaking, or Pages/
repo settings drifting, not "GitHub stopped scheduling anything here,"
which is a real gap, written down rather than silently assumed away.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.site import git_commits

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
RESULTS_DIR = REPO_ROOT / "data" / "results"
CALIBRATION_PATH = REPO_ROOT / "data" / "calibration.json"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

WORKFLOW_RECENCY_MAX_HOURS = 48.0
_USES_RE = re.compile(r"uses:\s*([\w.\-/]+)@([\w.\-]+)")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _read_ledger_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _content_hash(record_without_id: dict) -> str:
    return hashlib.sha256(json.dumps(record_without_id, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------- checks -----

def check_blame_cache_integrity() -> CheckResult:
    """The explicit Stage-5 deferral: once a record's commit SHA is
    cached, it's supposed to be authoritative forever, even if a future
    ledger reformat would make a fresh `git blame` disagree. This re-runs
    blame from scratch (bypassing the cache) for everything the cache
    already claims to know, and fails loudly on any disagreement --
    catching either a corrupted/hand-edited cache or history that got
    rewritten underneath it."""
    mismatches: list[dict] = []
    for ledger_dir in (PREDICTIONS_DIR, RESULTS_DIR):
        if not ledger_dir.exists():
            continue
        for ledger_path in sorted(ledger_dir.glob("gw*.jsonl")):
            records = _read_ledger_lines(ledger_path)
            mismatches += git_commits.verify_cache_matches_fresh_blame(ledger_path, records)
    if mismatches:
        detail = "; ".join(
            f"{m['record_id'][:12]} in {Path(m['path']).name}: cached={str(m['cached_sha'])[:12]} fresh={str(m['fresh_sha'])[:12]}"
            for m in mismatches
        )
        return CheckResult("blame_cache_integrity", False, f"{len(mismatches)} cached commit SHA(s) disagree with a fresh git blame: {detail}")
    return CheckResult("blame_cache_integrity", True, "all cached commit SHAs agree with a fresh git blame")


def check_pages_source_is_actions() -> CheckResult:
    """The exact failure class caught in Stage 6 review: a branch-based
    Pages source never rebuilds from a workflow's own GITHUB_TOKEN
    commits, so the pipeline can run green forever while the published
    site is frozen. Nothing else in this project would ever notice if
    this setting got reverted by hand."""
    try:
        owner_repo = git_commits._origin_owner_repo()
    except Exception as e:
        return CheckResult("pages_source", False, f"could not determine origin owner/repo: {e}")
    result = subprocess.run(["gh", "api", f"repos/{owner_repo}/pages"], capture_output=True, text=True)
    if result.returncode != 0:
        return CheckResult("pages_source", False, f"gh api repos/{owner_repo}/pages failed (Pages not enabled yet, or gh not authenticated): {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return CheckResult("pages_source", False, f"gh api returned invalid JSON: {e}")
    build_type = payload.get("build_type")
    if build_type != "workflow":
        return CheckResult(
            "pages_source", False,
            f"Pages build_type is {build_type!r}, expected 'workflow' -- see CLAUDE.md's Pages-source rule and SETUP.md §3. "
            "Commits from this pipeline will NOT trigger a redeploy while this is wrong.",
        )
    return CheckResult("pages_source", True, "Pages is sourced from GitHub Actions, as required")


def check_workflow_actions_are_sha_pinned() -> CheckResult:
    """Regression-guard on the Stage 6 SHA-pinning fix: a floating tag
    (@v4) can be repointed by the action's maintainer to different code
    without this repo's history showing any change at all."""
    unpinned = []
    if not WORKFLOWS_DIR.exists():
        return CheckResult("workflow_sha_pinning", False, f"{WORKFLOWS_DIR} does not exist")
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = wf_path.read_text()
        for match in _USES_RE.finditer(text):
            action, ref = match.groups()
            if not re.fullmatch(r"[0-9a-f]{40}", ref):
                unpinned.append(f"{wf_path.name}: {action}@{ref}")
    if unpinned:
        return CheckResult("workflow_sha_pinning", False, f"{len(unpinned)} action(s) not pinned to a commit SHA: {'; '.join(unpinned)}")
    return CheckResult("workflow_sha_pinning", True, "every third-party action reference is SHA-pinned")


def check_raw_captures_not_gitignored() -> CheckResult:
    """Regression-guard on the Stage 6 .gitignore fix: data/raw/ getting
    re-ignored would silently make every future prediction's
    data_sources[].sha256 point at evidence nobody can ever check again,
    without any test or the pipeline itself ever noticing."""
    probe = "data/raw/gw01/bootstrap_static/probe.json"
    result = subprocess.run(["git", "check-ignore", "-q", probe], cwd=REPO_ROOT)
    if result.returncode == 0:
        return CheckResult("raw_not_gitignored", False, f"{probe} would be gitignored -- data/raw/ has regressed back into .gitignore; see CLAUDE.md's data taxonomy")
    if result.returncode == 1:
        return CheckResult("raw_not_gitignored", True, "data/raw/ is not gitignored")
    return CheckResult("raw_not_gitignored", False, f"git check-ignore exited {result.returncode} (expected 0 or 1)")


def check_raw_capture_hashes() -> CheckResult:
    """Now that data/raw/ is actually committed (Stage 6), the
    sha256 every prediction records against its raw evidence is a real,
    checkable claim for the first time -- so check it."""
    missing, mismatches = [], []
    if not PREDICTIONS_DIR.exists():
        return CheckResult("raw_capture_hashes", True, "no predictions yet")
    for ledger_path in sorted(PREDICTIONS_DIR.glob("gw*.jsonl")):
        lines = _read_ledger_lines(ledger_path)
        if not lines:
            continue
        for source in lines[-1].get("data_sources", []) or []:
            raw_path = REPO_ROOT / source["raw_cache_path"]
            if not raw_path.exists():
                missing.append(source["raw_cache_path"])
                continue
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source["sha256"]:
                mismatches.append(source["raw_cache_path"])
    if missing or mismatches:
        parts = []
        if missing:
            parts.append(f"missing: {missing}")
        if mismatches:
            parts.append(f"sha256 mismatch: {mismatches}")
        return CheckResult("raw_capture_hashes", False, "; ".join(parts))
    return CheckResult("raw_capture_hashes", True, "every current prediction's referenced raw capture exists and matches its recorded sha256")


def check_ledger_integrity() -> CheckResult:
    """Structural integrity of the append-only ledgers themselves: valid
    JSON per line, record_id matches its own recomputed content hash
    (catches tampering or corruption), supersedes chains resolve to the
    immediately prior line's record_id, and no record_id repeats."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    for ledger_dir in (PREDICTIONS_DIR, RESULTS_DIR):
        if not ledger_dir.exists():
            continue
        for ledger_path in sorted(ledger_dir.glob("gw*.jsonl")):
            prior_id = None
            for i, line in enumerate(ledger_path.read_text().splitlines()):
                if not line.strip():
                    continue
                loc = f"{ledger_path.name}:{i + 1}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"{loc}: invalid JSON ({e})")
                    continue
                record_id = record.get("record_id")
                if not record_id:
                    problems.append(f"{loc}: missing record_id")
                    continue
                if record_id in seen_ids:
                    problems.append(f"{loc}: duplicate record_id {record_id[:12]}")
                seen_ids.add(record_id)
                without_id = {k: v for k, v in record.items() if k != "record_id"}
                if _content_hash(without_id) != record_id:
                    problems.append(f"{loc}: record_id does not match its recomputed content hash")
                supersedes = record.get("supersedes")
                if supersedes is not None and supersedes != prior_id:
                    problems.append(f"{loc}: supersedes={str(supersedes)[:12]} does not match the prior line's record_id ({str(prior_id)[:12]})")
                prior_id = record_id
    if problems:
        shown = "; ".join(problems[:10]) + (" ..." if len(problems) > 10 else "")
        return CheckResult("ledger_integrity", False, f"{len(problems)} problem(s): {shown}")
    return CheckResult("ledger_integrity", True, "all ledger lines are valid JSON, content-hash-consistent, and correctly chained")


def check_calibration_json() -> CheckResult:
    if not CALIBRATION_PATH.exists():
        return CheckResult("calibration_json", False, f"{CALIBRATION_PATH} does not exist")
    try:
        cal = json.loads(CALIBRATION_PATH.read_text())
    except json.JSONDecodeError as e:
        return CheckResult("calibration_json", False, f"invalid JSON: {e}")
    source_commit = cal.get("source_commit")
    if not source_commit:
        return CheckResult("calibration_json", False, "missing source_commit")
    result = subprocess.run(["git", "cat-file", "-e", source_commit], cwd=REPO_ROOT, capture_output=True)
    if result.returncode != 0:
        return CheckResult("calibration_json", False, f"source_commit {source_commit!r} is not a commit reachable in this repo's history")
    return CheckResult("calibration_json", True, "calibration.json is valid JSON with a real, reachable source_commit")


def check_scheduled_workflow_recency(max_age_hours: float = WORKFLOW_RECENCY_MAX_HOURS) -> CheckResult:
    """Catches the schedule silently dying -- 60-day inactivity
    auto-disable, a YAML error introduced later, or the runner simply
    never dispatching. See the module docstring for the one thing this
    can't catch (GitHub disabling every scheduled workflow in the repo
    at once, including this healthcheck)."""
    try:
        owner_repo = git_commits._origin_owner_repo()
    except Exception as e:
        return CheckResult("workflow_recency", False, f"could not determine origin owner/repo: {e}")
    result = subprocess.run(
        ["gh", "run", "list", "--repo", owner_repo, "--workflow", "pipeline.yml",
         "--status", "success", "--limit", "1", "--json", "createdAt"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return CheckResult("workflow_recency", False, f"gh run list failed: {result.stderr.strip()}")
    try:
        runs = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return CheckResult("workflow_recency", False, f"gh run list returned invalid JSON: {e}")
    if not runs:
        return CheckResult("workflow_recency", False, "no successful pipeline.yml run found at all")
    created_at = datetime.strptime(runs[0]["createdAt"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
    if age_hours > max_age_hours:
        return CheckResult("workflow_recency", False, f"last successful pipeline.yml run was {age_hours:.1f}h ago (over the {max_age_hours:.0f}h threshold) -- the daily schedule may have stopped firing")
    return CheckResult("workflow_recency", True, f"last successful pipeline.yml run was {age_hours:.1f}h ago")


LOCAL_CHECKS = [
    check_blame_cache_integrity,
    check_workflow_actions_are_sha_pinned,
    check_raw_captures_not_gitignored,
    check_raw_capture_hashes,
    check_ledger_integrity,
    check_calibration_json,
]
GH_CHECKS = [
    check_pages_source_is_actions,
    check_scheduled_workflow_recency,
]


def _run_one(check) -> CheckResult:
    # One check crashing (a malformed ledger, an unreadable file) must
    # not blind the report to every other check's result -- the crash
    # itself is reported as that check's own failure instead.
    try:
        return check()
    except Exception as e:
        return CheckResult(check.__name__, False, f"check raised {type(e).__name__}: {e}")


def run(include_gh_checks: bool = True) -> int:
    checks = LOCAL_CHECKS + (GH_CHECKS if include_gh_checks else [])
    results = [_run_one(check) for check in checks]
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gh", action="store_true", help="Skip checks that shell out to `gh` (Pages source, workflow recency) -- for offline/local runs.")
    args = parser.parse_args()
    raise SystemExit(run(include_gh_checks=not args.no_gh))
