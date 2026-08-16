"""Resolves which git commit introduced a specific line of an append-only
ledger, for the "proof this prediction preceded the deadline" link on
each gameweek page (Phase 13, Stage 5).

Two real problems, both surfaced while designing this stage:

1. Sequencing. build_site.py runs BEFORE the commit that will contain
   today's newly-appended ledger lines (per CLAUDE.md: /data and /docs
   are committed together, after the whole pipeline finishes). `git
   blame` already reports an uncommitted line with the synthetic
   all-zero SHA ("Not Committed Yet") -- resolve_commit_shas_for_ledger()
   surfaces that as None rather than a broken link. Since build_site.py
   fully regenerates the site from scratch every run, an unresolved link
   self-heals the next time it runs, once that commit exists.

2. Fragility. `git blame` resolves to whichever commit last touched a
   line's CURRENT position in the file -- so a future reformat of a
   ledger (re-indentation, a tooling change) would silently reassign
   every already-resolved SHA, even though the ledger's actual content
   never semantically changed. The fix: once a record's SHA has been
   resolved to a real commit, it's cached by record_id in
   data/site/commit_sha_cache.json and never recomputed from blame
   again -- the cache is authoritative from that point on, not blame.
   (Stage 7 adds a healthcheck that independently re-runs blame and
   asserts it still agrees with the cache, to catch the cache itself
   being corrupted or hand-edited -- not implemented in this module.)
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = REPO_ROOT / "data" / "site" / "commit_sha_cache.json"

NOT_COMMITTED_SHA = "0" * 40
_BLAME_HEADER_RE = re.compile(r"^([0-9a-f]{40}) (\d+) (\d+)(?: \d+)?$")


def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text())


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def _blame_shas(path: Path) -> list[str]:
    """One SHA per line of `path` as it exists in the working tree right
    now, in file order. Blaming the working tree (no revision argument)
    rather than HEAD is deliberate -- it's what makes an uncommitted line
    show up as NOT_COMMITTED_SHA instead of being silently absent. A file
    git doesn't know about at all (never added) blames every line as
    not-committed the same way."""
    if not path.exists():
        return []
    result = subprocess.run(
        ["git", "blame", "--line-porcelain", "--", str(path)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return [NOT_COMMITTED_SHA] * len(path.read_text().splitlines())
    shas = []
    for line in result.stdout.splitlines():
        m = _BLAME_HEADER_RE.match(line)
        if m:
            shas.append(m.group(1))
    return shas


def resolve_commit_shas_for_ledger(path: Path, records: list[dict]) -> dict[str, str | None]:
    """`records` is the full, in-file-order list of ledger lines for
    `path` (each a dict with a `record_id` key). Returns record_id ->
    resolved commit SHA, or None if that line hasn't been committed yet.
    Cache-first per the module docstring: a record_id already in the
    persisted cache is never re-blamed."""
    cache = _load_cache()
    blamed: list[str] | None = None
    resolved: dict[str, str | None] = {}
    cache_dirty = False

    for i, record in enumerate(records):
        record_id = record["record_id"]
        if record_id in cache:
            resolved[record_id] = cache[record_id]
            continue
        if blamed is None:
            blamed = _blame_shas(path)
        sha = blamed[i] if i < len(blamed) else None
        if sha is not None and sha != NOT_COMMITTED_SHA:
            resolved[record_id] = sha
            cache[record_id] = sha
            cache_dirty = True
        else:
            resolved[record_id] = None

    if cache_dirty:
        _save_cache(cache)
    return resolved


def verify_cache_matches_fresh_blame(path: Path, records: list[dict]) -> list[dict]:
    """Stage 7 healthcheck support: for every record_id in `records`
    (the ledger's current, full, in-file-order content) that's already
    present in the persisted cache, re-blames `path` from scratch --
    bypassing the cache entirely, unlike resolve_commit_shas_for_ledger
    -- and reports any disagreement. Returns a list of mismatch dicts
    (empty means the cache is still trustworthy). This is the actual
    check for the fragility this module's docstring describes: the
    cache is supposed to be authoritative forever once written, so any
    disagreement here means either the cache was corrupted/hand-edited,
    or something rewrote history underneath it -- either way, it should
    never legitimately happen, and both are more than the risk of a
    reformat this module was built to defend against."""
    cache = _load_cache()
    blamed = _blame_shas(path)
    mismatches = []
    for i, record in enumerate(records):
        record_id = record["record_id"]
        if record_id not in cache:
            continue
        fresh = blamed[i] if i < len(blamed) else None
        if fresh != cache[record_id]:
            mismatches.append({"record_id": record_id, "path": str(path), "cached_sha": cache[record_id], "fresh_sha": fresh})
    return mismatches


def _origin_owner_repo() -> str:
    url = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if url.endswith(".git"):
        url = url[: -len(".git")]
    if url.startswith("git@github.com:"):
        return url[len("git@github.com:"):]
    if "github.com/" in url:
        return url.split("github.com/", 1)[1]
    raise ValueError(f"origin remote is not a recognizable github.com URL: {url!r}")


def commit_url(sha: str) -> str:
    return f"https://github.com/{_origin_owner_repo()}/commit/{sha}"
