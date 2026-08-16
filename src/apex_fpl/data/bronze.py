"""Bronze-layer immutable snapshot capture for FPL data sources.

Per the project's anti-leakage data design: the FPL API exposes no
historical point-in-time mode, so the only way to ever have a genuine
pre-deadline snapshot for a given gameweek is to capture one ourselves,
now, before that deadline passes. Snapshots written here are never
mutated or overwritten after being written — each capture gets its own
timestamped file plus a sidecar metadata file recording provenance
(source, retrieved_at, payload hash, schema version, HTTP status).

This module deliberately does NOT parse payloads into canonical entities
(that is Silver-layer work, src/apex_fpl/entities/). Bronze's only job is
faithful, immutable, provenance-tracked raw capture — including on a
malformed or unexpectedly-shaped response: retry/backoff below only
covers transient network failures (timeouts, connection errors, 5xx), not
schema problems, and capture still succeeds and writes the raw bytes even
if the payload's shape has changed underneath us. Schema validation is
deliberately NOT here — it lives in pipeline/fpl_client.py instead, which
wraps this module rather than modifying its capture-always-succeeds
contract (see that module's docstring for why).

`snapshot_root` is an optional override on both `capture_snapshot()` and
`latest_snapshot()`, added so the live Phase-13 pipeline can write into a
committed, per-gameweek location (data/raw/gw{n}/) instead of this
module's own default (data/snapshots/bronze/, gitignored, symlinked to
outside ~/Documents for local-launchd/TCC reasons — see
docs/phase12_production_system_report.md). Every existing call site that
doesn't pass it gets byte-for-byte the same behavior as before this
parameter existed — see tests/unit/test_bronze.py's dedicated coverage
for that claim, not just an assertion of it in prose.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_ROOT = REPO_ROOT / "data" / "snapshots" / "bronze"

SCHEMA_VERSION = "1"
LICENSE_STATUS = (
    "unauthenticated public read-only FPL API; no auth token or ToS "
    "acceptance required as of 2026-08-14. Re-verify periodically — this "
    "is not a substitute for reading the Premier League's actual terms."
)

SOURCES: dict[str, str] = {
    "bootstrap_static": "https://fantasy.premierleague.com/api/bootstrap-static/",
    "fixtures": "https://fantasy.premierleague.com/api/fixtures/",
}

USER_AGENT = "apex-fpl-research/0.1 (local research use, non-commercial)"


@dataclass(frozen=True)
class SnapshotMeta:
    source: str
    url: str
    retrieved_at: str  # ISO8601 UTC, second precision
    season: str
    current_event_id: int | None
    next_event_id: int | None
    raw_payload_hash: str  # sha256 of raw response bytes
    schema_version: str
    license_status: str
    http_status: int
    content_length: int


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


MAX_FETCH_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 30.0


def fetch_raw(source: str, timeout: int = 30, max_attempts: int = MAX_FETCH_ATTEMPTS) -> tuple[bytes, int]:
    """Capped exponential backoff on transient failures (timeouts,
    connection errors, HTTP error status via raise_for_status) — 2s, 4s,
    8s, capped at BACKOFF_CAP_SECONDS. A successful first attempt is
    unaffected (no delay, same return value as before this existed). On
    final exhaustion, re-raises the last real exception rather than
    swallowing it — callers that already handled/propagated
    RequestException see identical behavior to before, just later."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {list(SOURCES)}")
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(SOURCES[source], timeout=timeout, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            return resp.content, resp.status_code
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_CAP_SECONDS))
    assert last_exc is not None
    raise last_exc


def _extract_event_ids(bootstrap_payload: dict[str, Any]) -> tuple[int | None, int | None]:
    current = next_ = None
    for e in bootstrap_payload.get("events", []):
        if e.get("is_current"):
            current = e["id"]
        if e.get("is_next"):
            next_ = e["id"]
    return current, next_


def capture_snapshot(source: str, season: str = "2026/27", snapshot_root: Path | None = None) -> Path:
    """Fetch `source` and write an immutable Bronze snapshot + metadata.

    Never overwrites an existing file. On a same-second collision (two
    captures of the same source within one second) a numeric suffix is
    appended rather than clobbering the earlier snapshot.

    `snapshot_root` defaults to this module's own SNAPSHOT_ROOT (looked
    up fresh from the module namespace on every call, not bound at def
    time — this is what makes existing tests' `monkeypatch.setattr(bronze,
    "SNAPSHOT_ROOT", tmp_path)` pattern keep working unchanged). Pass it
    explicitly to write somewhere else, e.g. the live pipeline's
    data/raw/gw{n}/ (see module docstring).
    """
    resolved_root = snapshot_root if snapshot_root is not None else SNAPSHOT_ROOT
    raw, status = fetch_raw(source)
    ts = _utc_now_compact()

    out_dir = resolved_root / source
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_path = out_dir / f"{ts}.json"
    suffix = 0
    while payload_path.exists():
        suffix += 1
        payload_path = out_dir / f"{ts}_{suffix}.json"

    payload_path.write_bytes(raw)

    current_event_id = next_event_id = None
    if source == "bootstrap_static":
        try:
            parsed = json.loads(raw)
            current_event_id, next_event_id = _extract_event_ids(parsed)
        except json.JSONDecodeError:
            pass

    meta = SnapshotMeta(
        source=source,
        url=SOURCES[source],
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        season=season,
        current_event_id=current_event_id,
        next_event_id=next_event_id,
        raw_payload_hash=_sha256(raw),
        schema_version=SCHEMA_VERSION,
        license_status=LICENSE_STATUS,
        http_status=status,
        content_length=len(raw),
    )
    meta_path = payload_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(asdict(meta), indent=2))
    return payload_path


def capture_all(season: str = "2026/27") -> dict[str, Path]:
    return {source: capture_snapshot(source, season=season) for source in SOURCES}


def latest_snapshot(source: str, snapshot_root: Path | None = None) -> Path | None:
    """Return the most recent payload path for `source`, or None if none exist."""
    resolved_root = snapshot_root if snapshot_root is not None else SNAPSHOT_ROOT
    out_dir = resolved_root / source
    if not out_dir.exists():
        return None
    payloads = sorted(p for p in out_dir.glob("*.json") if not p.name.endswith(".meta.json"))
    return payloads[-1] if payloads else None


if __name__ == "__main__":
    for src, path in capture_all().items():
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        print(f"{src}: wrote {path.relative_to(REPO_ROOT)}  "
              f"({meta['content_length']} bytes, sha256={meta['raw_payload_hash'][:12]}...)")
