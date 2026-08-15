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
faithful, immutable, provenance-tracked raw capture.
"""
from __future__ import annotations

import hashlib
import json
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


def fetch_raw(source: str, timeout: int = 30) -> tuple[bytes, int]:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known: {list(SOURCES)}")
    resp = requests.get(
        SOURCES[source], timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    return resp.content, resp.status_code


def _extract_event_ids(bootstrap_payload: dict[str, Any]) -> tuple[int | None, int | None]:
    current = next_ = None
    for e in bootstrap_payload.get("events", []):
        if e.get("is_current"):
            current = e["id"]
        if e.get("is_next"):
            next_ = e["id"]
    return current, next_


def capture_snapshot(source: str, season: str = "2026/27") -> Path:
    """Fetch `source` and write an immutable Bronze snapshot + metadata.

    Never overwrites an existing file. On a same-second collision (two
    captures of the same source within one second) a numeric suffix is
    appended rather than clobbering the earlier snapshot.
    """
    raw, status = fetch_raw(source)
    ts = _utc_now_compact()

    out_dir = SNAPSHOT_ROOT / source
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


def latest_snapshot(source: str) -> Path | None:
    """Return the most recent payload path for `source`, or None if none exist."""
    out_dir = SNAPSHOT_ROOT / source
    if not out_dir.exists():
        return None
    payloads = sorted(p for p in out_dir.glob("*.json") if not p.name.endswith(".meta.json"))
    return payloads[-1] if payloads else None


if __name__ == "__main__":
    for src, path in capture_all().items():
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        print(f"{src}: wrote {path.relative_to(REPO_ROOT)}  "
              f"({meta['content_length']} bytes, sha256={meta['raw_payload_hash'][:12]}...)")
