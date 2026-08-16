"""Live FPL data client for the automated pipeline (Phase 13, Stage 2).

Wraps `apex_fpl.data.bronze.capture_snapshot()` rather than re-fetching
independently — one HTTP client, one retry/backoff implementation, one
place to fix when the FPL API changes (see bronze.py's own docstring for
the reasoning, and the decision record in the Stage-1 design
conversation this pipeline was built from). Retry/backoff and
fail-loud-on-HTTP-error already live in bronze.py; this module adds the
one thing that must NOT live there: schema validation that can refuse to
let a run proceed. Bronze's job is "always capture, even a payload that's
changed shape underneath us" — that raw evidence is exactly what you'd
need to diagnose the schema break later. If validation lived inside
capture itself, a schema change would mean losing the very evidence
needed to debug it.

Field specs below are deliberately narrow: only what
apex_fpl.serving.live_data / apex_fpl.entities.silver / the squad
optimizer actually read today, not a general-purpose mirror of FPL's
full schema. Types were checked against a real, already-captured live
bootstrap-static/fixtures payload (not assumed from memory) —
`selected_by_percent` is a surprising one: it's a STRING in the real API
("31.2"), not a JSON number, confirmed directly rather than guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

from apex_fpl.data import bronze

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"


class SchemaValidationError(Exception):
    """Raised when a captured payload doesn't match the shape this
    pipeline depends on. The raw payload is already safely written to
    disk by the time this can ever be raised — this only blocks the
    pipeline from proceeding to use data it can't trust, it never causes
    data loss."""


# nullable fields are expressed as a tuple including NoneType, matching
# real, confirmed API behavior (e.g. an unplayed fixture's scores)
_NULLABLE_INT = (int, type(None))
_NULLABLE_STR = (str, type(None))

BOOTSTRAP_LIST_FIELD_SPECS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "events": {
        "id": int, "deadline_time": str, "finished": bool,
        "data_checked": bool, "is_current": bool, "is_next": bool,
        "average_entry_score": int,  # needed by pipeline/score.py's average-manager baseline; confirmed present (0 pre-season) against real data
    },
    "teams": {
        "id": int, "name": str,
        "strength_overall_home": int, "strength_overall_away": int,
    },
    "elements": {
        "id": int, "web_name": str, "team": int, "element_type": int,
        "status": str, "now_cost": int,
        "selected_by_percent": str,  # confirmed: a string in the real API, not a number
        "event_points": int,  # points scored in the most recently settled gameweek -- see pipeline/score.py's docstring for the staleness caveat this implies
    },
    "element_types": {
        "id": int, "squad_select": int, "squad_min_play": int, "squad_max_play": int,
    },
}

FIXTURE_FIELD_SPEC: dict[str, type | tuple[type, ...]] = {
    "id": int, "event": _NULLABLE_INT, "team_h": int, "team_a": int,
    "kickoff_time": _NULLABLE_STR, "finished": bool,
    "team_h_score": _NULLABLE_INT, "team_a_score": _NULLABLE_INT,
    "team_h_difficulty": int, "team_a_difficulty": int,
}


def _check_field(value_holder: dict, field: str, expected_type, path: str) -> None:
    if field not in value_holder:
        raise SchemaValidationError(f"{path}.{field}: missing")
    value = value_holder[field]
    if not isinstance(value, expected_type):
        expected_name = expected_type.__name__ if isinstance(expected_type, type) else expected_type
        raise SchemaValidationError(f"{path}.{field}: expected {expected_name}, got {type(value).__name__} ({value!r})")


def validate_bootstrap_static(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise SchemaValidationError(f"bootstrap_static: expected a dict at top level, got {type(payload).__name__}")
    for list_key, item_spec in BOOTSTRAP_LIST_FIELD_SPECS.items():
        items = payload.get(list_key)
        if items is None:
            raise SchemaValidationError(f"bootstrap_static.{list_key}: missing")
        if not isinstance(items, list):
            raise SchemaValidationError(f"bootstrap_static.{list_key}: expected list, got {type(items).__name__}")
        for i, item in enumerate(items):
            for field, expected_type in item_spec.items():
                _check_field(item, field, expected_type, f"bootstrap_static.{list_key}[{i}]")


def validate_fixtures(payload: list) -> None:
    if not isinstance(payload, list):
        raise SchemaValidationError(f"fixtures: expected a list at top level, got {type(payload).__name__}")
    for i, item in enumerate(payload):
        for field, expected_type in FIXTURE_FIELD_SPEC.items():
            _check_field(item, field, expected_type, f"fixtures[{i}]")


_VALIDATORS = {"bootstrap_static": validate_bootstrap_static, "fixtures": validate_fixtures}


def fetch_and_validate(source: str, gameweek: int, season: str = "2026/27") -> Path:
    """Captures `source` via bronze (with its own retry/backoff) into the
    committed data/raw/gw{n:02d}/ location, then validates its shape.
    Returns the path to the raw payload on success. Raises
    SchemaValidationError on a shape mismatch — the raw file is still on
    disk at that point, this just refuses to let the caller trust it."""
    if source not in _VALIDATORS:
        raise ValueError(f"no validator registered for source {source!r}; known: {list(_VALIDATORS)}")

    snapshot_root = RAW_DATA_ROOT / f"gw{gameweek:02d}"
    path = bronze.capture_snapshot(source, season=season, snapshot_root=snapshot_root)
    payload = json.loads(path.read_bytes())
    _VALIDATORS[source](payload)
    return path
