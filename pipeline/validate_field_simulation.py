#!/usr/bin/env python3
"""Field-simulator validation check (Phase 13 — the second half of item
5 of the "explicitly disclosed, not built" gaps). Once a gameweek
SETTLES, compares pipeline/predict_field.py's pre-deadline prediction
of the competitive field's mean score against the REAL ground truth
(bootstrap-static events[].average_entry_score) — the independent
validation source Phase 10's own research explicitly said didn't exist
in the historical archive (see predict_field.py's module docstring).

Same append-only-ledger, checked-once-never-rechecked discipline as
pipeline/check_execution_divergence.py and for the same reason: this is
a permanent research-validity record, not something to quietly
recompute if a later run's bootstrap-static snapshot happens to differ.

This is NOT a pass/fail gate — there is no threshold below which the
field simulator is declared "wrong." It is a transparency record: the
raw predicted-vs-actual numbers, left for a human (or a later,
statistically-powered comparison across many gameweeks) to judge, matching
this project's standing "no promotion without pre-registered statistical
significance" discipline rather than eyeballing one gameweek's error.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze

from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "field_simulation_predictions"
LEDGER_DIR = REPO_ROOT / "data" / "field_simulation_validation"

SCHEMA_VERSION = "1.0"


def _content_hash(record_without_id: dict) -> str:
    return hashlib.sha256(json.dumps(record_without_id, sort_keys=True).encode()).hexdigest()


def _read_ledger_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def check_gameweek(target_gw: int, bootstrap_static: dict, fixtures: dict, now: datetime) -> dict | None:
    """Returns the appended record, or None if there was nothing to
    check (or nothing NEW to check) this run -- None is a normal,
    common return value, not an error."""
    prediction_lines = _read_ledger_lines(PREDICTIONS_DIR / f"gw{target_gw:02d}.jsonl")
    if not prediction_lines:
        return None
    prediction = prediction_lines[-1]
    if prediction["status"] != "PUBLISHED":
        return None  # BLANK_GAMEWEEK -- no field prediction was ever made

    ledger_path = LEDGER_DIR / f"gw{target_gw:02d}.jsonl"
    if _read_ledger_lines(ledger_path):
        return None  # already checked once -- never re-checked, see module docstring

    phase_info = gs.gameweek_phase(bootstrap_static, fixtures, target_gw, now)
    if phase_info.phase != gs.Phase.SETTLED:
        return None  # not settled yet -- average_entry_score isn't final

    event = next(e for e in bootstrap_static["events"] if e["id"] == target_gw)
    actual_average_entry_score = float(event["average_entry_score"])

    predicted = prediction["prediction"]["predicted_field_mean_score"]
    naive = prediction["prediction"]["naive_ownership_weighted_mean_score"]
    absolute_error = predicted - actual_average_entry_score

    body = {
        "schema_version": SCHEMA_VERSION,
        "gameweek": target_gw,
        "predicted_field_mean_score": predicted,
        "naive_ownership_weighted_mean_score": naive,
        "actual_average_entry_score": actual_average_entry_score,
        "absolute_error": round(absolute_error, 3),
        "percent_error": round(absolute_error / actual_average_entry_score * 100.0, 2) if actual_average_entry_score else None,
        "prediction_record_id": prediction["record_id"],
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    record = {**body, "record_id": _content_hash(body)}
    _append_ledger(ledger_path, record)
    return record


def run() -> int:
    print(f"===== validate_field_simulation.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")
    if not PREDICTIONS_DIR.exists():
        print("No field-simulation predictions directory exists yet — nothing to check.")
        return 0

    bootstrap_static = json.loads(bronze.fetch_raw("bootstrap_static")[0])
    fixtures = json.loads(bronze.fetch_raw("fixtures")[0])
    now = datetime.now(timezone.utc)

    target_gws = sorted(int(p.stem[2:]) for p in PREDICTIONS_DIR.glob("gw*.jsonl"))
    for gw in target_gws:
        record = check_gameweek(gw, bootstrap_static, fixtures, now)
        if record is not None:
            print(f"GW{gw}: predicted={record['predicted_field_mean_score']:.2f} actual={record['actual_average_entry_score']:.2f} error={record['absolute_error']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
