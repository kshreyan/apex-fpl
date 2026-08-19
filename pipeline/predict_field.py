#!/usr/bin/env python3
"""Field-simulator live prediction runner (Phase 13 — item 5 of the
"explicitly disclosed, not built" gaps: external validation of the
field/rank simulator).

Phase 10's field/rank simulator (`apex_fpl.simulation.field`) was built
and demonstrated (`scripts/run_phase10_field_simulation_demo.py`)
entirely against the historical Vaastav archive, whose own docstring
states the real limitation this module exists to close: "there is no
real average_entry_score or rank-distribution data in this historical
archive to validate the field simulation's absolute scale against ...
a genuinely independent validation source ... is a clear next step."
The live 2026/27 season has exactly that independent ground truth
(bootstrap-static events[].average_entry_score, already used by
pipeline/score.py's average-manager baseline) — but only once a
gameweek settles, which is too late to be useful for a decision. So,
same commit-before-the-outcome-is-known pattern as predict.py/score.py:
this module records the field simulator's predicted mean score for the
live field BEFORE each gameweek's deadline, using REAL live ownership
(bootstrap-static's selected_by_percent via
apex_fpl.serving.live_data.load_players, not the historical archive
Phase 10's own scripts use); pipeline/validate_field_simulation.py
compares that prediction against the real average_entry_score once the
gameweek settles.

This is a validation harness, not a live decision feature — nothing
here feeds a squad, transfer, or chip recommendation, and a bug here
must never cost predict.py's own deadline. Wired as an independent,
best-effort pipeline.yml step, same discipline as predict_transfers.py
and predict_chips.py.

N_RIVALS=2000, SEED=2026: Phase 10's own validated demo configuration,
reused unchanged rather than re-picked here.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze
from apex_fpl.serving import live_data as ld
from apex_fpl.simulation import field as fsim

from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
LEDGER_DIR = REPO_ROOT / "data" / "field_simulation_predictions"

SCHEMA_VERSION = "1.0"
MIN_HOURS_BEFORE_DEADLINE = 2.0
N_RIVALS = 2000  # matches scripts/run_phase10_field_simulation_demo.py's own validated choice
SEED = 2026

EXIT_OK = 0
EXIT_TOO_CLOSE_TO_DEADLINE = 3


def _display_path(path: Path) -> str:
    """Mirrors pipeline.predict's own helper of the same name and for
    the same reason: LEDGER_DIR can legitimately point outside
    REPO_ROOT (a test redirecting it to a tmp directory for isolation)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _content_hash(record_without_id: dict) -> str:
    return hashlib.sha256(json.dumps(record_without_id, sort_keys=True).encode()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "log", "-1", "--format=%H"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _read_last_ledger_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def _append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _base_record(target_gw: int, prior_record: dict | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "supersedes": prior_record["record_id"] if prior_record else None,
        "gameweek": target_gw,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": _git_sha(),
    }


def _build_status_record(target_gw: int, status: str, detail: str, prior_record: dict | None) -> dict:
    body = {**_base_record(target_gw, prior_record), "status": status, "detail": detail, "n_rivals": N_RIVALS, "prediction": None}
    return {**body, "record_id": _content_hash(body)}


def _build_published_record(target_gw: int, predicted_field_mean_score: float, naive_ownership_weighted_mean: float, n_owned_candidates: int, prior_record: dict | None) -> dict:
    body = {
        **_base_record(target_gw, prior_record),
        "status": "PUBLISHED",
        "n_rivals": N_RIVALS,
        "detail": None,
        "prediction": {
            "predicted_field_mean_score": round(predicted_field_mean_score, 3),
            "naive_ownership_weighted_mean_score": round(naive_ownership_weighted_mean, 3),
            "n_owned_candidates": n_owned_candidates,
        },
        "caveats": [
            "Synthetic rival squads are sampled from real live ownership (selected_by_percent) but are NOT budget-constrained and do not model cross-player ownership correlation -- see apex_fpl.simulation.field's own module docstring for the full, stated limitations.",
            "predicted_field_mean_score is the field Monte Carlo's mean; naive_ownership_weighted_mean_score is an independent, cruder cross-check (no bench/captaincy adjustment), not a second ground truth.",
        ],
    }
    return {**body, "record_id": _content_hash(body)}


def run(dry_run: bool = False) -> int:
    print(f"===== predict_field.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")

    print("--- Probing bootstrap-static/fixtures to determine gameweek/phase (not cached) ---")
    bootstrap_probe = json.loads(bronze.fetch_raw("bootstrap_static")[0])
    fixtures_probe = json.loads(bronze.fetch_raw("fixtures")[0])

    now = datetime.now(timezone.utc)
    target_gw = gs.next_prediction_gameweek(bootstrap_probe, now)
    if target_gw is None:
        print("Season has ended (no gameweek with a future deadline) — nothing to do.")
        return EXIT_OK

    phase_info = gs.gameweek_phase(bootstrap_probe, fixtures_probe, target_gw, now)
    print(f"GW{target_gw}: phase={phase_info.phase.value}, hours_until_deadline={phase_info.hours_until_deadline:.2f}")

    if phase_info.phase != gs.Phase.PRE_DEADLINE:
        print(f"Phase is {phase_info.phase.value}, not PRE_DEADLINE — exiting cleanly, no-op.")
        return EXIT_OK
    if phase_info.hours_until_deadline < MIN_HOURS_BEFORE_DEADLINE:
        print(f"WARNING: only {phase_info.hours_until_deadline:.2f}h until deadline — refusing to run this close.")
        return EXIT_TOO_CLOSE_TO_DEADLINE

    ledger_path = LEDGER_DIR / f"gw{target_gw:02d}.jsonl"
    prior_record = _read_last_ledger_record(ledger_path)

    print(f"--- Building an expected-points forecast for the full live pool (GW{target_gw}) ---")
    from run_production_recommendation import build_player_forecasts
    try:
        forecast = build_player_forecasts(target_gw, log=print)
    except ValueError as e:
        print(f"GW{target_gw} has no live fixtures (a blank gameweek) — {e}")
        record = _build_status_record(target_gw, "BLANK_GAMEWEEK", f"GW{target_gw} has zero fixtures", prior_record)
        return _finalize(record, ledger_path, dry_run)

    sim_results = forecast["sim_results"]
    candidates_meta = forecast["candidates_meta"]

    print("--- Loading real live ownership (selected_by_percent) ---")
    players = ld.load_players()
    ownership_fractions = {pid: meta["selected_by_percent"] / 100.0 for pid, meta in players.items() if pid in candidates_meta}
    n_owned = len(ownership_fractions)
    print(f"Owned candidates in this gameweek's pool: {n_owned}")

    print(f"--- Sampling {N_RIVALS} synthetic rival squads from real live ownership ---")
    rival_squads = fsim.sample_synthetic_rival_squads(ownership_fractions, candidates_meta, n_rivals=N_RIVALS, seed=SEED)
    field_scores = fsim.simulate_field_scores(rival_squads, sim_results, candidates_meta)
    predicted_field_mean_score = float(field_scores.mean())
    naive_mean = fsim.naive_ownership_weighted_mean_score(ownership_fractions, sim_results)
    print(f"Field mean simulated score: {predicted_field_mean_score:.2f} (naive cross-check: {naive_mean:.2f})")

    record = _build_published_record(target_gw, predicted_field_mean_score, naive_mean, n_owned, prior_record)
    return _finalize(record, ledger_path, dry_run)


def _finalize(record: dict, ledger_path: Path, dry_run: bool) -> int:
    if record["supersedes"]:
        print(f"Superseding prior GW{record['gameweek']} field prediction (prior record_id={record['supersedes'][:12]}...)")
    if dry_run:
        print("--- DRY RUN: record built successfully, NOT appended to the ledger ---")
        print(json.dumps(record, indent=2))
        return EXIT_OK
    _append_ledger(ledger_path, record)
    print(f"Appended to {_display_path(ledger_path)} (record_id={record['record_id'][:12]}..., status={record['status']})")
    return EXIT_OK


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Do everything except append to the field-prediction ledger.")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))
