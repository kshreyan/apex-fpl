#!/usr/bin/env python3
"""Prediction runner (Phase 13, Stage 3) — the file this whole project's
credibility rests on, per the brief this was built from. Every write
here is append-only (data/predictions/gw{n:02d}.jsonl); nothing is ever
edited or deleted; a failure writes nothing at all.

Sequence:
1. A throwaway probe fetch (bronze.fetch_raw, no caching) of
   bootstrap-static AND fixtures, just to determine which gameweek is
   next and what phase it's in. Two independent facts, not one global
   phase — see pipeline/gw_state.py's docstring for why.
2. If the season has ended, or the target gameweek isn't PRE_DEADLINE,
   or the deadline is under MIN_HOURS_BEFORE_DEADLINE away: exit without
   writing anything. The under-2-hours case gets its own exit code
   (EXIT_TOO_CLOSE_TO_DEADLINE) so the workflow can flag it as a warning,
   not a silent no-op — everything else is ordinary, expected daily
   behavior for most of a season.
3. A fully blank gameweek (every team missing a fixture) gets a
   `status: "BLANK_GAMEWEEK"` record WITHOUT ever invoking the model —
   checked from the probe data alone, so the model's own
   `no live fixtures found` exception path never has to fire for this
   expected case.
4. Otherwise: a REAL fetch+validate+cache via pipeline.fpl_client (this
   is the one that gets recorded in data_sources and feeds Silver), a
   Silver rebuild scoped to this gameweek's freshly-captured snapshots,
   then the model itself via
   scripts/run_production_recommendation.py::generate_recommendation
   (write_artifact=False — this module owns the one real prediction
   record now, not a second differently-shaped file).
5. The record is mapped into the Stage-1 schema and appended. If a prior
   record already exists for this gameweek (a re-run closer to the
   deadline with fresher data), the new one's `supersedes` names the
   prior record's id — the old line is never touched.

If the model raises at any point, the exception propagates unchanged —
no partial record is ever built, let alone written.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze
from apex_fpl.entities import silver

from pipeline import fpl_client
from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
# scripts/ isn't a package (its files are run as standalone scripts elsewhere in this
# project too, e.g. run_phase9_chip_valuation_demo.py importing a sibling the same way)
# -- matching that existing convention rather than inventing a new one.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"

SCHEMA_VERSION = "1.0"
MIN_HOURS_BEFORE_DEADLINE = 2.0

EXIT_OK = 0
EXIT_TOO_CLOSE_TO_DEADLINE = 3


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    """Path relative to REPO_ROOT when possible, else the absolute path
    -- mirrors apex_fpl.backtesting.replay's own helper of the same name
    and for the same reason: a raw snapshot can legitimately live outside
    REPO_ROOT (e.g. a test redirecting RAW_DATA_ROOT to a tmp directory
    for isolation), and `Path.relative_to` raises ValueError rather than
    degrading gracefully in that case."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _content_hash(record_without_id: dict) -> str:
    return hashlib.sha256(json.dumps(record_without_id, sort_keys=True).encode()).hexdigest()


def _git_sha(path: str | None = None) -> str:
    args = ["git", "log", "-1", "--format=%H"]
    if path is not None:
        args += ["--", path]
    return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _read_last_ledger_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def _append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _base_record(target_gw: int, phase_info: "gs.GameweekPhaseInfo", prior_record: dict | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "supersedes": prior_record["record_id"] if prior_record else None,
        "gameweek": target_gw,
        "teams_without_fixture": sorted(phase_info.teams_without_fixture),
        "teams_with_double_fixture": sorted(phase_info.teams_with_double_fixture),
        "deadline_time_utc": phase_info.deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gw_status_at_generation": phase_info.phase.value,
        "model_version": _git_sha("src/apex_fpl"),
        "pipeline_version": _git_sha(),
    }


def _build_blank_gameweek_record(target_gw: int, phase_info, prior_record: dict | None) -> dict:
    body = {
        **_base_record(target_gw, phase_info, prior_record),
        "status": "BLANK_GAMEWEEK",
        "training_data_fingerprint": None,
        "data_sources": [],
        "model_caveats": [],
        "squad": None,
        "calls": [],
    }
    return {**body, "record_id": _content_hash(body)}


def _build_published_record(target_gw: int, phase_info, result: dict, prior_record: dict | None, data_sources: list[dict]) -> dict:
    squad_by_id = {p["player_id"]: p for p in result["squad"]}
    double_teams = set(result["teams_with_double_fixture"])

    calls = []
    for pid in result["starting_xi"]:
        p = squad_by_id[pid]
        fixture_count = 2 if p["team"] in double_teams else 1
        calls.append({
            "id": f"gw{target_gw:02d}-player-{pid}-points", "type": "points_forecast",
            "subject": {"kind": "player", "player_id": pid, "fixture_count": fixture_count},
            "claim": f"{p['name']} projected {p['expected_points']} pts", "value": p["expected_points"],
        })

    captain_id = result["captain"]
    captain_ep = squad_by_id[captain_id]["expected_points"]
    calls.append({
        "id": f"gw{target_gw:02d}-captain", "type": "points_forecast",
        "subject": {"kind": "captain", "player_id": captain_id}, "value": round(captain_ep * 2, 3),
    })
    calls.append({
        "id": f"gw{target_gw:02d}-squad-total", "type": "points_forecast",
        "subject": {"kind": "squad_total"}, "value": result["projected_gw_points"],
    })
    calls.append({
        "id": f"gw{target_gw:02d}-captain-haul", "type": "binary_probability",
        # threshold lives here as DATA, not just in the human-readable claim string below --
        # score.py needs a number to check the outcome against, not prose to parse.
        "subject": {"kind": "captain", "player_id": captain_id, "threshold": result["captain_haul_threshold"]},
        "claim": f"Captain scores {result['captain_haul_threshold']}+ points",
        "probability": result["captain_haul_probability"],
    })

    body = {
        **_base_record(target_gw, phase_info, prior_record),
        "status": "PUBLISHED",
        "training_data_fingerprint": result["training_data_fingerprint"],
        "data_sources": data_sources,
        "model_caveats": result["caveats"],
        "squad": {
            "starting_xi": [{"player_id": p["player_id"], "name": p["name"], "position": p["position"], "team": p["team"], "price": p["price"]} for p in result["squad"] if p["player_id"] in result["starting_xi"]],
            "bench_order": [{"player_id": p["player_id"], "name": p["name"], "position": p["position"], "team": p["team"], "price": p["price"]} for p in result["squad"] if p["player_id"] in result["bench_order"]],
            "captain_player_id": result["captain"],
            "vice_captain_player_id": result["vice_captain_fallback"],
        },
        "calls": calls,
    }
    return {**body, "record_id": _content_hash(body)}


def run(dry_run: bool = False) -> int:
    print(f"===== predict.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")

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
        print(f"WARNING: only {phase_info.hours_until_deadline:.2f}h until deadline "
              f"(minimum {MIN_HOURS_BEFORE_DEADLINE}h) — refusing to run this close, given possible Actions delay.")
        return EXIT_TOO_CLOSE_TO_DEADLINE

    ledger_path = PREDICTIONS_DIR / f"gw{target_gw:02d}.jsonl"
    prior_record = _read_last_ledger_record(ledger_path)

    all_teams = {t["name"] for t in bootstrap_probe["teams"]}
    if phase_info.teams_without_fixture == frozenset(all_teams):
        print(f"GW{target_gw} is a fully blank gameweek (zero fixtures for the entire league) — "
              f"recording without invoking the model.")
        record = _build_blank_gameweek_record(target_gw, phase_info, prior_record)
        return _finalize(record, ledger_path, dry_run)

    print("--- Fetching, validating, and caching live data for this gameweek ---")
    bs_path = fpl_client.fetch_and_validate("bootstrap_static", gameweek=target_gw)
    fx_path = fpl_client.fetch_and_validate("fixtures", gameweek=target_gw)
    data_sources = [
        {"source": "bootstrap_static", "raw_cache_path": _display_path(bs_path), "sha256": _hash_file(bs_path)},
        {"source": "fixtures", "raw_cache_path": _display_path(fx_path), "sha256": _hash_file(fx_path)},
    ]

    print("--- Rebuilding Silver from the freshly-captured snapshots ---")
    silver.run_build(bronze_root=RAW_DATA_ROOT / f"gw{target_gw:02d}")

    print("--- Invoking the model ---")
    from run_production_recommendation import generate_recommendation
    result = generate_recommendation(target_gw, write_artifact=False)

    record = _build_published_record(target_gw, phase_info, result, prior_record, data_sources)
    return _finalize(record, ledger_path, dry_run)


def _finalize(record: dict, ledger_path: Path, dry_run: bool) -> int:
    if record["supersedes"]:
        print(f"Superseding prior GW{record['gameweek']} prediction (prior record_id={record['supersedes'][:12]}...)")

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
    parser.add_argument("--dry-run", action="store_true", help="Do everything except append to the predictions ledger.")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))
