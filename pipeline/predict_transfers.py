#!/usr/bin/env python3
"""Transfer recommendation runner (Phase 13 Block 2.5) — the same
append-only-ledger, never-edited-never-deleted discipline as
pipeline/predict.py, extended to a genuinely different kind of decision:
which transfer(s), if any, this project's real FPL entry should make
before its NEXT deadline, given the squad it actually holds right now.

**Structurally distinct from predict.py, not a variant of it.**
predict.py answers "what's the best 15-man squad from scratch this
week" — every week, independent of any prior week. This module answers
"starting from the REAL squad the entry actually holds, what should
change" — which requires knowing that real squad
(apex_fpl.serving.entry_state), something predict.py never needs. A bug
here must never affect predict.py's own run: the two are wired as
independent steps in pipeline.yml, and this module's own failures are
caught and recorded as a status, not allowed to raise past run().

**Cannot produce a recommendation before any gameweek has settled.**
Transfers are relative to an existing squad; GW1 is the entry's first
squad, not a transfer decision. `entry_state.build_current_squad_state`
returning None is a real, expected, common state for a large part of
this project's life this season (all of GW1) — recorded as a
NO_SETTLED_GAMEWEEK_YET status, not treated as a failure.

**Scope, disclosed here and in every record's caveats, not just in a
report:** this first version uses `horizon=1` (myopic single-gameweek
lookahead), not the `horizon>=4` lookahead policy Phase 7 decisively
validated as superior across 4 independent seasons
(docs/phase7_multiweek_optimizer_report.md). horizon=1 still has its
own real, statistically significant evidence ("transfers help at all,
even myopically" — the same report's other confirmed finding) — but is
not the strongest validated policy. Extending to a genuine multi-
gameweek horizon needs a live multi-gameweek-ahead EP forecasting
pipeline (repeating the team/minutes/attacking models forward across
future fixtures) that doesn't exist yet — real, separate, undone work,
not implemented here to avoid shipping something untested this close to
a deadline.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze
from apex_fpl.optimization import transfers as tr
from apex_fpl.serving import entry_state as es
from apex_fpl.serving import live_data as ld

from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
LEDGER_DIR = REPO_ROOT / "data" / "transfer_recommendations"

SCHEMA_VERSION = "1.0"
MIN_HOURS_BEFORE_DEADLINE = 2.0
HORIZON = 1  # myopic -- see module docstring's "Scope" section
HIT_COST = -4.0
MAX_FREE_TRANSFERS = 5

EXIT_OK = 0
EXIT_TOO_CLOSE_TO_DEADLINE = 3


def _display_path(path: Path) -> str:
    """Mirrors pipeline.predict's own helper of the same name and for
    the same reason: LEDGER_DIR can legitimately point outside
    REPO_ROOT (a test redirecting it to a tmp directory for isolation),
    and Path.relative_to raises rather than degrading gracefully."""
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
        "entry_id": es.ENTRY_ID,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": _git_sha(),
    }


def _build_status_record(target_gw: int, status: str, detail: str, prior_record: dict | None) -> dict:
    body = {**_base_record(target_gw, prior_record), "status": status, "detail": detail, "horizon": HORIZON, "recommendation": None}
    return {**body, "record_id": _content_hash(body)}


def _build_published_record(target_gw: int, squad_state: es.CurrentSquadState, plan_step, candidates_meta: dict, prior_record: dict | None) -> dict:
    def _player(pid: str) -> dict:
        m = candidates_meta.get(pid, {})
        return {"player_id": pid, "name": m.get("name", "unknown"), "position": m.get("position", "unknown"), "team": m.get("team", "unknown")}

    body = {
        **_base_record(target_gw, prior_record),
        "status": "PUBLISHED",
        "horizon": HORIZON,
        "sell_price_approximation_disclosed": True,  # see apex_fpl.serving.entry_state's module docstring
        "recommendation": {
            "as_of_settled_gameweek": squad_state.as_of_gw,
            "free_transfers_available": plan_step.free_transfers_available,
            "transfers_in": [_player(pid) for pid in plan_step.transfers_in],
            "transfers_out": [_player(pid) for pid in plan_step.transfers_out],
            "paid_transfers": plan_step.paid_transfers,
            "hit_points": plan_step.hit_points,
            "bank_after": round(plan_step.bank_after, 1),
            "net_expected_points_this_gw": None,  # see caveats: horizon=1 net EP isn't separately meaningful beyond hit_points, already captured above
        },
        "caveats": [
            "horizon=1 (myopic single-gameweek lookahead), not the horizon>=4 policy Phase 7 decisively validated as stronger across 4 independent seasons -- real evidence exists for myopic too (transfers help at all), just not the strongest validated policy. See module docstring.",
            "Sell prices are approximated as each player's CURRENT price, not their true purchase-price-adjusted value (FPL keeps only 50% of a price rise since purchase on sale) -- this can slightly overstate the money freed by selling a player who has risen in price. See apex_fpl.serving.entry_state's module docstring.",
        ],
    }
    return {**body, "record_id": _content_hash(body)}


def run(dry_run: bool = False) -> int:
    print(f"===== predict_transfers.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")

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

    print(f"--- Fetching real entry state (entry {es.ENTRY_ID}) ---")
    players = ld.load_players()
    now_cost_by_element = {int(pid): int(round(meta["price"] * 10)) for pid, meta in players.items()}
    squad_state = es.build_current_squad_state(now_cost_by_element)

    if squad_state is None:
        print("No gameweek has settled yet for this entry — no transfer decision is possible (this is GW1's expected state).")
        record = _build_status_record(target_gw, "NO_SETTLED_GAMEWEEK_YET", "entry_state.build_current_squad_state returned None", prior_record)
        return _finalize(record, ledger_path, dry_run)

    if squad_state.as_of_gw + 1 != target_gw:
        detail = f"entry's most recently settled gameweek is {squad_state.as_of_gw}, but the pipeline's target gameweek is {target_gw} (expected {squad_state.as_of_gw + 1})"
        print(f"WARNING: {detail} — skipping rather than risk an inconsistent recommendation.")
        record = _build_status_record(target_gw, "SKIPPED_INCONSISTENT_STATE", detail, prior_record)
        return _finalize(record, ledger_path, dry_run)

    print(f"Real squad as of GW{squad_state.as_of_gw}: {len(squad_state.squad_ids)} players, bank={squad_state.bank}, free_transfers={squad_state.free_transfers}")

    print("--- Building this gameweek's expected-points forecast for the full live pool ---")
    from run_production_recommendation import build_player_forecasts
    forecasts = build_player_forecasts(target_gw, log=print)
    sim_results = forecasts["sim_results"]
    candidates_meta = forecasts["candidates_meta"]

    horizon_players = [
        tr.HorizonPlayer(player_id=pid, position=m["position"], team=m["team"], price=m["price"], ep_by_gw=(sim_results[pid].mean_points,))
        for pid, m in candidates_meta.items() if pid in sim_results
    ]

    print("--- Solving the transfer plan (horizon=1) ---")
    plan = tr.plan_transfers(
        current_squad_ids=squad_state.squad_ids, sell_price_by_id=squad_state.sell_price_by_id,
        bank=squad_state.bank, free_transfers=squad_state.free_transfers,
        players=horizon_players, horizon=HORIZON, hit_cost=HIT_COST, max_free_transfers=MAX_FREE_TRANSFERS,
    )
    step = plan.gameweeks[0]
    if step.transfers_in:
        print(f"Recommended: IN {step.transfers_in}, OUT {step.transfers_out}, hit={step.hit_points}")
    else:
        print("Recommended: no transfer this gameweek.")

    record = _build_published_record(target_gw, squad_state, step, candidates_meta, prior_record)
    return _finalize(record, ledger_path, dry_run)


def _finalize(record: dict, ledger_path: Path, dry_run: bool) -> int:
    if record["supersedes"]:
        print(f"Superseding prior GW{record['gameweek']} transfer recommendation (prior record_id={record['supersedes'][:12]}...)")
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
    parser.add_argument("--dry-run", action="store_true", help="Do everything except append to the transfer-recommendation ledger.")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))
