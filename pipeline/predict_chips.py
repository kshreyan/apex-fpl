#!/usr/bin/env python3
"""Chip recommendation runner (Phase 13 Block 2.8 (a)) — same append-
only-ledger discipline as predict.py/predict_transfers.py, extended to
a third kind of decision: should this week's chip (Bench Boost/Triple
Captain/Free Hit) be played now, later, or is it not available.

**Bench Boost / Triple Captain use the MODEL'S OWN recommended squad for
THIS gameweek, not the real entry's actual live picks — deliberately.**
The real entry's picks for gameweek t only become visible via the
public API AFTER gameweek t's deadline passes (apex_fpl.serving.
entry_state, confirmed live: 404 pre-deadline) — by which point a BB/TC
decision for that same gameweek is already too late to act on. Using
the model's own from-scratch recommended squad (already computed for
the main prediction, before the deadline) sidesteps this entirely, and
matches CLAUDE.md's own stated premise that the real entry plays this
model's picks.

**Free Hit uses the REAL held squad instead** (apex_fpl.serving.
entry_state.build_current_squad_state, the same one predict_transfers.py
uses) — Free Hit's value is inherently a comparison against the squad
you already own and would revert to, which has no such timing problem:
it reflects the last SETTLED gameweek, always knowable well before the
next deadline.

**Wildcard is NOT computed here — a real, disclosed gap, not an
oversight.** Its value needs a multi-gameweek-ahead EP forecast (the
same live forecasting pipeline predict_transfers.py's horizon=1 scope
explicitly deferred) to compare a constrained vs. unconstrained horizon
total; that pipeline doesn't exist yet.

**The ledger is one file PER CHIP, not per gameweek** (data/chip_
observations/{chip_name}.jsonl) — deliberately different from
predictions/transfer_recommendations' per-gameweek-file convention,
because this ledger's natural key is a TIME SERIES within a chip's
window, not a single gameweek's decision; the 1/e stopping rule needs
the whole sequence observed so far within the current half's window to
decide anything.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze
from apex_fpl.optimization import chips as ch
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import chip_windows as cw
from apex_fpl.serving import entry_state as es
from apex_fpl.serving import live_data as ld

from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
LEDGER_DIR = REPO_ROOT / "data" / "chip_observations"

SCHEMA_VERSION = "1.0"
MIN_HOURS_BEFORE_DEADLINE = 2.0
CHIP_NAMES = ["bboost", "3xc", "freehit"]  # wildcard excluded -- see module docstring

EXIT_OK = 0
EXIT_TOO_CLOSE_TO_DEADLINE = 3


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _content_hash(record_without_id: dict) -> str:
    return hashlib.sha256(json.dumps(record_without_id, sort_keys=True).encode()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "log", "-1", "--format=%H"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _read_ledger(chip_name: str) -> list[dict]:
    path = LEDGER_DIR / f"{chip_name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _latest_per_gw(records: list[dict]) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for r in records:
        latest[r["gameweek"]] = r  # later lines in the file are more recent -- last write wins
    return latest


def _append_ledger(chip_name: str, record: dict) -> None:
    path = LEDGER_DIR / f"{chip_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _window_size(window: cw.ChipWindow) -> int:
    return window.stop_event - window.start_event + 1


def evaluate_chip(chip_name: str, target_gw: int, marginal_value: float | None, model_version: str) -> dict:
    """Pure decision logic, given an already-computed marginal_value for
    THIS gameweek (None if no valuation could be computed this run, e.g.
    Free Hit with no settled squad state yet) -- separated from the live
    data-fetching in run() so it's directly testable without a live API
    or a full Monte Carlo simulation. A None marginal_value is stored as
    such and correctly excluded from future gameweeks' sequence
    reconstruction (_latest_per_gw's caller already filters on
    `is not None`) -- it must never be silently treated as an observed
    zero, which would corrupt the 1/e rule's calibration threshold."""
    window = cw.active_window(chip_name, target_gw)
    already_played_all = es.already_played_chips()
    already_played_this_half = any(
        c.get("name") == chip_name and window is not None and window.start_event <= c.get("event", -1) <= window.stop_event
        for c in already_played_all
    )

    existing = _latest_per_gw(_read_ledger(chip_name))
    prior_record = existing.get(target_gw)

    if window is None:
        decision, window_info = "WINDOW_NOT_OPEN", None
    elif already_played_this_half:
        decision, window_info = "ALREADY_PLAYED_THIS_HALF", {"half": window.half, "start_event": window.start_event, "stop_event": window.stop_event}
    elif marginal_value is None:
        decision, window_info = "NO_VALUATION_AVAILABLE", {"half": window.half, "start_event": window.start_event, "stop_event": window.stop_event}
    else:
        window_size = _window_size(window)
        sequence = [existing[gw]["marginal_value"] for gw in sorted(existing) if window.start_event <= gw < target_gw and existing[gw].get("marginal_value") is not None]
        sequence.append(marginal_value)
        r = max(1, round(window_size / math.e))
        if len(sequence) <= r:
            decision = "OBSERVING"
        elif ch.should_play_chip_now(sequence, window_size):
            decision = "PLAY_NOW"
        else:
            decision = "WAIT"
        window_info = {"half": window.half, "start_event": window.start_event, "stop_event": window.stop_event, "observation_phase_length": r, "n_observed_including_this_gw": len(sequence)}

    body = {
        "schema_version": SCHEMA_VERSION,
        "supersedes": prior_record["record_id"] if prior_record else None,
        "chip_name": chip_name, "gameweek": target_gw,
        "marginal_value": marginal_value if (window is not None and not already_played_this_half) else None,
        "window": window_info,
        "decision": decision,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": model_version,
    }
    return {**body, "record_id": _content_hash(body)}


def run(dry_run: bool = False) -> int:
    print(f"===== predict_chips.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")

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

    print("--- Building this gameweek's expected-points forecast for the full live pool ---")
    from run_production_recommendation import build_player_forecasts
    forecasts = build_player_forecasts(target_gw, log=print)
    sim_results = forecasts["sim_results"]
    candidates_meta = forecasts["candidates_meta"]

    print("--- Selecting the model's own from-scratch squad this gameweek (used for Bench Boost/Triple Captain) ---")
    ev_candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points, m["availability_probability"]) for pid, m in candidates_meta.items() if pid in sim_results]
    my_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    my_xi = sq.select_starting_xi(my_squad)
    bench_ep = [sim_results[p.player_id].mean_points for p in my_xi.bench]
    captain_ep = sim_results[my_xi.captain.player_id].mean_points
    best_possible_xi_ep = sum(sim_results[p.player_id].mean_points for p in my_xi.starters) + captain_ep

    print("--- Fetching real held squad (used for Free Hit) ---")
    players = ld.load_players()
    now_cost_by_element = {int(pid): int(round(meta["price"] * 10)) for pid, meta in players.items()}
    squad_state = es.build_current_squad_state(now_cost_by_element)

    marginal_values: dict[str, float | None] = {"bboost": ch.value_bench_boost(bench_ep), "3xc": ch.value_triple_captain(captain_ep)}
    if squad_state is not None and squad_state.as_of_gw + 1 == target_gw:
        real_candidates = [sq.PlayerCandidate(pid, candidates_meta[pid]["position"], candidates_meta[pid]["team"], candidates_meta[pid]["price"], sim_results[pid].mean_points, candidates_meta[pid]["availability_probability"]) for pid in squad_state.squad_ids if pid in sim_results and pid in candidates_meta]
        if len(real_candidates) == len(squad_state.squad_ids):
            real_xi = sq.select_starting_xi(real_candidates)
            current_xi_ep = sum(sim_results[p.player_id].mean_points for p in real_xi.starters) + sim_results[real_xi.captain.player_id].mean_points
            marginal_values["freehit"] = ch.value_free_hit(current_xi_ep, best_possible_xi_ep)
        else:
            marginal_values["freehit"] = None
            print("Real squad has a player outside the live candidate pool — skipping Free Hit valuation this run.")
    else:
        marginal_values["freehit"] = None
        print("No settled-gameweek squad state available yet (or inconsistent with target_gw) — skipping Free Hit valuation.")

    records = []
    for chip_name in CHIP_NAMES:
        record = evaluate_chip(chip_name, target_gw, marginal_values[chip_name], _git_sha())
        records.append(record)
        print(f"{chip_name}: decision={record['decision']}  marginal_value={record['marginal_value']}")
        if not dry_run:
            _append_ledger(chip_name, record)

    if dry_run:
        print("--- DRY RUN: records built successfully, NOT appended to any ledger ---")
    else:
        print(f"Appended {len(records)} record(s) under {_display_path(LEDGER_DIR)}/")
    return EXIT_OK


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Do everything except append to the chip-observation ledgers.")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))
