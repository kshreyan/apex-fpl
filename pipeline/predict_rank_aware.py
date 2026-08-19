#!/usr/bin/env python3
"""Rank-aware squad prediction runner (Phase 13 — item 3 of the
"explicitly disclosed, not built" gaps: the biggest and most novel,
deliberately built last).

Computes `apex_fpl.optimization.rank_aware.select_rank_aware_squad` for
the live target gameweek: the existing max-EV squad, plus a small set
of ownership-differential alternatives, each scored against a real
live-ownership synthetic field for expected percentile / P(top decile)
rather than raw expected points — see that module's own docstring for
why this can't just be "add rank to the existing MILP objective."

**A SEPARATE, informational artifact — not a replacement for predict.py's
own squad.** This project's standing promotion discipline (pre-
registered comparison, statistical significance) is what would ever
make a rank-aware pick the production default; a single gameweek's
squad choice can't earn that on its own. Wired as an independent,
best-effort pipeline.yml step, same discipline as predict_field.py.

N_RIVALS=2000, SEED=2026: Phase 10's own validated configuration, reused
unchanged. MAX_CANDIDATES=5, MAX_EV_LOSS_FRACTION=0.05, TARGET_METRIC=
"p_top10pct": a deliberately modest first choice (at most 5% of the
max-EV squad's own EV traded away, judged by probability of finishing
in the field's top decile) — not tuned or validated against real
outcomes yet; see the module docstring's own "genuinely differentiable
or multi-swap formulation" note for what a validated version of this
would need.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze
from apex_fpl.optimization import rank_aware as ra
from apex_fpl.optimization import squad as sq
from apex_fpl.serving import live_data as ld

from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
LEDGER_DIR = REPO_ROOT / "data" / "rank_aware_predictions"

SCHEMA_VERSION = "1.0"
MIN_HOURS_BEFORE_DEADLINE = 2.0
N_RIVALS = 2000  # matches scripts/run_phase10_field_simulation_demo.py's own validated choice
SEED = 2026
MAX_CANDIDATES = 5
MAX_EV_LOSS_FRACTION = 0.05
TARGET_METRIC = "p_top10pct"

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
    body = {**_base_record(target_gw, prior_record), "status": status, "detail": detail, "candidates": None, "selected": None}
    return {**body, "record_id": _content_hash(body)}


def _candidate_dict(c: ra.CandidateSquadResult, candidates_meta: dict) -> dict:
    def _player(pid: str) -> dict:
        m = candidates_meta.get(pid, {})
        return {"player_id": pid, "name": m.get("name", "unknown"), "position": m.get("position", "unknown"), "team": m.get("team", "unknown")}

    return {
        "label": c.label,
        "squad": [_player(pid) for pid in c.squad_ids],
        "swapped_out": _player(c.swapped_out) if c.swapped_out else None,
        "swapped_in": _player(c.swapped_in) if c.swapped_in else None,
        "mean_ev": round(c.mean_ev, 3),
        "mean_simulated_score": round(c.mean_simulated_score, 3),
        "mean_percentile": round(c.mean_percentile, 4),
        "p_top10pct": round(c.p_top10pct, 4),
        "p_top25pct": round(c.p_top25pct, 4),
    }


def _build_published_record(target_gw: int, result: ra.RankAwareSelectionResult, candidates_meta: dict, n_owned_candidates: int, prior_record: dict | None) -> dict:
    body = {
        **_base_record(target_gw, prior_record),
        "status": "PUBLISHED",
        "detail": None,
        "n_rivals": N_RIVALS,
        "max_ev_loss_fraction": MAX_EV_LOSS_FRACTION,
        "target_metric": result.target_metric,
        "n_owned_candidates": n_owned_candidates,
        "candidates": [_candidate_dict(c, candidates_meta) for c in result.candidates],
        "selected": _candidate_dict(result.selected, candidates_meta),
        "caveats": [
            "Informational only -- NOT a replacement for the squad predict.py publishes, and not auto-applied to any real decision. See apex_fpl.optimization.rank_aware's own module docstring.",
            "Differentials try at most ONE swap each (the most heavily-owned squad member first) within max_ev_loss_fraction of the max-EV squad's own total EV -- not a combinatorial search.",
            "Synthetic rival squads are sampled from real live ownership but are NOT budget-constrained and do not model cross-player ownership correlation -- see apex_fpl.simulation.field's own module docstring.",
        ],
    }
    return {**body, "record_id": _content_hash(body)}


def run(dry_run: bool = False) -> int:
    print(f"===== predict_rank_aware.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")

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

    all_candidates = [
        sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points, m.get("availability_probability", 1.0))
        for pid, m in candidates_meta.items() if pid in sim_results
    ]

    print(f"--- Selecting a rank-aware squad (n_rivals={N_RIVALS}, max_candidates={MAX_CANDIDATES}, target={TARGET_METRIC}) ---")
    result = ra.select_rank_aware_squad(
        all_candidates, ownership_fractions, sim_results, candidates_meta,
        budget=sq.BUDGET, n_rivals=N_RIVALS, seed=SEED,
        max_candidates=MAX_CANDIDATES, max_ev_loss_fraction=MAX_EV_LOSS_FRACTION, target_metric=TARGET_METRIC,
    )
    print(f"{len(result.candidates)} candidate squad(s) scored; selected={result.selected.label} (p_top10pct={result.selected.p_top10pct:.3f}, mean_ev={result.selected.mean_ev:.2f})")

    record = _build_published_record(target_gw, result, candidates_meta, n_owned, prior_record)
    return _finalize(record, ledger_path, dry_run)


def _finalize(record: dict, ledger_path: Path, dry_run: bool) -> int:
    if record["supersedes"]:
        print(f"Superseding prior GW{record['gameweek']} rank-aware prediction (prior record_id={record['supersedes'][:12]}...)")
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
    parser.add_argument("--dry-run", action="store_true", help="Do everything except append to the rank-aware-prediction ledger.")
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))
