#!/usr/bin/env python3
"""Prediction scoring (Phase 13, Stage 4) — turns a settled gameweek's
already-frozen prediction into a scored, append-only result. Same
never-edited-never-deleted discipline as predict.py, extended to results
per the explicit ruling this was built from: results are the *grade*,
not the guess, and the incentive to quietly clean one up after the fact
is if anything higher than for a prediction — so this module gets the
same ledger treatment, not a lighter one.

**Read-only on data/predictions/**: every access to that directory in
this module is a plain read. Never enforced by a permissions trick
(fragile under concurrent access / different CI environments) — enforced
structurally, by never constructing a write-mode handle against that
path anywhere in this file, and checked directly in
tests/pipeline/test_score.py rather than just claimed here.

**Runs once per gameweek under normal operation.** "Has a prediction,
has no result yet, is SETTLED" is a one-time transition, not something
re-checked and re-scored on every daily run — once a result exists, the
default path does nothing further for that gameweek. Corrections (bonus
revised after the fact, a bug found in this module months later, a
genuine data correction) are a separate, deliberately-invoked path
(`--correct <gw> --reason <enum>`) — the automated daily path never
appends a correction on its own, or the whole point of an append-only
ledger — a correction being a rare, deliberate, reasoned act, not
routine — would be defeated.

**A known, stated limitation, not silently accepted.** "Actual points
this gameweek" is read from bootstrap-static's `elements[].event_points`,
which reflects the MOST RECENTLY SETTLED gameweek — correct in the
normal case (this fires within days of a gameweek settling, well before
the next one could also settle and overwrite it), but genuinely wrong if
the automated pipeline goes down across more than one full gameweek
cycle. The robust fix is capturing `event/{id}/live` (a per-gameweek-
durable endpoint this project has not integrated) — not attempted here,
flagged rather than silently risked.

**Top-10k baseline**: reads `data/raw/standings/gw{n}/extract.json` if
it exists (a separate, independent capture job — deferred to its own
workflow, not built as part of this module) and is `null` otherwise. Not
an error — the metric is honestly unavailable until that job exists and
has run for a given gameweek.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.entities import silver
from apex_fpl.optimization import squad as sq

from pipeline import fpl_client
from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
RESULTS_DIR = REPO_ROOT / "data" / "results"
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"
STANDINGS_DIR = REPO_ROOT / "data" / "raw" / "standings"

SCHEMA_VERSION = "1.0"
SUPERSEDE_REASONS = {"bonus_revised", "scoring_bug", "data_correction", "schema_migration"}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _content_hash(record_without_id: dict) -> str:
    return hashlib.sha256(json.dumps(record_without_id, sort_keys=True).encode()).hexdigest()


def _read_ledger_lines(path: Path) -> list[dict]:
    """Read-only: this is the ONLY function in this module that ever
    opens a prediction (or result) ledger file, and it always opens in
    read mode. No function in this module ever calls .write_text(),
    .open("w"/"a"), or similar against PREDICTIONS_DIR."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_ledger(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def compute_template_team(players: dict[str, dict]) -> dict:
    """The most-owned valid XI — NOT an official FPL concept, our own
    stated definition (see the Stage-1 design record this was built
    from): top-K most-owned players per position form a 15-player pool
    with NO budget constraint (a real template reflects accumulated
    squad value over a season, not a fresh draft, so budget-constraining
    it would be wrong), then the EXISTING, already-tested
    apex_fpl.optimization.squad.select_starting_xi picks the XI/captain
    from that pool — using ownership share as the "value" it optimizes,
    reusing real, tested formation-legality code rather than writing a
    parallel optimizer for a simpler problem."""
    by_position: dict[str, list[tuple[str, dict]]] = {}
    for pid, p in players.items():
        by_position.setdefault(p["position"], []).append((pid, p))
    for pos in by_position:
        by_position[pos].sort(key=lambda kv: -kv[1]["selected_by_percent"])

    pool_ids = []
    for pos, quota in sq.SQUAD_QUOTAS.items():
        pool_ids += [pid for pid, _ in by_position.get(pos, [])[:quota]]

    candidates = [sq.PlayerCandidate(pid, players[pid]["position"], players[pid]["team"], 0.0, players[pid]["selected_by_percent"]) for pid in pool_ids]
    xi = sq.select_starting_xi(candidates)
    return {"starting_xi": [p.player_id for p in xi.starters], "captain_player_id": xi.captain.player_id}


def _load_top_10k_average(target_gw: int) -> float | None:
    extract_path = STANDINGS_DIR / f"gw{target_gw:02d}" / "extract.json"
    if not extract_path.exists():
        return None
    entries = json.loads(extract_path.read_text())["entries"]
    return round(sum(e["event_total"] for e in entries) / len(entries), 2) if entries else None


def _score_call(call: dict, actual_points: dict[str, int], prediction: dict) -> dict:
    subject = call["subject"]
    if call["type"] == "points_forecast":
        if subject["kind"] == "player":
            actual = actual_points.get(subject["player_id"], 0)
        elif subject["kind"] == "captain":
            actual = actual_points.get(subject["player_id"], 0) * 2
        elif subject["kind"] == "squad_total":
            starters = prediction["squad"]["starting_xi"]
            captain_id = prediction["squad"]["captain_player_id"]
            actual = sum(actual_points.get(p["player_id"], 0) for p in starters) + actual_points.get(captain_id, 0)
        else:
            raise ValueError(f"unknown points_forecast subject kind: {subject['kind']!r}")
        return {"call_id": call["id"], "type": "points_forecast", "subject": subject, "predicted": call["value"], "actual": actual, "error": round(actual - call["value"], 3)}

    if call["type"] == "binary_probability":
        actual = actual_points.get(subject["player_id"], 0)  # undoubled -- the threshold is defined on the raw score, not the captained one
        outcome = actual >= subject["threshold"]
        return {"call_id": call["id"], "type": "binary_probability", "subject": subject, "predicted_probability": call["probability"], "outcome": outcome}

    raise ValueError(f"unknown call type: {call['type']!r}")


def score_gameweek(target_gw: int, correct_reason: str | None = None) -> dict | None:
    if correct_reason is not None and correct_reason not in SUPERSEDE_REASONS:
        raise ValueError(f"invalid correct_reason {correct_reason!r}; must be one of {sorted(SUPERSEDE_REASONS)}")

    prediction_path = PREDICTIONS_DIR / f"gw{target_gw:02d}.jsonl"
    prediction_lines = _read_ledger_lines(prediction_path)
    if not prediction_lines:
        print(f"GW{target_gw}: no prediction exists — nothing to score.")
        return None
    prediction = prediction_lines[-1]

    result_path = RESULTS_DIR / f"gw{target_gw:02d}.jsonl"
    prior_results = _read_ledger_lines(result_path)
    prior_result = prior_results[-1] if prior_results else None

    if prior_result is not None and correct_reason is None:
        print(f"GW{target_gw}: already scored (record_id={prior_result['record_id'][:12]}...) — normal run does not re-score. Use --correct to override.")
        return None
    if prior_result is None and correct_reason is not None:
        raise ValueError(f"--correct given but GW{target_gw} has never been scored — nothing to correct")

    if prediction["status"] == "BLANK_GAMEWEEK":
        print(f"GW{target_gw}: prediction was BLANK_GAMEWEEK — nothing to score, recording that plainly.")
        record = _build_blank_result(target_gw, prediction, prior_result, correct_reason)
        _append_ledger(result_path, record)
        print(f"Appended to {_display_path(result_path)} (record_id={record['record_id'][:12]}..., status={record['status']})")
        return record

    print(f"--- Fetching settlement data for GW{target_gw} ---")
    bs_path = fpl_client.fetch_and_validate("bootstrap_static", gameweek=target_gw)
    fx_path = fpl_client.fetch_and_validate("fixtures", gameweek=target_gw)
    bootstrap_static = json.loads(bs_path.read_bytes())
    fixtures = json.loads(fx_path.read_bytes())

    now = datetime.now(timezone.utc)
    phase_info = gs.gameweek_phase(bootstrap_static, fixtures, target_gw, now)
    if phase_info.phase != gs.Phase.SETTLED:
        print(f"GW{target_gw}: phase is {phase_info.phase.value}, not SETTLED — not ready to score yet.")
        return None

    print("--- Rebuilding Silver from freshly-captured settlement snapshots ---")
    silver.run_build(bronze_root=RAW_DATA_ROOT / f"gw{target_gw:02d}")

    actual_points = {str(el["id"]): el["event_points"] for el in bootstrap_static["elements"]}
    call_results = [_score_call(call, actual_points, prediction) for call in prediction["calls"]]

    starters = prediction["squad"]["starting_xi"]
    captain_id = prediction["squad"]["captain_player_id"]
    starting_xi_points = sum(actual_points.get(p["player_id"], 0) for p in starters)
    captain_actual_points = actual_points.get(captain_id, 0)
    squad_actual = {
        "starting_xi_points": starting_xi_points,
        "captain_actual_points": captain_actual_points,
        "total_points": starting_xi_points + captain_actual_points,
    }

    print("--- Computing baselines ---")
    average_manager_score = next(e["average_entry_score"] for e in bootstrap_static["events"] if e["id"] == target_gw)

    from apex_fpl.serving import live_data as ld
    players = ld.load_players()
    template = compute_template_team(players)
    template_starting_points = sum(actual_points.get(pid, 0) for pid in template["starting_xi"])
    template_captain_points = actual_points.get(template["captain_player_id"], 0)
    template_team_score = template_starting_points + template_captain_points

    top_10k_average = _load_top_10k_average(target_gw)

    settlement_data_sources = [
        {"source": "bootstrap_static", "raw_cache_path": _display_path(bs_path), "sha256": _hash_file(bs_path)},
        {"source": "fixtures", "raw_cache_path": _display_path(fx_path), "sha256": _hash_file(fx_path)},
    ]

    body = {
        "schema_version": SCHEMA_VERSION,
        "supersedes": prior_result["record_id"] if prior_result else None,
        "supersede_reason": correct_reason,
        "gameweek": target_gw,
        "status": "SCORED",
        "prediction_ref": {
            "record_id": prediction["record_id"], "path": _display_path(prediction_path),
            "line_number": len(prediction_lines), "total_prediction_lines": len(prediction_lines), "had_revisions": len(prediction_lines) > 1,
        },
        "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_checked_confirmed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settlement_data_sources": settlement_data_sources,
        "call_results": call_results,
        "squad_actual": squad_actual,
        "baselines": {
            "average_manager_score": average_manager_score,
            "template_team_score": template_team_score,
            "top_10k_average": top_10k_average,
        },
        "baseline_notes": {
            "average_manager_score": "bootstrap-static events[].average_entry_score",
            "template_team_score": "most-owned valid XI by position (players.csv selected_by_percent), captain = most-owned player in it -- our own definition, not an official FPL concept",
            "top_10k_average": "computed from data/raw/standings/gw{n}/extract.json if that capture has run for this gameweek; null otherwise" if top_10k_average is None else "computed from data/raw/standings/gw{n}/extract.json",
        },
    }
    record = {**body, "record_id": _content_hash(body)}

    if record["supersedes"]:
        print(f"Superseding prior GW{target_gw} result (prior record_id={record['supersedes'][:12]}..., reason={correct_reason})")

    _append_ledger(result_path, record)
    print(f"Appended to {_display_path(result_path)} (record_id={record['record_id'][:12]}..., status={record['status']})")
    return record


def _build_blank_result(target_gw: int, prediction: dict, prior_result: dict | None, correct_reason: str | None) -> dict:
    body = {
        "schema_version": SCHEMA_VERSION,
        "supersedes": prior_result["record_id"] if prior_result else None,
        "supersede_reason": correct_reason,
        "gameweek": target_gw,
        "status": "BLANK_GAMEWEEK_NO_SCORING",
        "prediction_ref": {"record_id": prediction["record_id"], "path": _display_path(PREDICTIONS_DIR / f"gw{target_gw:02d}.jsonl"), "line_number": None, "total_prediction_lines": None, "had_revisions": None},
        "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_checked_confirmed_at_utc": None,
        "settlement_data_sources": [],
        "call_results": [],
        "squad_actual": None,
        "baselines": {"average_manager_score": None, "template_team_score": None, "top_10k_average": None},
        "baseline_notes": {},
    }
    return {**body, "record_id": _content_hash(body)}


def run(target_gws: list[int] | None = None, correct_reason: str | None = None) -> int:
    if target_gws is None:
        if not PREDICTIONS_DIR.exists():
            print("No predictions directory exists yet — nothing to score.")
            return 0
        target_gws = sorted(int(p.stem[2:]) for p in PREDICTIONS_DIR.glob("gw*.jsonl"))

    for gw in target_gws:
        score_gameweek(gw, correct_reason=correct_reason)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct", type=int, metavar="GW", help="Force a re-score of GW, appending a correction (requires --reason).")
    parser.add_argument("--reason", choices=sorted(SUPERSEDE_REASONS), help="Required with --correct.")
    args = parser.parse_args()

    if args.correct is not None:
        if args.reason is None:
            parser.error("--correct requires --reason")
        sys.exit(0 if score_gameweek(args.correct, correct_reason=args.reason) is not None else 1)

    sys.exit(run())
