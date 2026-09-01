#!/usr/bin/env python3
"""Execution-divergence check (Phase 13 -- the second half of CLAUDE.md's
"real FPL entry: execution is human-in-the-loop" rule). The squad-state
reader (apex_fpl.serving.entry_state) has existed since Block 2.5, used
for transfer/chip decisions; this module compares the real entry's
ACTUAL picks for a settled gameweek against what the entry SHOULD hold
given this project's own advice -- and records whether they matched.

**Schema 1.1 correction (found auditing GW1/GW2): the reference squad
was wrong.** Schema 1.0 compared the real squad against predict.py's
own from-scratch squad for EVERY gameweek. That's fine for GW1 (there's
no prior real squad to diff against), but wrong from GW2 onward:
predict.py recomputes a squad every week completely unconstrained by
the real entry's actual budget, prior squad value, or free-transfer
count -- exactly what's disclosed everywhere else on the site ("not a
transfer plan... will not tell you who to sell or buy"). Comparing a
real, budget-and-history-constrained squad against that unconstrained
ideal will diverge most weeks regardless of whether the real entry
followed every recommendation -- confirmed live: predicted/real overlap
fell from 12/15 at GW1 to 7/15 at GW2, exactly the drift this
comparison would produce even under perfect manual execution, making
the old "DIVERGED" signal unable to ever distinguish a real execution
failure from business as usual.

The correct reference for GW >= 2 is: the real entry's OWN prior-
gameweek squad, with predict_transfers.py's PUBLISHED recommendation
for THIS gameweek applied (transfers_out removed, transfers_in added)
-- the actual, disclosed decision a manager following this project's
advice is meant to make. GW1 keeps predict.py's own squad as the
reference, unchanged, since no prior real squad exists yet.

If a prior real squad exists but no PUBLISHED transfer recommendation
does (the recommender hasn't run yet, or found an inconsistent state),
the check is skipped rather than guessed -- see check_gameweek's own
docstring.

Captain comparison is unchanged (still against predict.py's own
choice) -- a real, disclosed gap: no live artifact currently computes
"best captain among the real held squad" the way predict_transfers.py
computes transfers. Flagged, not silently pretended solved.

**Once checked, a gameweek is NEVER re-checked under normal
operation** -- same append-only-ledger discipline as every other
ledger in this project, applied here specifically because CLAUDE.md's
rule requires it: "not corrected, not backdated, not silently
reconciled." A methodology bug in HOW this project computes divergence
is a different kind of fact than a manager's own missed deadline,
though -- exactly like score.py's own `--correct` escape hatch for "a
bug was found in this module months later," this module has the same
explicit, deliberately-invoked correction path
(`--correct <gw> --reason <enum>`), never taken by the automated daily
run on its own.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.data import bronze
from apex_fpl.serving import entry_state as es

from pipeline import gw_state as gs

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
TRANSFER_RECOMMENDATIONS_DIR = REPO_ROOT / "data" / "transfer_recommendations"
LEDGER_DIR = REPO_ROOT / "data" / "execution_divergence"

SCHEMA_VERSION = "1.1"
SUPERSEDE_REASONS = {"comparison_methodology_bug", "data_correction", "schema_migration"}


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


def _expected_squad_from_prior_and_transfer(prior_squad_ids: set[str], target_gw: int) -> tuple[set[str], dict] | None:
    """Prior real squad + this gameweek's PUBLISHED transfer
    recommendation applied. Returns (expected_ids, transfer_record), or
    None if no PUBLISHED recommendation exists for this gameweek -- the
    caller must skip rather than guess in that case."""
    lines = _read_ledger_lines(TRANSFER_RECOMMENDATIONS_DIR / f"gw{target_gw:02d}.jsonl")
    if not lines:
        return None
    transfer_record = lines[-1]
    if transfer_record["status"] != "PUBLISHED":
        return None
    rec = transfer_record["recommendation"]
    out_ids = {p["player_id"] for p in rec["transfers_out"]}
    in_ids = {p["player_id"] for p in rec["transfers_in"]}
    return (prior_squad_ids - out_ids) | in_ids, transfer_record


def check_gameweek(target_gw: int, bootstrap_static: dict, fixtures: dict, now: datetime, entry_id: int = es.ENTRY_ID, correct_reason: str | None = None) -> dict | None:
    """Returns the appended record, or None if there was nothing to
    check (or nothing NEW to check) this run -- None is a normal,
    common return value, not an error."""
    if correct_reason is not None and correct_reason not in SUPERSEDE_REASONS:
        raise ValueError(f"invalid correct_reason {correct_reason!r}; must be one of {sorted(SUPERSEDE_REASONS)}")

    prediction_lines = _read_ledger_lines(PREDICTIONS_DIR / f"gw{target_gw:02d}.jsonl")
    if not prediction_lines:
        return None
    prediction = prediction_lines[-1]
    if prediction["status"] != "PUBLISHED":
        return None  # BLANK_GAMEWEEK -- no squad was ever recommended

    ledger_path = LEDGER_DIR / f"gw{target_gw:02d}.jsonl"
    prior_records = _read_ledger_lines(ledger_path)
    if prior_records and correct_reason is None:
        return None  # already checked once -- never re-checked outside --correct, see module docstring
    if not prior_records and correct_reason is not None:
        raise ValueError(f"--correct given but GW{target_gw} has never been checked -- nothing to correct")
    prior_record = prior_records[-1] if prior_records else None

    phase_info = gs.gameweek_phase(bootstrap_static, fixtures, target_gw, now)
    if phase_info.phase != gs.Phase.SETTLED:
        return None  # not settled yet -- real picks aren't knowable as final

    real_picks = es.fetch_entry_picks(target_gw, entry_id=entry_id)
    if real_picks is None:
        return None  # bootstrap-static says settled, but the entry's own picks endpoint disagrees -- a real, rare inconsistency; wait and retry rather than guess

    lineup = es.parse_gameweek_lineup(real_picks, target_gw)
    real_squad_ids = sorted(lineup.squad_ids)

    comparison_basis = None
    transfer_record_id = None
    expected_ids: set[str] = set()
    if target_gw > 1:
        prior_picks = es.fetch_entry_picks(target_gw - 1, entry_id=entry_id)
        if prior_picks is not None:
            prior_lineup = es.parse_gameweek_lineup(prior_picks, target_gw - 1)
            resolved = _expected_squad_from_prior_and_transfer(set(prior_lineup.squad_ids), target_gw)
            if resolved is not None:
                expected_ids, transfer_record = resolved
                comparison_basis = "prior_squad_plus_recommended_transfer"
                transfer_record_id = transfer_record["record_id"]

    if comparison_basis is None:
        if target_gw > 1:
            return None  # a prior real squad exists but no PUBLISHED transfer recommendation (or prior picks unavailable) -- can't confidently determine the expected squad; wait, don't guess
        expected_ids = {p["player_id"] for p in prediction["squad"]["starting_xi"]} | {p["player_id"] for p in prediction["squad"]["bench_order"]}
        comparison_basis = "no_prior_squad_used_from_scratch_prediction"

    expected_squad_ids = sorted(expected_ids)
    squad_diverged = set(real_squad_ids) != set(expected_squad_ids)
    captain_diverged = lineup.captain_id != prediction["squad"]["captain_player_id"]
    status = "DIVERGED" if (squad_diverged or captain_diverged) else "MATCHED"

    body = {
        "schema_version": SCHEMA_VERSION,
        "supersedes": prior_record["record_id"] if prior_record else None,
        "supersede_reason": correct_reason,
        "gameweek": target_gw,
        "entry_id": entry_id,
        "status": status,
        "comparison_basis": comparison_basis,
        "squad_diverged": squad_diverged,
        "captain_diverged": captain_diverged,
        "real_squad_ids": real_squad_ids,
        "expected_squad_ids": expected_squad_ids,
        "real_captain_id": lineup.captain_id,
        "predicted_captain_id": prediction["squad"]["captain_player_id"],
        "prediction_record_id": prediction["record_id"],
        "transfer_recommendation_record_id": transfer_record_id,
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    record = {**body, "record_id": _content_hash(body)}
    _append_ledger(ledger_path, record)
    return record


def run(correct_reason: str | None = None, target_gws: list[int] | None = None) -> int:
    print(f"===== check_execution_divergence.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")
    if not PREDICTIONS_DIR.exists():
        print("No predictions directory exists yet — nothing to check.")
        return 0

    bootstrap_static = json.loads(bronze.fetch_raw("bootstrap_static")[0])
    fixtures = json.loads(bronze.fetch_raw("fixtures")[0])
    now = datetime.now(timezone.utc)

    if target_gws is None:
        target_gws = sorted(int(p.stem[2:]) for p in PREDICTIONS_DIR.glob("gw*.jsonl"))
    for gw in target_gws:
        record = check_gameweek(gw, bootstrap_static, fixtures, now, correct_reason=correct_reason)
        if record is not None:
            detail = f"basis={record['comparison_basis']}"
            if record["status"] == "DIVERGED":
                detail = f"squad_diverged={record['squad_diverged']}, captain_diverged={record['captain_diverged']}, {detail}"
            print(f"GW{gw}: {record['status']} ({detail})")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct", type=int, metavar="GW", help="Force a re-check of GW, appending a correction (requires --reason).")
    parser.add_argument("--reason", choices=sorted(SUPERSEDE_REASONS), help="Required with --correct.")
    args = parser.parse_args()
    if args.correct is not None:
        if args.reason is None:
            parser.error("--correct requires --reason")
        sys.exit(run(correct_reason=args.reason, target_gws=[args.correct]))
    else:
        sys.exit(run())
