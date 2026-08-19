#!/usr/bin/env python3
"""Execution-divergence check (Phase 13 -- the second half of CLAUDE.md's
"real FPL entry: execution is human-in-the-loop" rule). The squad-state
reader (apex_fpl.serving.entry_state) has existed since Block 2.5, used
for transfer/chip decisions; this is the piece CLAUDE.md explicitly
named as still missing: comparing the real entry's ACTUAL picks for a
settled gameweek against what predict.py published for that same
gameweek, and recording whether they matched.

**Once checked, a gameweek is NEVER re-checked** — same append-only-
ledger discipline as every other ledger in this project, applied here
specifically because CLAUDE.md's rule requires it: "not corrected, not
backdated, not silently reconciled." If this script runs before the
real entry's picks are actually visible (a genuine settlement-detection
race, not expected in normal operation), it simply doesn't write a
record for that gameweek yet and tries again on the next run — it does
NOT record a false "matched" or invent a placeholder.

**Only meaningful for gameweeks with a PUBLISHED prediction.** A
BLANK_GAMEWEEK prediction never recommended a squad in the first place,
so there's nothing for the real entry to have diverged from — skipped,
not reported as any kind of failure.

Real picks are compared as a SET of 15 player IDs (starting XI + bench
combined) plus the captain ID specifically — bench ORDER and starting-
XI formation are not compared, since a manager reordering an unchanged
15-man squad or bench is not what this rule is protecting against
(CLAUDE.md's concern is "the manual entry step didn't happen in time,"
i.e. a genuinely different squad or a missed pick, not a legal
within-squad reshuffle).
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
LEDGER_DIR = REPO_ROOT / "data" / "execution_divergence"

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


def check_gameweek(target_gw: int, bootstrap_static: dict, fixtures: dict, now: datetime, entry_id: int = es.ENTRY_ID) -> dict | None:
    """Returns the appended record, or None if there was nothing to
    check (or nothing NEW to check) this run -- None is a normal,
    common return value, not an error."""
    prediction_lines = _read_ledger_lines(PREDICTIONS_DIR / f"gw{target_gw:02d}.jsonl")
    if not prediction_lines:
        return None
    prediction = prediction_lines[-1]
    if prediction["status"] != "PUBLISHED":
        return None  # BLANK_GAMEWEEK -- no squad was ever recommended

    ledger_path = LEDGER_DIR / f"gw{target_gw:02d}.jsonl"
    if _read_ledger_lines(ledger_path):
        return None  # already checked once -- never re-checked, see module docstring

    phase_info = gs.gameweek_phase(bootstrap_static, fixtures, target_gw, now)
    if phase_info.phase != gs.Phase.SETTLED:
        return None  # not settled yet -- real picks aren't knowable as final

    real_picks = es.fetch_entry_picks(target_gw, entry_id=entry_id)
    if real_picks is None:
        return None  # bootstrap-static says settled, but the entry's own picks endpoint disagrees -- a real, rare inconsistency; wait and retry rather than guess

    lineup = es.parse_gameweek_lineup(real_picks, target_gw)
    real_squad_ids = sorted(lineup.squad_ids)
    predicted_squad_ids = sorted({p["player_id"] for p in prediction["squad"]["starting_xi"]} | {p["player_id"] for p in prediction["squad"]["bench_order"]})

    squad_diverged = set(real_squad_ids) != set(predicted_squad_ids)
    captain_diverged = lineup.captain_id != prediction["squad"]["captain_player_id"]
    status = "DIVERGED" if (squad_diverged or captain_diverged) else "MATCHED"

    body = {
        "schema_version": SCHEMA_VERSION,
        "gameweek": target_gw,
        "entry_id": entry_id,
        "status": status,
        "squad_diverged": squad_diverged,
        "captain_diverged": captain_diverged,
        "real_squad_ids": real_squad_ids,
        "predicted_squad_ids": predicted_squad_ids,
        "real_captain_id": lineup.captain_id,
        "predicted_captain_id": prediction["squad"]["captain_player_id"],
        "prediction_record_id": prediction["record_id"],
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    record = {**body, "record_id": _content_hash(body)}
    _append_ledger(ledger_path, record)
    return record


def run() -> int:
    print(f"===== check_execution_divergence.py run: {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")
    if not PREDICTIONS_DIR.exists():
        print("No predictions directory exists yet — nothing to check.")
        return 0

    bootstrap_static = json.loads(bronze.fetch_raw("bootstrap_static")[0])
    fixtures = json.loads(bronze.fetch_raw("fixtures")[0])
    now = datetime.now(timezone.utc)

    target_gws = sorted(int(p.stem[2:]) for p in PREDICTIONS_DIR.glob("gw*.jsonl"))
    for gw in target_gws:
        record = check_gameweek(gw, bootstrap_static, fixtures, now)
        if record is not None:
            print(f"GW{gw}: {record['status']}" + (f" (squad_diverged={record['squad_diverged']}, captain_diverged={record['captain_diverged']})" if record["status"] == "DIVERGED" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(run())
