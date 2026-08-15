#!/usr/bin/env python3
"""The launchd-scheduled entry point (Phase 12 addendum): runs daily,
brings Silver up to date from whatever new Bronze snapshots exist, checks
whether any upcoming gameweek deadline is close enough to be worth a
fresh recommendation, and generates one if so — idempotently (never
regenerates the same freeze twice in the same run) but WILL regenerate on
a later day if the deadline is still upcoming, deliberately: the whole
point of running this daily is to use the freshest data as the deadline
approaches, not to freeze once and go stale.

Requires macOS Full Disk Access granted to whatever binary launchd
invokes this through (this project's plists use /bin/bash) — confirmed
empirically (not assumed) that launchd cannot otherwise read anything
under data/canonical, data/external, or data/logs in this repo (the same
TCC restriction documented in
~/Library/Application Support/apex-fpl/capture_snapshot.sh, which is why
THAT job runs entirely outside ~/Documents instead). Without the grant,
this script's Silver read/write step will fail with "Operation not
permitted" — a real, externally-verified precondition, not a
hypothetical one.

DEADLINE_WINDOW_HOURS=48 is a deliberately generous window: since real
FPL deadlines fall on varying days (not a fixed weekly slot), and this
job runs once daily, 48 hours guarantees at least one (usually two)
daily runs land inside the window before any deadline, regardless of
which day of the week it falls on.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "production_recommendations"
DEADLINE_WINDOW_HOURS = 48


def already_generated_today(target_gw: int) -> bool:
    """True if a recommendation for this gameweek was already frozen
    today (UTC) — avoids generating a second, near-identical artifact
    within the same run/day; a fresh one is still produced on a LATER
    day if the deadline is still upcoming (see module docstring)."""
    if not ARTIFACT_DIR.exists():
        return False
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prefix = f"gw{target_gw:02d}_recommendation_{today}"
    return any(p.name.startswith(prefix) for p in ARTIFACT_DIR.glob(f"gw{target_gw:02d}_recommendation_*.json"))


def main() -> None:
    from apex_fpl.entities import silver
    from apex_fpl.serving import live_data as ld

    print(f"===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} =====")

    print("--- Rebuilding Silver from any new Bronze snapshots ---")
    counts = silver.run_build()
    print(f"Silver rows appended: {counts or 'none (already up to date)'}")

    gameweeks = ld.load_gameweeks()
    upcoming = [g for eid, g in gameweeks.items() if not g["finished"]]
    if not upcoming:
        print("No upcoming (unfinished) gameweeks found — nothing to do.")
        return

    now = datetime.now(timezone.utc)
    next_gw_id, next_gw = min(
        ((eid, g) for eid, g in gameweeks.items() if not g["finished"]),
        key=lambda kv: kv[1]["deadline_time"].replace(tzinfo=timezone.utc),
    )
    deadline = next_gw["deadline_time"].replace(tzinfo=timezone.utc)
    hours_until = (deadline - now).total_seconds() / 3600
    print(f"Next deadline: GW{next_gw_id} ({next_gw['name']}) at {deadline.isoformat()} ({hours_until:.1f}h from now)")

    if hours_until < 0:
        print("Deadline already passed but gameweek not yet marked finished — skipping (data may be stale).")
        return
    if hours_until > DEADLINE_WINDOW_HOURS:
        print(f"More than {DEADLINE_WINDOW_HOURS}h away — not yet due, skipping.")
        return
    if already_generated_today(next_gw_id):
        print(f"Already generated a GW{next_gw_id} recommendation today — skipping.")
        return

    print(f"--- Deadline within {DEADLINE_WINDOW_HOURS}h window: generating GW{next_gw_id} recommendation ---")
    from run_production_recommendation import generate_recommendation

    generate_recommendation(next_gw_id)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    main()
