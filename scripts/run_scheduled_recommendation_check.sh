#!/bin/bash
# launchd entry point for the Phase 12 production-recommendation job.
# Requires Full Disk Access granted to /bin/bash (System Settings ->
# Privacy & Security -> Full Disk Access) -- confirmed empirically that
# launchd cannot otherwise read data/canonical, data/external, or
# data/logs under this repo (same TCC restriction documented in
# ~/Library/Application Support/apex-fpl/capture_snapshot.sh). Without
# the grant this will fail with "Operation not permitted" in the log
# below, not silently.
set -euo pipefail

REPO="/Users/kshreyan12/Documents/FantasyPL/apex-fpl"
LOG_FILE="$REPO/data/logs/recommendation_check.log"

mkdir -p "$REPO/data/logs"
{
  cd "$REPO"
  PYTHONPATH="$REPO/src:$REPO/scripts" "$REPO/.venv/bin/python" "$REPO/scripts/run_scheduled_recommendation_check.py"
} >> "$LOG_FILE" 2>&1
