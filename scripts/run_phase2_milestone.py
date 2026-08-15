#!/usr/bin/env python3
"""Phase 2 baseline milestone (spec Part LXVI) — thin driver around the
reusable apex_fpl.backtesting.replay.run_gameweek(), which now also backs
the Phase 3 replay loop (scripts/run_phase3_replay.py). See
docs/phase2_milestone_report.md for the full writeup of this specific run.
"""
from __future__ import annotations

from pathlib import Path

from apex_fpl.backtesting.replay import run_gameweek

REPO_ROOT = Path(__file__).resolve().parents[1]
SEASON = "2022-23"
TARGET_GW = 20


def main() -> None:
    print(f"=== Phase 2 milestone: {SEASON} GW{TARGET_GW} ===\n")
    result = run_gameweek(
        SEASON, TARGET_GW,
        artifact_dir=REPO_ROOT / "artifacts" / "phase2_milestone",
        verbose=True,
    )
    rec, ev = result.recommendation, result.evaluation

    print("\nTeam model projections:")
    for f in rec["fixture_projections"]:
        print(f"  {f['home_team']:<20} {f['expected_home_goals']:.2f} - {f['expected_away_goals']:.2f} {f['away_team']}")

    print(f"\nOptimal squad: {len(rec['squad'])} players, "
          f"cost £{sum(p['price'] for p in rec['squad']):.1f}m, "
          f"projected squad EP={sum(p['expected_points'] for p in rec['squad']):.2f}")

    print(f"\nFROZEN recommendation: {result.recommendation_path.relative_to(REPO_ROOT)}")
    print(f"Artifact hash: {ev['artifact_hash']}")

    print("\n=== Revealing actual GW results (evaluation only, recommendation already frozen) ===")
    print(f"Projected GW points (frozen): {rec['projected_gw_points']}")
    print(f"Realized GW points (actual, no auto-subs applied): {ev['model_squad_realized_points']}")

    squad_by_id = {p["player_id"]: p for p in rec["squad"]}
    print("\nPer-starter (name, projected):")
    for pid in rec["starting_xi"]:
        p = squad_by_id[pid]
        cap_tag = " (C)" if pid == rec["captain"] else ""
        print(f"  {p['name']:<22} proj={p['expected_points']:5.2f}{cap_tag}")

    print(f"\n=== Result ===")
    print(f"Model-driven squad realized: {ev['model_squad_realized_points']}")
    print(f"Recent-form baseline realized: {ev['baseline_recent_form_realized_points']}")
    print(f"Difference: {ev['difference']:+d}")
    print(f"\nEvaluation written to {result.evaluation_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
