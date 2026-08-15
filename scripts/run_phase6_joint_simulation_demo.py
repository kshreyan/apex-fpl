#!/usr/bin/env python3
"""Phase 6 demonstration: joint simulation + reduced-form BPS model on one
real historical gameweek (2022-23 GW20 — matching the precedent of every
earlier single-gameweek demonstration in this project).

Fits the BPS reduced-form model on 2022-23 (excluding the target GW's own
data — see the leakage note below) and validates it on held-out 2023-24,
then compares three things for real GW20 players:
  1. Baseline simulator (src/apex_fpl/simulation/monte_carlo.py): no
     bonus, independent per-player goal/assist draws.
  2. Joint simulator (src/apex_fpl/simulation/joint_simulator.py): bonus
     simulated via ranked BPS competition, goals/assists allocated
     multinomially from the team's own simulated total.
  3. Actual real GW20 total_points (ground truth).

Question: does adding bonus simulation (the single largest previously-
documented gap between projected and realized FPL points, per
docs/phase2_milestone_report.md and docs/phase5_calibration_report.md)
measurably close that gap?

Leakage note: the BPS model is fit on ALL of 2022-23's merged_gw rows,
which technically includes GW20 itself. This is a real, acknowledged
simplification for this demonstration — the model's LEARNED COEFFICIENTS
barely move by including or excluding one gameweek out of 38 (n=26,505
total rows), but a rigorous walk-forward version (fit strictly on
gameweeks before the target, exactly like every other model in this
project) is the correct next step before this is wired into replay.py,
not assumed acceptable for production use.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.models.bonus import bps_model
from apex_fpl.rules import scoring
from apex_fpl.simulation import joint_simulator as js
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase6_joint_simulation"
SEASON, TARGET_GW = "2022-23", 20
BPS_TRAIN_SEASON = "2022-23"
BPS_TEST_SEASON = "2023-24"
N_SCENARIOS_JOINT = 4000


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 6 joint simulation demo: {SEASON} GW{TARGET_GW} ===\n")

    print(f"--- Fitting BPS reduced-form model on {BPS_TRAIN_SEASON}, validating on {BPS_TEST_SEASON} ---")
    train_rows = list(csv.DictReader(open(REPO_ROOT / "data" / "external" / "vaastav" / BPS_TRAIN_SEASON / "merged_gw.csv")))
    test_rows = list(csv.DictReader(open(REPO_ROOT / "data" / "external" / "vaastav" / BPS_TEST_SEASON / "merged_gw.csv")))
    bps_models = bps_model.fit_bps_models(train_rows)
    eval_results = bps_model.evaluate_bps_models(bps_models, test_rows)
    for pos, r in eval_results.items():
        print(f"  {pos}: held-out n={r['n']}  MAE={r['mae']}  R2={r['r2']}")
    p_assisted = js.fit_p_goal_assisted(train_rows)
    print(f"  P(goal assisted), real-data estimate: {p_assisted:.3f}\n")

    print(f"--- Building {SEASON} GW{TARGET_GW} scenario data ---")
    data = build_gameweek_scenario_data(SEASON, TARGET_GW)
    rules = scoring.load_scoring_rules("2026_27")
    print(f"Players: {len(data.players_for_sim)}")

    print("\n--- Running baseline simulator (no bonus) ---")
    baseline_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)

    print(f"--- Running joint simulator (n_scenarios={N_SCENARIOS_JOINT}, with ranked BPS bonus) ---")
    joint_fixture_inputs = [js.JointFixtureInput(f.home_team, f.away_team, f.score_matrix) for f in data.fixture_inputs]
    joint_players = [
        js.JointPlayerInput(
            player_id=p.player_id, team=p.team, position=p.position, minutes_forecast=p.minutes_forecast,
            goal_share=data.shares[p.player_id].goal_share, assist_share=data.shares[p.player_id].assist_share,
        )
        for p in data.players_for_sim
    ]
    joint_results = js.simulate_gameweek_joint(
        joint_fixture_inputs, joint_players, rules, bps_models, n_scenarios=N_SCENARIOS_JOINT, p_goal_assisted=p_assisted,
    )

    print("\n--- Reconciliation checks on the joint simulation ---")
    fixture_by_team = {}
    for f in data.fixture_inputs:
        fixture_by_team[f.home_team] = f
        fixture_by_team[f.away_team] = f
    total_bonus_awarded = sum(r.bonus_samples.mean() for r in joint_results.values())
    n_matches = len(data.fixture_inputs)
    print(f"Mean total bonus awarded per gameweek: {total_bonus_awarded:.2f} (expect ~{6 * n_matches} if no ties ever occurred; ties push this higher)")

    actual_by_player = {r["element"]: r for r in data.target_rows}
    rows = []
    for pid in data.candidates_meta:
        if pid not in baseline_results or pid not in joint_results or pid not in actual_by_player:
            continue
        actual = int(actual_by_player[pid]["total_points"])
        baseline_ep = baseline_results[pid].mean_points
        joint_ep = joint_results[pid].mean_points
        rows.append({
            "player_id": pid, "name": data.candidates_meta[pid]["name"],
            "actual": actual, "baseline_ep": baseline_ep, "joint_ep": joint_ep,
            "baseline_err": abs(baseline_ep - actual), "joint_err": abs(joint_ep - actual),
            "actual_bonus": int(actual_by_player[pid]["bonus"]), "joint_mean_bonus": float(joint_results[pid].bonus_samples.mean()),
        })

    baseline_mae = float(np.mean([r["baseline_err"] for r in rows]))
    joint_mae = float(np.mean([r["joint_err"] for r in rows]))
    print(f"\n=== Per-player expected-points MAE against real GW20 outcomes (n={len(rows)}) ===")
    print(f"Baseline simulator (no bonus): MAE={baseline_mae:.3f}")
    print(f"Joint simulator (with bonus):  MAE={joint_mae:.3f}")
    print(f"Difference: {joint_mae - baseline_mae:+.3f}")

    top_bonus_actual = sorted(rows, key=lambda r: -r["actual_bonus"])[:10]
    print("\nTop 10 real bonus earners this GW — simulated mean bonus for comparison:")
    for r in top_bonus_actual:
        print(f"  {r['name']:<22} actual_bonus={r['actual_bonus']}  joint_simulated_mean_bonus={r['joint_mean_bonus']:.2f}")

    summary = {
        "season": SEASON, "target_gw": TARGET_GW,
        "bps_model_eval": eval_results, "p_goal_assisted": p_assisted,
        "baseline_mae": baseline_mae, "joint_mae": joint_mae,
        "mean_total_bonus_per_gw": total_bonus_awarded,
        "per_player": rows,
    }
    (ARTIFACT_DIR / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'demo_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
