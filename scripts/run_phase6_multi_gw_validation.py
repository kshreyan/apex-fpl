#!/usr/bin/env python3
"""Multi-gameweek validation of the Phase 6 joint simulator / BPS model
(docs/phase6_joint_simulation_report.md's "concrete next steps" #1 and
#2), following the exact precedent set by the CVaR multi-gameweek replay
(scripts/run_cvar_multi_gw_replay.py): 8 gameweeks per season (every 4th
from GW7), across the same 4 independent seasons used throughout this
project (2020-21, 2022-23, 2023-24, 2024-25 — 2021-22 stays excluded from
this test set as the designated tuning season, 2019-20 stays excluded for
its incompatible older schema).

Two questions, not one:

1. Does the single-gameweek finding replicate? (2022-23 GW20's demo found
   the joint simulator's per-player MAE was WORSE in aggregate (+0.064)
   but BETTER specifically for actual bonus-winners (-0.229) and worse for
   non-winners (+0.079) — a real signal masked by aggregation, or a lucky
   draw for that one gameweek's specific bonus-winners? 32 independent
   gameweek observations, block-bootstrapped, answers this properly.

2. Does fixing the walk-forward leakage flagged in the single-gameweek
   demo change anything? That demo fit the BPS model on ALL of 2022-23,
   including the target GW20 itself. Here the BPS model (and
   p_goal_assisted) are fit ONCE on 2021-22 only — a season that never
   overlaps with any of the 4 test seasons below, exactly the same
   train/test season separation this project already uses for the
   calibrator (docs/phase5_calibration_report.md) and the team-model
   tournament's inner-tune fold.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.models.bonus import bps_model
from apex_fpl.rules import scoring
from apex_fpl.simulation import joint_simulator as js
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase6_multi_gw_validation"
BPS_TRAIN_SEASON = "2021-22"  # tuning season only — never overlaps with any test season below
HELD_OUT_VALIDATION_SEASON = "2023-24"  # same held-out check as the single-GW demo, to confirm 1-season training doesn't degrade the model
TEST_SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GAMEWEEKS = [7, 11, 15, 19, 23, 27, 31, 35]
N_SCENARIOS_JOINT = 3000
SEED = 2026
N_BOOTSTRAP = 5000


def load_merged_gw(season: str) -> list[dict]:
    with open(REPO_ROOT / "data" / "external" / "vaastav" / season / "merged_gw.csv") as f:
        return list(csv.DictReader(f))


def run_one_gameweek(season: str, gw: int, bps_models, p_assisted: float) -> dict | None:
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None

    rules = scoring.load_scoring_rules("2026_27")
    baseline_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)

    joint_fixture_inputs = [js.JointFixtureInput(f.home_team, f.away_team, f.score_matrix) for f in data.fixture_inputs]
    joint_players = [
        js.JointPlayerInput(
            player_id=p.player_id, team=p.team, position=p.position, minutes_forecast=p.minutes_forecast,
            goal_share=data.shares[p.player_id].goal_share, assist_share=data.shares[p.player_id].assist_share,
        )
        for p in data.players_for_sim
    ]
    joint_results = js.simulate_gameweek_joint(
        joint_fixture_inputs, joint_players, rules, bps_models, n_scenarios=N_SCENARIOS_JOINT, p_goal_assisted=p_assisted, seed=SEED,
    )

    actual_by_player = {r["element"]: r for r in data.target_rows}
    winner_base, winner_joint, nonwinner_base, nonwinner_joint = [], [], [], []
    all_base, all_joint = [], []
    pred_bonus, actual_bonus = [], []
    for pid in data.candidates_meta:
        if pid not in baseline_results or pid not in joint_results or pid not in actual_by_player:
            continue
        actual = int(actual_by_player[pid]["total_points"])
        b_err = abs(baseline_results[pid].mean_points - actual)
        j_err = abs(joint_results[pid].mean_points - actual)
        all_base.append(b_err)
        all_joint.append(j_err)
        ab = int(actual_by_player[pid]["bonus"])
        pred_bonus.append(float(joint_results[pid].bonus_samples.mean()))
        actual_bonus.append(ab)
        if ab > 0:
            winner_base.append(b_err)
            winner_joint.append(j_err)
        else:
            nonwinner_base.append(b_err)
            nonwinner_joint.append(j_err)

    return {
        "season": season, "gw": gw, "n_players": len(all_base),
        "n_winners": len(winner_base), "n_nonwinners": len(nonwinner_base),
        "all_base_mae": float(np.mean(all_base)), "all_joint_mae": float(np.mean(all_joint)),
        "winner_base_mae": float(np.mean(winner_base)) if winner_base else None,
        "winner_joint_mae": float(np.mean(winner_joint)) if winner_joint else None,
        "nonwinner_base_mae": float(np.mean(nonwinner_base)), "nonwinner_joint_mae": float(np.mean(nonwinner_joint)),
        "pred_bonus": pred_bonus, "actual_bonus": actual_bonus,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 6 multi-gameweek validation: {len(TEST_SEASONS)} seasons x {len(GAMEWEEKS)} gameweeks ===\n")

    print(f"--- Fitting BPS model + p_goal_assisted on {BPS_TRAIN_SEASON} only (strictly walk-forward vs all 4 test seasons) ---")
    train_rows = load_merged_gw(BPS_TRAIN_SEASON)
    bps_models = bps_model.fit_bps_models(train_rows)
    p_assisted = js.fit_p_goal_assisted(train_rows)
    held_out_rows = load_merged_gw(HELD_OUT_VALIDATION_SEASON)
    eval_results = bps_model.evaluate_bps_models(bps_models, held_out_rows)
    for pos, r in eval_results.items():
        print(f"  {pos}: held-out ({HELD_OUT_VALIDATION_SEASON}) n={r['n']}  MAE={r['mae']}  R2={r['r2']}")
    print(f"  P(goal assisted), {BPS_TRAIN_SEASON}-only estimate: {p_assisted:.3f}\n")

    rows, blanks = [], []
    for season in TEST_SEASONS:
        for gw in GAMEWEEKS:
            result = run_one_gameweek(season, gw, bps_models, p_assisted)
            if result is None:
                print(f"{season} GW{gw}: blank gameweek, skipped")
                blanks.append((season, gw))
                continue
            rows.append(result)
            diff = result["all_joint_mae"] - result["all_base_mae"]
            wdiff = (result["winner_joint_mae"] - result["winner_base_mae"]) if result["winner_base_mae"] is not None else None
            wdiff_s = f"{wdiff:+.3f}" if wdiff is not None else "n/a"
            print(f"{season} GW{gw:2d}: n={result['n_players']:3d}  all_MAE diff={diff:+.3f}  "
                  f"winners(n={result['n_winners']:2d}) diff={wdiff_s}  nonwinners diff={result['nonwinner_joint_mae']-result['nonwinner_base_mae']:+.3f}")
            (ARTIFACT_DIR / "validation_results.json").write_text(json.dumps({"eval_results": eval_results, "p_goal_assisted": p_assisted, "rows": rows, "blanks": blanks}, indent=2))

    all_diffs = np.array([r["all_joint_mae"] - r["all_base_mae"] for r in rows])
    winner_rows = [r for r in rows if r["winner_base_mae"] is not None]
    winner_diffs = np.array([r["winner_joint_mae"] - r["winner_base_mae"] for r in winner_rows])
    nonwinner_diffs = np.array([r["nonwinner_joint_mae"] - r["nonwinner_base_mae"] for r in rows])

    all_pred_bonus = np.array([v for r in rows for v in r["pred_bonus"]])
    all_actual_bonus = np.array([v for r in rows for v in r["actual_bonus"]])
    pooled_corr = float(np.corrcoef(all_pred_bonus, all_actual_bonus)[0, 1])

    all_mean, all_lo, all_hi = bootstrap_ci(all_diffs)
    win_mean, win_lo, win_hi = bootstrap_ci(winner_diffs)
    non_mean, non_lo, non_hi = bootstrap_ci(nonwinner_diffs)

    summary = {
        "n_gameweeks": len(rows), "n_gameweeks_with_winners": len(winner_rows), "n_blank": len(blanks), "blanks": blanks,
        "bps_held_out_eval": eval_results, "p_goal_assisted": p_assisted,
        "pooled_corr_pred_vs_actual_bonus": pooled_corr,
        "all_players_mae_diff": {"mean": all_mean, "ci95": [all_lo, all_hi]},
        "bonus_winners_mae_diff": {"mean": win_mean, "ci95": [win_lo, win_hi], "n_gameweek_obs": len(winner_rows)},
        "non_winners_mae_diff": {"mean": non_mean, "ci95": [non_lo, non_hi]},
        "rows": rows,
    }
    (ARTIFACT_DIR / "validation_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({len(rows)} gameweeks, {len(blanks)} blank, {len(winner_rows)} gameweeks had >=1 bonus winner) ===")
    print(f"All-players MAE diff (joint - baseline):    {all_mean:+.3f}  95% CI [{all_lo:+.3f}, {all_hi:+.3f}]")
    print(f"Bonus-winners MAE diff:                     {win_mean:+.3f}  95% CI [{win_lo:+.3f}, {win_hi:+.3f}]  (n={len(winner_rows)} gameweek obs)")
    print(f"Non-winners MAE diff:                       {non_mean:+.3f}  95% CI [{non_lo:+.3f}, {non_hi:+.3f}]")
    print(f"Pooled correlation(predicted mean bonus, actual bonus): {pooled_corr:.3f}")
    print(f"\nWritten to {(ARTIFACT_DIR / 'validation_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
