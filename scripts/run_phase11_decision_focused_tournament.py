#!/usr/bin/env python3
"""Phase 11: Track A (prediction-focused, unmodified) vs Track B
(decision-focused: EP shrinkage TUNED to directly maximize realized
squad points, not prediction accuracy) vs Track C (a simple hybrid
average of the two), evaluated on held-out decision regret — spec Part
XXVII's own framing, and research_plan.md's Phase 11 entry.

Follows the exact tune/test season split already established throughout
this project: shrinkage is tuned on 2021-22 (the project's standing
tuning-only season, held out from every decision-level test everywhere
else — docs/phase4b_tournament_report.md onward) across 8 gameweeks
(every 4th from GW7), then evaluated on the SAME 4 independent seasons
used for the CVaR/MAD tournament (2020-21, 2022-23, 2023-24, 2024-25),
32 gameweek observations, block-bootstrapped.

"Without sacrificing calibration" (spec Part XXVII) is checked with a
simple, honest proxy for a continuous (non-probability) quantity: the
aggregate ratio of each track's predicted EP to the ACTUAL realized
points for the players it actually selects as starters, summed across
all 32 test gameweeks — a real, if simple, bias check, not the full
log-loss/Brier machinery built for binary probabilities in Phase 5
(which doesn't directly apply to a continuous points prediction).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.models import decision_focused as df
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase11_decision_focused_tournament"
TUNE_SEASON = "2021-22"
TEST_SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GAMEWEEKS = [7, 11, 15, 19, 23, 27, 31, 35]
SHRINKAGE_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
SEED = 2026
N_BOOTSTRAP = 5000


def build_gw_data(season: str, gw: int, rules: dict):
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None
    sim_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    ep_by_id = {pid: sim_results[pid].mean_points for pid in sim_results}
    actual_by_player = {r["element"]: int(r["total_points"]) for r in data.target_rows}
    return data.candidates_meta, ep_by_id, actual_by_player


def realized_points(xi, actual_by_player: dict[str, int]) -> int:
    return sum(actual_by_player.get(p.player_id, 0) for p in xi.starters) + actual_by_player.get(xi.captain.player_id, 0)


def select_and_score(candidates_meta: dict, ep_by_id: dict[str, float], actual_by_player: dict[str, int]):
    candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], ep_by_id[pid]) for pid, m in candidates_meta.items() if pid in ep_by_id]
    squad = sq.select_squad(candidates)
    xi = sq.select_starting_xi(squad)
    starter_ep = sum(ep_by_id[p.player_id] for p in xi.starters) + ep_by_id[xi.captain.player_id]
    starter_actual = realized_points(xi, actual_by_player)
    return starter_actual, starter_ep


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rules = scoring.load_scoring_rules("2026_27")
    print(f"=== Phase 11: decision-focused tournament (tune={TUNE_SEASON}, test={TEST_SEASONS}) ===\n")

    print(f"--- Tuning shrinkage on {TUNE_SEASON} ---")
    tuning_gameweeks = []
    for gw in GAMEWEEKS:
        result = build_gw_data(TUNE_SEASON, gw, rules)
        if result is None:
            print(f"{TUNE_SEASON} GW{gw}: blank gameweek, skipped")
            continue
        tuning_gameweeks.append(result)
    chosen_shrinkage = df.tune_shrinkage(tuning_gameweeks, SHRINKAGE_GRID, sq.select_squad, sq.select_starting_xi, sq.PlayerCandidate)
    print(f"Chosen shrinkage: {chosen_shrinkage}\n")

    print("--- Evaluating Track A/B/C on held-out test seasons ---")
    rows, blanks = [], []
    for season in TEST_SEASONS:
        for gw in GAMEWEEKS:
            result = build_gw_data(season, gw, rules)
            if result is None:
                print(f"{season} GW{gw}: blank gameweek, skipped")
                blanks.append((season, gw))
                continue
            candidates_meta, ep_by_id, actual_by_player = result
            position_by_id = {pid: m["position"] for pid, m in candidates_meta.items()}

            a_actual, a_ep = select_and_score(candidates_meta, ep_by_id, actual_by_player)
            adjusted_ep = df.apply_shrinkage(ep_by_id, position_by_id, chosen_shrinkage)
            b_actual, b_ep = select_and_score(candidates_meta, adjusted_ep, actual_by_player)
            hybrid = df.hybrid_ep(ep_by_id, adjusted_ep, weight_b=0.5)
            c_actual, c_ep = select_and_score(candidates_meta, hybrid, actual_by_player)

            rows.append({"season": season, "gw": gw, "a_actual": a_actual, "a_ep": a_ep, "b_actual": b_actual, "b_ep": b_ep, "c_actual": c_actual, "c_ep": c_ep})
            print(f"{season} GW{gw:2d}: A={a_actual:3d} (ep={a_ep:5.1f})  B={b_actual:3d} (ep={b_ep:5.1f})  C={c_actual:3d} (ep={c_ep:5.1f})")
            (ARTIFACT_DIR / "tournament_results.json").write_text(json.dumps({"chosen_shrinkage": chosen_shrinkage, "rows": rows, "blanks": blanks}, indent=2))

    a_vals = np.array([r["a_actual"] for r in rows], dtype=float)
    b_vals = np.array([r["b_actual"] for r in rows], dtype=float)
    c_vals = np.array([r["c_actual"] for r in rows], dtype=float)

    b_mean, b_lo, b_hi = bootstrap_ci(b_vals - a_vals)
    c_mean, c_lo, c_hi = bootstrap_ci(c_vals - a_vals)

    a_ep_sum = sum(r["a_ep"] for r in rows)
    b_ep_sum = sum(r["b_ep"] for r in rows)
    c_ep_sum = sum(r["c_ep"] for r in rows)
    a_actual_sum = a_vals.sum()
    b_actual_sum = b_vals.sum()
    c_actual_sum = c_vals.sum()

    summary = {
        "tune_season": TUNE_SEASON, "test_seasons": TEST_SEASONS, "chosen_shrinkage": chosen_shrinkage,
        "n_gameweeks": len(rows), "n_blank": len(blanks), "blanks": blanks,
        "mean_a": float(a_vals.mean()), "mean_b": float(b_vals.mean()), "mean_c": float(c_vals.mean()),
        "b_vs_a": {"mean_diff": b_mean, "ci95": [b_lo, b_hi]},
        "c_vs_a": {"mean_diff": c_mean, "ci95": [c_lo, c_hi]},
        "calibration_check": {
            "a_predicted_ep_sum": a_ep_sum, "a_actual_sum": float(a_actual_sum), "a_ratio": a_ep_sum / a_actual_sum,
            "b_predicted_ep_sum": b_ep_sum, "b_actual_sum": float(b_actual_sum), "b_ratio": b_ep_sum / b_actual_sum,
            "c_predicted_ep_sum": c_ep_sum, "c_actual_sum": float(c_actual_sum), "c_ratio": c_ep_sum / c_actual_sum,
        },
        "rows": rows,
    }
    (ARTIFACT_DIR / "tournament_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({len(rows)} gameweeks, {len(blanks)} blank, chosen shrinkage={chosen_shrinkage}) ===")
    print(f"Mean realized: A={a_vals.mean():.2f}  B={b_vals.mean():.2f}  C={c_vals.mean():.2f}")
    print(f"B (decision-focused) vs A: mean diff {b_mean:+.3f}  95% CI [{b_lo:+.3f}, {b_hi:+.3f}]")
    print(f"C (hybrid) vs A:           mean diff {c_mean:+.3f}  95% CI [{c_lo:+.3f}, {c_hi:+.3f}]")
    print(f"Calibration ratio (predicted EP sum / actual sum, 1.0=unbiased): A={a_ep_sum / a_actual_sum:.3f}  B={b_ep_sum / b_actual_sum:.3f}  C={c_ep_sum / c_actual_sum:.3f}")
    print(f"\nWritten to {(ARTIFACT_DIR / 'tournament_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
