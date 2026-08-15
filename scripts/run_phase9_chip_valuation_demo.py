#!/usr/bin/env python3
"""Phase 9 real-data demonstration: chip valuation with option-value
framing, on 2022-23 (the same season used for the Phase 7 single-season
rolling-horizon demo). Reuses the ALREADY-VALIDATED lookahead transfer
policy (horizon=4, docs/phase7_multiweek_optimizer_report.md) to produce
one realistic, evolving squad through the season, then values each chip
against that real squad trajectory at each real gameweek.

Scope note: this project has not audited what chip rules/windows
actually applied in the historical 2022-23 season (only the CONFIRMED
2026/27 ruleset is in configs/seasons/2026_27.yaml, and older seasons
are known to have had different chip structures/counts — not something
this pass verifies). This script therefore treats the full GW2-38 range
as one open decision window for the "when would you play a one-shot
chip" analysis, rather than asserting specific historical chip-window
mechanics for 2022-23 — a methodological demonstration of the
option-value / optimal-stopping IDEA on real EP data, not a claim about
what was actually legal to do in that season.

Bench Boost and Triple Captain (memoryless — playing one doesn't change
future squad state) get the full optimal-stopping comparison: hindsight-
optimal (the true best gameweek, only knowable in a backtest) vs the 1/e
observe-then-commit rule (a real, causal, no-lookahead decision policy)
vs a naive "play at the first opportunity" baseline.

Free Hit is valued at every real gameweek (cheap — one extra
single-gameweek EV squad selection per week). Wildcard is valued at a
representative 8-gameweek sample (expensive — needs a second MILP solve
per sampled gameweek) using a 3-gameweek-ahead horizon, comparing the
REAL constrained free-transfer count the squad actually had at that
point against an unconstrained (free_transfers=15) rebuild over the same
window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.optimization import chips
from apex_fpl.optimization import squad as sq
from apex_fpl.optimization import transfers as tf
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase9_chip_valuation"
SEASON = "2022-23"
GW_RANGE = range(2, 39)
HORIZON = 4
SHORTLIST_PER_POSITION = 15
TIME_LIMIT = 60.0
WILDCARD_SAMPLE_GWS = [6, 11, 15, 19, 23, 27, 31, 35]
WILDCARD_HORIZON = 3
WILDCARD_FREE_TRANSFERS = 15


def build_gw_universe(season: str, gw: int, rules: dict):
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None
    sim_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    universe = [
        tf.HorizonPlayer(pid, meta["position"], meta["team"], meta["price"], (sim_results[pid].mean_points,))
        for pid, meta in data.candidates_meta.items() if pid in sim_results
    ]
    return universe


def assemble_window(gw_universes: list, t: int, horizon: int) -> list:
    eff_horizon = min(horizon, len(gw_universes) - t)
    by_offset = [{p.player_id: p for p in gw_universes[t + offset]} for offset in range(eff_horizon)]
    all_ids = set().union(*(set(lookup) for lookup in by_offset))
    window_players = []
    for pid in all_ids:
        ref = next(lookup[pid] for lookup in by_offset if pid in lookup)
        ep_tuple = tuple(by_offset[offset][pid].ep_by_gw[0] if pid in by_offset[offset] else 0.0 for offset in range(eff_horizon))
        window_players.append(tf.HorizonPlayer(pid, ref.position, ref.team, ref.price, ep_tuple))
    return window_players


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rules = scoring.load_scoring_rules("2026_27")

    print(f"=== Phase 9 chip valuation demo: {SEASON} GW{GW_RANGE.start}-{GW_RANGE.stop - 1} ===\n")
    print("--- Building per-gameweek simulated EP universes ---")
    gw_universes, real_gws = [], []
    for gw in GW_RANGE:
        universe = build_gw_universe(SEASON, gw, rules)
        if universe is None:
            print(f"GW{gw}: blank gameweek, skipped")
            continue
        gw_universes.append(universe)
        real_gws.append(gw)
    print(f"{len(real_gws)} valid gameweeks\n")

    player_meta = {p.player_id: (p.position, p.team) for p in gw_universes[0]}
    ep0 = {p.player_id: p.ep_by_gw[0] for p in gw_universes[0]}
    initial_candidates = [sq.PlayerCandidate(pid, pos, team, next(p.price for p in gw_universes[0] if p.player_id == pid), ep0[pid]) for pid, (pos, team) in player_meta.items()]
    initial_squad = sq.select_squad(initial_candidates, budget=sq.BUDGET)
    initial_ids = [p.player_id for p in initial_squad]
    initial_bank = sq.BUDGET - sum(p.price for p in initial_squad)
    initial_sell_prices = {p.player_id: p.price for p in initial_squad}
    print(f"--- Building the lookahead (horizon={HORIZON}) squad trajectory (the same validated Phase 7 policy) ---")
    steps = tf.rolling_horizon_transfers(
        initial_ids, dict(initial_sell_prices), initial_bank, 1, gw_universes, horizon=HORIZON,
        time_limit=TIME_LIMIT, shortlist_per_position=SHORTLIST_PER_POSITION,
    )
    print(f"Trajectory built: {len(steps)} gameweeks\n")

    print("--- Bench Boost / Triple Captain values at every real gameweek ---")
    bb_values, tc_values = [], []
    fh_values = []
    for gw, step, universe in zip(real_gws, steps, gw_universes):
        ep_by_id = {p.player_id: p.ep_by_gw[0] for p in universe}
        bench_ep = [ep_by_id.get(pid, 0.0) for pid in step.squad if pid not in step.starters]
        bb_values.append(chips.value_bench_boost(bench_ep))
        tc_values.append(chips.value_triple_captain(ep_by_id.get(step.captain, 0.0)))

        current_xi_ep = sum(ep_by_id.get(pid, 0.0) for pid in step.starters) + ep_by_id.get(step.captain, 0.0)
        fh_candidates = [sq.PlayerCandidate(pid, player_meta.get(pid, (None, None))[0] or next(p.position for p in universe if p.player_id == pid),
                                             player_meta.get(pid, (None, None))[1] or next(p.team for p in universe if p.player_id == pid),
                                             next(p.price for p in universe if p.player_id == pid), ep) for pid, ep in ep_by_id.items()]
        best_squad = sq.select_squad(fh_candidates, budget=sq.BUDGET)
        best_xi = sq.select_starting_xi(best_squad)
        best_possible_ep = sum(p.expected_points for p in best_xi.starters) + best_xi.captain.expected_points
        fh_values.append(chips.value_free_hit(current_xi_ep, best_possible_ep))
        print(f"  GW{gw:2d}: bench_boost={bb_values[-1]:6.2f}  triple_captain={tc_values[-1]:6.2f}  free_hit={fh_values[-1]:6.2f}")

    def stopping_analysis(values: list[float], label: str) -> dict:
        hindsight_idx = int(np.argmax(values))
        stopping_idx = chips.apply_1e_stopping_rule(values)
        naive_idx = 0
        result = {
            "hindsight_gw": real_gws[hindsight_idx], "hindsight_value": values[hindsight_idx],
            "stopping_gw": real_gws[stopping_idx], "stopping_value": values[stopping_idx],
            "naive_gw": real_gws[naive_idx], "naive_value": values[naive_idx],
        }
        print(f"\n{label}: hindsight-optimal GW{result['hindsight_gw']} ({result['hindsight_value']:.2f}), "
              f"1/e-rule GW{result['stopping_gw']} ({result['stopping_value']:.2f}), "
              f"naive-first GW{result['naive_gw']} ({result['naive_value']:.2f})")
        return result

    bb_analysis = stopping_analysis(bb_values, "Bench Boost")
    tc_analysis = stopping_analysis(tc_values, "Triple Captain")

    print("\n--- Wildcard values at a representative 8-gameweek sample (horizon=3, unconstrained vs actual FT count) ---")
    wildcard_rows = []
    for gw in WILDCARD_SAMPLE_GWS:
        t = real_gws.index(gw)
        step = steps[t]
        window = assemble_window(gw_universes, t, WILDCARD_HORIZON)
        actual_ft = step.free_transfers_available
        constrained_plan = tf.plan_transfers(step.squad, {}, step.bank_after, actual_ft, window, horizon=min(WILDCARD_HORIZON, len(gw_universes) - t),
                                              time_limit=TIME_LIMIT, shortlist_per_position=SHORTLIST_PER_POSITION)
        unconstrained_plan = tf.plan_transfers(step.squad, {}, step.bank_after, WILDCARD_FREE_TRANSFERS, window, horizon=min(WILDCARD_HORIZON, len(gw_universes) - t),
                                                time_limit=TIME_LIMIT, shortlist_per_position=SHORTLIST_PER_POSITION)
        wc_value = chips.value_wildcard(constrained_plan.total_net_expected_points, unconstrained_plan.total_net_expected_points)
        wildcard_rows.append({"gw": gw, "actual_free_transfers": actual_ft, "wildcard_value": wc_value})
        print(f"  GW{gw:2d}: actual_ft={actual_ft}  wildcard_value={wc_value:6.2f}")

    summary = {
        "season": SEASON, "real_gws": real_gws,
        "bench_boost_values": bb_values, "triple_captain_values": tc_values, "free_hit_values": fh_values,
        "bench_boost_stopping_analysis": bb_analysis, "triple_captain_stopping_analysis": tc_analysis,
        "wildcard_sample": wildcard_rows,
    }
    (ARTIFACT_DIR / "chip_valuation_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'chip_valuation_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
