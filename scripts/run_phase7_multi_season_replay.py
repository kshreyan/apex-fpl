#!/usr/bin/env python3
"""Extends scripts/run_phase7_rolling_horizon_replay.py's single-season
(2022-23) finding to the same 4 independent seasons used everywhere else
in this project (2020-21, 2022-23, 2023-24, 2024-25 — 2021-22 stays
excluded as the Phase 4b tuning season, 2019-20 stays excluded for its
incompatible older schema), directly analogous to the Phase 3 extension
(docs/phase3_extended_replay_report.md) and the CVaR multi-gameweek
replay: does the "lookahead vs myopic" gap (+0.25 pts/gw, 95% CI
[-3.25,+3.58] on one season — not yet significant) resolve with more
data, the same way Phase 4b's original 2-season inconclusive
decision-level finding did?

Each season gets its OWN fresh initial squad (chosen via the existing
single-gameweek EV optimizer on that season's own GW2 forecast) and its
own independent buy-hold / myopic / lookahead run — seasons are never
mixed mid-replay. Per-gameweek realized-point differences are pooled
across all 4 seasons for the final significance test, block-bootstrapped
with block = (season, gameweek), the same convention used throughout
this project (docs/phase3_extended_replay_report.md, the CVaR replay,
scripts/run_phase6_multi_gw_validation.py).

Same two honestly-flagged simplifications as the single-season script
(module docstring of apex_fpl.optimization.transfers): prices held fixed
within a single horizon call (no price-change model, Part XXII — also
the leakage-safe choice), and a blank gameweek is skipped from the
sequence entirely including free-transfer accrual (2022-23 GW7 is the
only genuine blank among these 4 seasons).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.optimization import squad as sq
from apex_fpl.optimization import transfers as tf
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase7_multi_season_replay"
SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GW_RANGE = range(2, 39)
HORIZON = 4
SHORTLIST_PER_POSITION = 15
TIME_LIMIT = 60.0
SEED = 2026
N_BOOTSTRAP = 5000


def build_gw_universe(season: str, gw: int, rules: dict):
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None
    sim_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    actual_by_player = {r["element"]: int(r["total_points"]) for r in data.target_rows}
    universe = [
        tf.HorizonPlayer(pid, meta["position"], meta["team"], meta["price"], (sim_results[pid].mean_points,))
        for pid, meta in data.candidates_meta.items() if pid in sim_results
    ]
    return universe, actual_by_player


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_season(season: str, rules: dict) -> dict:
    print(f"=== {season} ===")
    gw_universes, actuals_by_gw, real_gws = [], [], []
    for gw in GW_RANGE:
        result = build_gw_universe(season, gw, rules)
        if result is None:
            print(f"{season} GW{gw}: blank gameweek, skipped")
            continue
        universe, actual = result
        gw_universes.append(universe)
        actuals_by_gw.append(actual)
        real_gws.append(gw)
    print(f"{season}: {len(real_gws)} valid gameweeks")

    player_meta = {p.player_id: (p.position, p.team) for p in gw_universes[0]}
    ep0 = {p.player_id: p.ep_by_gw[0] for p in gw_universes[0]}
    initial_candidates = [sq.PlayerCandidate(pid, pos, team, next(p.price for p in gw_universes[0] if p.player_id == pid), ep0[pid]) for pid, (pos, team) in player_meta.items()]
    initial_squad = sq.select_squad(initial_candidates, budget=sq.BUDGET)
    initial_ids = [p.player_id for p in initial_squad]
    initial_bank = sq.BUDGET - sum(p.price for p in initial_squad)
    initial_sell_prices = {p.player_id: p.price for p in initial_squad}
    print(f"{season}: initial squad (GW{real_gws[0]}), bank={initial_bank:.1f}")

    buyhold_realized = []
    for universe, actual in zip(gw_universes, actuals_by_gw):
        ep_by_id = {p.player_id: p.ep_by_gw[0] for p in universe}
        candidates = [sq.PlayerCandidate(pid, player_meta[pid][0], player_meta[pid][1], 0.0, ep_by_id.get(pid, 0.0)) for pid in initial_ids]
        xi = sq.select_starting_xi(candidates)
        realized = sum(actual.get(p.player_id, 0) for p in xi.starters) + actual.get(xi.captain.player_id, 0)
        buyhold_realized.append(realized)

    def run_rolling(horizon: int, label: str):
        steps = tf.rolling_horizon_transfers(
            initial_ids, dict(initial_sell_prices), initial_bank, 1, gw_universes, horizon=horizon,
            time_limit=TIME_LIMIT, shortlist_per_position=SHORTLIST_PER_POSITION,
        )
        realized_list, transfers_count, hits_count = [], 0, 0
        for step, actual in zip(steps, actuals_by_gw):
            realized = sum(actual.get(pid, 0) for pid in step.starters) + actual.get(step.captain, 0) + step.hit_points
            realized_list.append(realized)
            transfers_count += len(step.transfers_in)
            hits_count += step.paid_transfers
        print(f"{season}: {label} total={sum(realized_list):.0f}  transfers={transfers_count}  hits={hits_count}")
        return realized_list, transfers_count, hits_count

    myopic_realized, myopic_transfers, myopic_hits = run_rolling(1, "myopic")
    lookahead_realized, lookahead_transfers, lookahead_hits = run_rolling(HORIZON, "lookahead")
    print()

    return {
        "season": season, "real_gws": real_gws,
        "buyhold": buyhold_realized, "myopic": myopic_realized, "lookahead": lookahead_realized,
        "myopic_transfers": myopic_transfers, "myopic_hits": myopic_hits,
        "lookahead_transfers": lookahead_transfers, "lookahead_hits": lookahead_hits,
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rules = scoring.load_scoring_rules("2026_27")
    print(f"=== Phase 7 multi-season rolling-horizon replay: {len(SEASONS)} seasons ===\n")

    season_results = []
    for season in SEASONS:
        result = run_season(season, rules)
        season_results.append(result)
        (ARTIFACT_DIR / "replay_results.json").write_text(json.dumps({"seasons": season_results}, indent=2))

    all_buyhold, all_myopic, all_lookahead = [], [], []
    total_myopic_transfers = total_myopic_hits = total_lookahead_transfers = total_lookahead_hits = 0
    for r in season_results:
        all_buyhold += r["buyhold"]
        all_myopic += r["myopic"]
        all_lookahead += r["lookahead"]
        total_myopic_transfers += r["myopic_transfers"]
        total_myopic_hits += r["myopic_hits"]
        total_lookahead_transfers += r["lookahead_transfers"]
        total_lookahead_hits += r["lookahead_hits"]

    buyhold_arr = np.array(all_buyhold, dtype=float)
    myopic_arr = np.array(all_myopic, dtype=float)
    lookahead_arr = np.array(all_lookahead, dtype=float)

    diff_myopic_vs_buyhold = myopic_arr - buyhold_arr
    diff_lookahead_vs_myopic = lookahead_arr - myopic_arr
    diff_lookahead_vs_buyhold = lookahead_arr - buyhold_arr

    m1, lo1, hi1 = bootstrap_ci(diff_myopic_vs_buyhold)
    m2, lo2, hi2 = bootstrap_ci(diff_lookahead_vs_myopic)
    m3, lo3, hi3 = bootstrap_ci(diff_lookahead_vs_buyhold)

    n_wins = int((diff_lookahead_vs_myopic > 0).sum())
    n_losses = int((diff_lookahead_vs_myopic < 0).sum())
    n_ties = int((diff_lookahead_vs_myopic == 0).sum())

    summary = {
        "seasons": SEASONS, "n_gameweeks_pooled": len(all_buyhold), "horizon": HORIZON,
        "buyhold_total": float(buyhold_arr.sum()), "myopic_total": float(myopic_arr.sum()), "lookahead_total": float(lookahead_arr.sum()),
        "myopic_transfers": total_myopic_transfers, "myopic_hits": total_myopic_hits,
        "lookahead_transfers": total_lookahead_transfers, "lookahead_hits": total_lookahead_hits,
        "myopic_vs_buyhold": {"mean_diff": m1, "ci95": [lo1, hi1]},
        "lookahead_vs_myopic": {"mean_diff": m2, "ci95": [lo2, hi2], "wins": n_wins, "losses": n_losses, "ties": n_ties},
        "lookahead_vs_buyhold": {"mean_diff": m3, "ci95": [lo3, hi3]},
        "per_season": season_results,
    }
    (ARTIFACT_DIR / "replay_results.json").write_text(json.dumps(summary, indent=2))

    print("=== Pooled summary across 4 seasons ===")
    print(f"n_gameweeks_pooled={len(all_buyhold)}")
    print(f"Totals: buy-hold={buyhold_arr.sum():.0f}  myopic={myopic_arr.sum():.0f}  lookahead={lookahead_arr.sum():.0f}")
    print(f"Myopic vs buy-hold:     mean diff {m1:+.3f}  95% CI [{lo1:+.3f}, {hi1:+.3f}]")
    print(f"Lookahead vs myopic:    mean diff {m2:+.3f}  95% CI [{lo2:+.3f}, {hi2:+.3f}]  (wins={n_wins} losses={n_losses} ties={n_ties})")
    print(f"  transfers: myopic={total_myopic_transfers}/hits={total_myopic_hits}, lookahead={total_lookahead_transfers}/hits={total_lookahead_hits}")
    print(f"Lookahead vs buy-hold:  mean diff {m3:+.3f}  95% CI [{lo3:+.3f}, {hi3:+.3f}]")
    print(f"\nWritten to {(ARTIFACT_DIR / 'replay_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
