#!/usr/bin/env python3
"""Phase 7 real-data validation: does a genuine multi-gameweek transfer
plan beat a myopic (single-gameweek-lookahead) rolling transfer policy,
and does either beat never transferring at all? Three policies, ONE
evolving squad each, run across every real, non-blank gameweek of
2022-23 (the same "grade-A audited" season used for the Phase 2/3
baselines) from GW2 (the earliest a team model can be fit) through GW38:

1. **Buy-and-hold**: the initial squad (chosen once via the existing
   single-gameweek EV optimizer on GW2's forecast) is never touched again
   — only starting XI / captain are re-picked each week. No transfers, no
   hits, ever.
2. **Myopic rolling**: `transfers.rolling_horizon_transfers(..., horizon=1)`
   — re-solves every real gameweek seeing only that gameweek's forecast.
3. **Lookahead rolling**: the same driver with `horizon=4` — sees up to 4
   real gameweeks of forecast ahead at every decision point.

All three start from the identical initial squad/bank/free-transfers, so
any difference is attributable to the transfer POLICY, not initial
conditions. Realized scores use REAL historical `total_points` (not the
simulated EP), exactly like every other decision-level replay in this
project (docs/phase3_extended_replay_report.md, the CVaR replay) — hit
costs (-4/extra transfer) are real point deductions applied to the actual
realized score, matching true FPL rules.

Two honestly-flagged simplifications carried over from
apex_fpl.optimization.transfers' own docstring, both consequences of
gaps already tracked in docs/fpl_gap_analysis.md (no price-change model,
Part XXII): each week's plan only ever sees THAT week's own real,
already-known price for the whole lookahead window (never a future
week's real price) — this is both the natural simplification given no
price model exists AND, as a side effect, the leakage-safe choice: a real
manager deciding at gameweek t cannot see gameweek t+2's actual price.
And: a genuine blank gameweek (2022-23 GW7) is skipped entirely from the
sequence for all three policies, including free-transfer accrual — a
real manager still banks a free transfer during a blank gameweek, this
script's simplified "one step per planned gameweek" accounting does not
model that, a small honestly-flagged deviation from the real rules.
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
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase7_rolling_horizon_replay"
SEASON = "2022-23"
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


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rules = scoring.load_scoring_rules("2026_27")

    print(f"=== Phase 7 rolling-horizon replay: {SEASON} GW{GW_RANGE.start}-{GW_RANGE.stop - 1} ===\n")
    print("--- Building per-gameweek simulated EP universes (shared across all 3 policies) ---")
    gw_universes, actuals_by_gw, real_gws = [], [], []
    for gw in GW_RANGE:
        result = build_gw_universe(SEASON, gw, rules)
        if result is None:
            print(f"GW{gw}: blank gameweek, skipped")
            continue
        universe, actual = result
        gw_universes.append(universe)
        actuals_by_gw.append(actual)
        real_gws.append(gw)
        print(f"GW{gw}: {len(universe)} players simulated")
    print(f"\n{len(real_gws)} valid gameweeks: {real_gws}\n")

    player_meta = {p.player_id: (p.position, p.team) for p in gw_universes[0]}
    ep0 = {p.player_id: p.ep_by_gw[0] for p in gw_universes[0]}
    initial_candidates = [sq.PlayerCandidate(pid, pos, team, next(p.price for p in gw_universes[0] if p.player_id == pid), ep0[pid]) for pid, (pos, team) in player_meta.items()]
    initial_squad = sq.select_squad(initial_candidates, budget=sq.BUDGET)
    initial_ids = [p.player_id for p in initial_squad]
    initial_bank = sq.BUDGET - sum(p.price for p in initial_squad)
    initial_sell_prices = {p.player_id: p.price for p in initial_squad}
    print(f"Initial squad (GW{real_gws[0]}): {len(initial_ids)} players, bank={initial_bank:.1f}\n")

    # ---- Policy 1: buy-and-hold ----
    buyhold_realized = []
    for universe, actual in zip(gw_universes, actuals_by_gw):
        ep_by_id = {p.player_id: p.ep_by_gw[0] for p in universe}
        candidates = [sq.PlayerCandidate(pid, player_meta[pid][0], player_meta[pid][1], 0.0, ep_by_id.get(pid, 0.0)) for pid in initial_ids]
        xi = sq.select_starting_xi(candidates)
        realized = sum(actual.get(p.player_id, 0) for p in xi.starters) + actual.get(xi.captain.player_id, 0)
        buyhold_realized.append(realized)
    print(f"--- Buy-and-hold: total={sum(buyhold_realized)}, mean={np.mean(buyhold_realized):.2f} ---\n")

    def run_rolling(horizon: int, label: str):
        print(f"--- {label} (horizon={horizon}) ---")
        steps = tf.rolling_horizon_transfers(
            initial_ids, dict(initial_sell_prices), initial_bank, 1, gw_universes, horizon=horizon,
            time_limit=TIME_LIMIT, shortlist_per_position=SHORTLIST_PER_POSITION,
        )
        realized_list, transfers_count, hits_count = [], 0, 0
        for gw, step, actual in zip(real_gws, steps, actuals_by_gw):
            realized = sum(actual.get(pid, 0) for pid in step.starters) + actual.get(step.captain, 0) + step.hit_points
            realized_list.append(realized)
            transfers_count += len(step.transfers_in)
            hits_count += step.paid_transfers
            print(f"  GW{gw:2d}: realized={realized:5.1f}  transfers_in={step.transfers_in}  paid={step.paid_transfers}")
        print(f"  total={sum(realized_list):.1f}  mean={np.mean(realized_list):.2f}  total_transfers={transfers_count}  total_hits={hits_count}\n")
        return realized_list, transfers_count, hits_count

    myopic_realized, myopic_transfers, myopic_hits = run_rolling(1, "Myopic rolling")
    lookahead_realized, lookahead_transfers, lookahead_hits = run_rolling(HORIZON, "Lookahead rolling")

    buyhold_arr = np.array(buyhold_realized, dtype=float)
    myopic_arr = np.array(myopic_realized, dtype=float)
    lookahead_arr = np.array(lookahead_realized, dtype=float)

    diff_myopic_vs_buyhold = myopic_arr - buyhold_arr
    diff_lookahead_vs_myopic = lookahead_arr - myopic_arr
    diff_lookahead_vs_buyhold = lookahead_arr - buyhold_arr

    m1, lo1, hi1 = bootstrap_ci(diff_myopic_vs_buyhold)
    m2, lo2, hi2 = bootstrap_ci(diff_lookahead_vs_myopic)
    m3, lo3, hi3 = bootstrap_ci(diff_lookahead_vs_buyhold)

    summary = {
        "season": SEASON, "n_gameweeks": len(real_gws), "real_gws": real_gws, "horizon": HORIZON,
        "buyhold_total": float(buyhold_arr.sum()), "myopic_total": float(myopic_arr.sum()), "lookahead_total": float(lookahead_arr.sum()),
        "myopic_transfers": myopic_transfers, "myopic_hits": myopic_hits,
        "lookahead_transfers": lookahead_transfers, "lookahead_hits": lookahead_hits,
        "myopic_vs_buyhold": {"mean_diff": m1, "ci95": [lo1, hi1]},
        "lookahead_vs_myopic": {"mean_diff": m2, "ci95": [lo2, hi2]},
        "lookahead_vs_buyhold": {"mean_diff": m3, "ci95": [lo3, hi3]},
        "per_gw": {
            "gw": real_gws, "buyhold": buyhold_realized, "myopic": myopic_realized, "lookahead": lookahead_realized,
        },
    }
    (ARTIFACT_DIR / "replay_results.json").write_text(json.dumps(summary, indent=2))

    print("=== Summary ===")
    print(f"Totals over {len(real_gws)} gameweeks: buy-hold={buyhold_arr.sum():.0f}  myopic={myopic_arr.sum():.0f}  lookahead={lookahead_arr.sum():.0f}")
    print(f"Myopic vs buy-hold:     mean diff {m1:+.3f}  95% CI [{lo1:+.3f}, {hi1:+.3f}]")
    print(f"Lookahead vs myopic:    mean diff {m2:+.3f}  95% CI [{lo2:+.3f}, {hi2:+.3f}]  (transfers: myopic={myopic_transfers}/hits={myopic_hits}, lookahead={lookahead_transfers}/hits={lookahead_hits})")
    print(f"Lookahead vs buy-hold:  mean diff {m3:+.3f}  95% CI [{lo3:+.3f}, {hi3:+.3f}]")
    print(f"\nWritten to {(ARTIFACT_DIR / 'replay_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
