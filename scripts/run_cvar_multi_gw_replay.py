#!/usr/bin/env python3
"""The multi-gameweek downside replay gating CVaR promotion (spec Part
XXVI's own standard, matching the bar used everywhere else in this
project — Phases 4a/4b, docs/phase3_extended_replay_report.md).

Question: does the CVaR-selected squad actually produce fewer disaster
gameweeks / a better realized-points tail across many REAL gameweeks, not
just in one gameweek's simulation (docs/robust_captaincy_report.md) or
across many scenario-count trials on ONE gameweek (the earlier
sensitivity sweep)? Those were necessary but not sufficient evidence —
this is the decision-level test.

Scope: 8 gameweeks per season (every 4th from GW7), across the 4
independent seasons used throughout this project (2020-21, 2022-23,
2023-24, 2024-25 — 2021-22 stays excluded as the Phase 4b tuning season,
2019-20 stays excluded for its incompatible older schema) = 32 total
gameweek observations. n=400 CVaR scenarios per gameweek (not 800-1600):
per-gameweek precision matters less here than gameweek COUNT for a valid
multi-observation significance test, and 400 already showed the correct
DIRECTION consistently in the sensitivity sweep — the point-by-point
noise it carries is exactly what aggregating across 32 independent
gameweeks is for.

CVAR_TIME_LIMIT bounds each gameweek's CVaR solve to 90s. A first
unbounded attempt at this sweep was killed after 2 gameweeks took ~33
minutes combined — some real problem instances (especially early-season
gameweeks with less differentiated player values) can take 15+ minutes to
prove optimality, a known MILP phenomenon, not a bug (see
robust.select_squad_cvar's docstring). A 90s-bounded solve returns
scipy's best feasible solution with a reported MIP gap instead — every
per-gameweek result records whether it was proven optimal or time-limited
and by how much, so solution quality is reported honestly rather than
assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.optimization import robust as rb
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "cvar_multi_gw_replay"
SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GAMEWEEKS = [7, 11, 15, 19, 23, 27, 31, 35]
N_SCENARIOS_FOR_CVAR = 400
CVAR_ALPHA = 0.2
SEED = 2026
N_BOOTSTRAP = 5000
CVAR_TIME_LIMIT = 90.0  # seconds; see robust.select_squad_cvar's docstring — real solve times vary from <10s to 15+ minutes unbounded, a known MILP phenomenon on less-differentiated early-season data, not a bug


def realized_points(xi, actual_by_player: dict[str, int]) -> int:
    return sum(actual_by_player.get(p.player_id, 0) for p in xi.starters) + actual_by_player.get(xi.captain.player_id, 0)


def run_one_gameweek(season: str, gw: int, seed: int) -> dict | None:
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None

    rules = scoring.load_scoring_rules("2026_27")
    sim_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    n_sims = len(next(iter(sim_results.values())).samples)

    ev_candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in data.candidates_meta.items()]
    ev_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    ev_xi = sq.select_starting_xi(ev_squad)

    rng = np.random.default_rng(seed)
    scenario_idx = rng.choice(n_sims, size=min(N_SCENARIOS_FOR_CVAR, n_sims), replace=False)
    cvar_candidates = [
        rb.ScenarioPlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].samples[scenario_idx])
        for pid, m in data.candidates_meta.items()
    ]
    cvar_squad_sc, cvar_diag = rb.select_squad_cvar(
        cvar_candidates, alpha=CVAR_ALPHA, budget=sq.BUDGET, time_limit=CVAR_TIME_LIMIT, return_diagnostics=True,
    )
    cvar_ids = {p.player_id for p in cvar_squad_sc}
    cvar_squad = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in data.candidates_meta.items() if pid in cvar_ids]
    cvar_xi = sq.select_starting_xi(cvar_squad)

    actual_by_player = {r["element"]: int(r["total_points"]) for r in data.target_rows}
    ev_realized = realized_points(ev_xi, actual_by_player)
    cvar_realized = realized_points(cvar_xi, actual_by_player)

    return {
        "season": season, "gw": gw,
        "ev_realized": ev_realized, "cvar_realized": cvar_realized,
        "diff": cvar_realized - ev_realized,
        "squad_overlap": len({p.player_id for p in ev_squad} & {p.player_id for p in cvar_squad}),
        "cvar_proven_optimal": cvar_diag["proven_optimal"],
        "cvar_mip_gap": cvar_diag["mip_gap"],
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== CVaR multi-gameweek downside replay: {len(SEASONS)} seasons x {len(GAMEWEEKS)} gameweeks ===\n")

    rows, blanks = [], []
    for season in SEASONS:
        for gw in GAMEWEEKS:
            result = run_one_gameweek(season, gw, seed=3000)
            if result is None:
                print(f"{season} GW{gw}: blank gameweek, skipped")
                blanks.append((season, gw))
                continue
            rows.append(result)
            opt_tag = "optimal" if result["cvar_proven_optimal"] else f"gap={result['cvar_mip_gap']:.3f}"
            print(f"{season} GW{gw:2d}: EV={result['ev_realized']:3d}  CVaR={result['cvar_realized']:3d}  "
                  f"diff={result['diff']:+3d}  overlap={result['squad_overlap']}/15  cvar_solve=({opt_tag})")
            (ARTIFACT_DIR / "replay_results.json").write_text(json.dumps({"rows": rows, "blanks": blanks}, indent=2))

    ev_vals = np.array([r["ev_realized"] for r in rows], dtype=float)
    cvar_vals = np.array([r["cvar_realized"] for r in rows], dtype=float)
    diffs = np.array([r["diff"] for r in rows], dtype=float)

    mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs)
    n_wins = int((diffs > 0).sum())
    n_losses = int((diffs < 0).sum())
    n_ties = int((diffs == 0).sum())

    tail_k = max(1, int(np.ceil(0.25 * len(rows))))
    ev_tail = float(np.mean(np.sort(ev_vals)[:tail_k]))
    cvar_tail = float(np.mean(np.sort(cvar_vals)[:tail_k]))

    summary = {
        "n_gameweeks": len(rows), "n_blank": len(blanks), "blanks": blanks,
        "n_proven_optimal": sum(1 for r in rows if r["cvar_proven_optimal"]),
        "n_time_limited": sum(1 for r in rows if not r["cvar_proven_optimal"]),
        "mean_mip_gap_when_time_limited": float(np.mean([r["cvar_mip_gap"] for r in rows if not r["cvar_proven_optimal"]])) if any(not r["cvar_proven_optimal"] for r in rows) else None,
        "mean_ev": float(ev_vals.mean()), "mean_cvar": float(cvar_vals.mean()),
        "worst_ev": float(ev_vals.min()), "worst_cvar": float(cvar_vals.min()),
        "bottom_25pct_avg_ev": ev_tail, "bottom_25pct_avg_cvar": cvar_tail,
        "mean_diff": mean_diff, "mean_diff_95ci": [ci_lo, ci_hi],
        "wins_cvar": n_wins, "losses_cvar": n_losses, "ties": n_ties,
        "rows": rows,
    }
    (ARTIFACT_DIR / "replay_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({len(rows)} gameweeks, {len(blanks)} blank) ===")
    print(f"Mean realized: EV={ev_vals.mean():.2f}  CVaR={cvar_vals.mean():.2f}  (diff {mean_diff:+.2f}, 95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}])")
    print(f"Worst single gameweek: EV={ev_vals.min():.0f}  CVaR={cvar_vals.min():.0f}")
    print(f"Bottom-25%-of-gameweeks average (empirical CVaR at the decision level): EV={ev_tail:.2f}  CVaR={cvar_tail:.2f}  (diff {cvar_tail - ev_tail:+.2f})")
    print(f"Gameweeks: CVaR won {n_wins}, EV won {n_losses}, tied {n_ties}")
    print(f"\nWritten to {(ARTIFACT_DIR / 'replay_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
