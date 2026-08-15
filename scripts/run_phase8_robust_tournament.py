#!/usr/bin/env python3
"""Phase 8: a 3-way single-gameweek squad-selection tournament — expected
value (EV) vs CVaR (tail-risk-averse) vs mean-variance/MAD
(dispersion-averse) — extending the CVaR multi-gameweek replay
(scripts/run_cvar_multi_gw_replay.py, docs/robust_captaincy_report.md) to
the SAME structurally-different second risk formulation described in
apex_fpl.optimization.robust's module docstring, rather than re-running
the already-closed CVaR question.

Same design as the CVaR replay for a clean, direct comparison: 8
gameweeks per season (every 4th from GW7), across the 4 independent
seasons used throughout this project (2020-21, 2022-23, 2023-24,
2024-25). n=400 scenarios per gameweek, both robust solves time-bounded
to 90s (see robust.select_squad_cvar's docstring for why solve time is
unpredictable on real gameweek data — a known MILP phenomenon, not a
bug), with MIP gap reported transparently rather than assumed.

alpha=0.2 (CVaR) and lambda_risk=0.5 (MAD) are the same/an analogous
starting choice to the CVaR replay's own alpha — not deeply tuned via a
grid search, which is out of scope for this pass and explicitly flagged
as a limitation, matching how the CVaR work's own alpha was chosen.
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
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase8_robust_tournament"
SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GAMEWEEKS = [7, 11, 15, 19, 23, 27, 31, 35]
N_SCENARIOS = 400
CVAR_ALPHA = 0.2
MAD_LAMBDA = 0.5
SEED = 2026
N_BOOTSTRAP = 5000
TIME_LIMIT = 90.0


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
    scenario_idx = rng.choice(n_sims, size=min(N_SCENARIOS, n_sims), replace=False)
    scenario_candidates = [
        rb.ScenarioPlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].samples[scenario_idx])
        for pid, m in data.candidates_meta.items()
    ]

    cvar_squad_sc, cvar_diag = rb.select_squad_cvar(scenario_candidates, alpha=CVAR_ALPHA, budget=sq.BUDGET, time_limit=TIME_LIMIT, return_diagnostics=True)
    cvar_ids = {p.player_id for p in cvar_squad_sc}
    cvar_squad = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in data.candidates_meta.items() if pid in cvar_ids]
    cvar_xi = sq.select_starting_xi(cvar_squad)

    mad_squad_sc, mad_diag = rb.select_squad_mean_variance(scenario_candidates, lambda_risk=MAD_LAMBDA, budget=sq.BUDGET, time_limit=TIME_LIMIT, return_diagnostics=True)
    mad_ids = {p.player_id for p in mad_squad_sc}
    mad_squad = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in data.candidates_meta.items() if pid in mad_ids]
    mad_xi = sq.select_starting_xi(mad_squad)

    actual_by_player = {r["element"]: int(r["total_points"]) for r in data.target_rows}
    ev_realized = realized_points(ev_xi, actual_by_player)
    cvar_realized = realized_points(cvar_xi, actual_by_player)
    mad_realized = realized_points(mad_xi, actual_by_player)

    return {
        "season": season, "gw": gw,
        "ev_realized": ev_realized, "cvar_realized": cvar_realized, "mad_realized": mad_realized,
        "cvar_overlap_ev": len({p.player_id for p in ev_squad} & {p.player_id for p in cvar_squad}),
        "mad_overlap_ev": len({p.player_id for p in ev_squad} & {p.player_id for p in mad_squad}),
        "cvar_proven_optimal": cvar_diag["proven_optimal"], "cvar_mip_gap": cvar_diag["mip_gap"],
        "mad_proven_optimal": mad_diag["proven_optimal"], "mad_mip_gap": mad_diag["mip_gap"],
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 8 robust tournament (EV vs CVaR vs MAD): {len(SEASONS)} seasons x {len(GAMEWEEKS)} gameweeks ===\n")

    rows, blanks = [], []
    for season in SEASONS:
        for gw in GAMEWEEKS:
            result = run_one_gameweek(season, gw, seed=3000)
            if result is None:
                print(f"{season} GW{gw}: blank gameweek, skipped")
                blanks.append((season, gw))
                continue
            rows.append(result)
            cvar_tag = "optimal" if result["cvar_proven_optimal"] else f"gap={result['cvar_mip_gap']:.3f}"
            mad_tag = "optimal" if result["mad_proven_optimal"] else f"gap={result['mad_mip_gap']:.3f}"
            print(f"{season} GW{gw:2d}: EV={result['ev_realized']:3d}  CVaR={result['cvar_realized']:3d} ({cvar_tag})  MAD={result['mad_realized']:3d} ({mad_tag})")
            (ARTIFACT_DIR / "tournament_results.json").write_text(json.dumps({"rows": rows, "blanks": blanks}, indent=2))

    ev_vals = np.array([r["ev_realized"] for r in rows], dtype=float)
    cvar_vals = np.array([r["cvar_realized"] for r in rows], dtype=float)
    mad_vals = np.array([r["mad_realized"] for r in rows], dtype=float)

    cvar_diff = cvar_vals - ev_vals
    mad_diff = mad_vals - ev_vals
    mad_vs_cvar_diff = mad_vals - cvar_vals

    cvar_mean, cvar_lo, cvar_hi = bootstrap_ci(cvar_diff)
    mad_mean, mad_lo, mad_hi = bootstrap_ci(mad_diff)
    mc_mean, mc_lo, mc_hi = bootstrap_ci(mad_vs_cvar_diff)

    tail_k = max(1, int(np.ceil(0.25 * len(rows))))
    ev_tail = float(np.mean(np.sort(ev_vals)[:tail_k]))
    cvar_tail = float(np.mean(np.sort(cvar_vals)[:tail_k]))
    mad_tail = float(np.mean(np.sort(mad_vals)[:tail_k]))

    summary = {
        "n_gameweeks": len(rows), "n_blank": len(blanks), "blanks": blanks,
        "mean_ev": float(ev_vals.mean()), "mean_cvar": float(cvar_vals.mean()), "mean_mad": float(mad_vals.mean()),
        "worst_ev": float(ev_vals.min()), "worst_cvar": float(cvar_vals.min()), "worst_mad": float(mad_vals.min()),
        "bottom_25pct_avg_ev": ev_tail, "bottom_25pct_avg_cvar": cvar_tail, "bottom_25pct_avg_mad": mad_tail,
        "realized_mad_ev": rb.compute_mad(ev_vals), "realized_mad_cvar": rb.compute_mad(cvar_vals), "realized_mad_mad": rb.compute_mad(mad_vals),
        "cvar_vs_ev": {"mean_diff": cvar_mean, "ci95": [cvar_lo, cvar_hi]},
        "mad_vs_ev": {"mean_diff": mad_mean, "ci95": [mad_lo, mad_hi]},
        "mad_vs_cvar": {"mean_diff": mc_mean, "ci95": [mc_lo, mc_hi]},
        "n_cvar_time_limited": sum(1 for r in rows if not r["cvar_proven_optimal"]),
        "n_mad_time_limited": sum(1 for r in rows if not r["mad_proven_optimal"]),
        "rows": rows,
    }
    (ARTIFACT_DIR / "tournament_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({len(rows)} gameweeks, {len(blanks)} blank) ===")
    print(f"Mean realized: EV={ev_vals.mean():.2f}  CVaR={cvar_vals.mean():.2f}  MAD={mad_vals.mean():.2f}")
    print(f"Worst single gameweek: EV={ev_vals.min():.0f}  CVaR={cvar_vals.min():.0f}  MAD={mad_vals.min():.0f}")
    print(f"Bottom-25%-of-gameweeks average: EV={ev_tail:.2f}  CVaR={cvar_tail:.2f}  MAD={mad_tail:.2f}")
    print(f"Dispersion of REALIZED scores (MAD across the 32 gameweeks): EV={rb.compute_mad(ev_vals):.2f}  CVaR={rb.compute_mad(cvar_vals):.2f}  MAD-optimizer={rb.compute_mad(mad_vals):.2f}")
    print(f"CVaR vs EV:  mean diff {cvar_mean:+.2f}  95% CI [{cvar_lo:+.2f}, {cvar_hi:+.2f}]")
    print(f"MAD vs EV:   mean diff {mad_mean:+.2f}  95% CI [{mad_lo:+.2f}, {mad_hi:+.2f}]")
    print(f"MAD vs CVaR: mean diff {mc_mean:+.2f}  95% CI [{mc_lo:+.2f}, {mc_hi:+.2f}]")
    print(f"\nWritten to {(ARTIFACT_DIR / 'tournament_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
