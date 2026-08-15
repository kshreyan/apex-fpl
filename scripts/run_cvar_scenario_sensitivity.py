#!/usr/bin/env python3
"""Tests whether increasing the CVaR optimizer's scenario count closes the
out-of-sample downside-protection gap found in docs/robust_captaincy_report.md.

Reuses the EXACT same simulation setup as that report (2022-23 GW20, same
fixed default seed for the underlying 15,000-simulation Monte Carlo draw,
via mc.simulate_gameweek's own default seed=2026) so the only variable
across configurations is how many scenarios the CVaR MILP is optimized
against. Every resulting squad is evaluated out-of-sample on the SAME
full simulation set for a fair, controlled comparison. Multiple random
subsample seeds are tried per scenario count, since a single subsample
draw could otherwise make "more scenarios helps/doesn't help" look like a
trend when it's really just which particular players got sampled well or
badly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_robust_captaincy_demo import CVAR_ALPHA, SEASON, TARGET_GW, build_players_for_sim  # noqa: E402

from apex_fpl.optimization import robust as rb  # noqa: E402
from apex_fpl.optimization import squad as sq  # noqa: E402
from apex_fpl.rules import scoring  # noqa: E402
from apex_fpl.simulation import monte_carlo as mc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "robust_captaincy_demo"
SCENARIO_COUNTS = [400, 800, 1600]
N_SEEDS_PER_COUNT = 2
EVAL_ALPHAS = (0.05, 0.10, 0.20)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== CVaR scenario-count sensitivity: {SEASON} GW{TARGET_GW} ===\n")

    fixture_inputs, players_for_sim, candidates_meta, target_rows = build_players_for_sim()
    rules = scoring.load_scoring_rules("2026_27")
    print("Running base simulation (fixed default seed, reused across every scenario-count test)...")
    sim_results = mc.simulate_gameweek(fixture_inputs, players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    n_sims = len(next(iter(sim_results.values())).samples)
    print(f"Simulations: {n_sims}\n")

    ev_candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in candidates_meta.items()]
    ev_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    ev_xi = sq.select_starting_xi(ev_squad)
    ev_totals_full = np.sum([sim_results[p.player_id].samples for p in ev_xi.starters], axis=0) + sim_results[ev_xi.captain.player_id].samples
    ev_mean = float(ev_totals_full.mean())
    ev_cvars = {a: rb.compute_cvar(ev_totals_full, a) for a in EVAL_ALPHAS}
    print(f"EV squad (fixed baseline): mean={ev_mean:.2f}  " + "  ".join(f"CVaR{a}={v:.2f}" for a, v in ev_cvars.items()) + "\n")

    results = []
    for n_scenarios in SCENARIO_COUNTS:
        for seed_offset in range(N_SEEDS_PER_COUNT):
            seed = 3000 + seed_offset  # distinct from the base simulation's own seed (2026)
            rng = np.random.default_rng(seed)
            scenario_idx = rng.choice(n_sims, size=min(n_scenarios, n_sims), replace=False)
            cvar_candidates = [
                rb.ScenarioPlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].samples[scenario_idx])
                for pid, m in candidates_meta.items()
            ]
            t0 = time.time()
            cvar_squad_sc = rb.select_squad_cvar(cvar_candidates, alpha=CVAR_ALPHA, budget=sq.BUDGET)
            solve_time = time.time() - t0
            cvar_ids = {p.player_id for p in cvar_squad_sc}
            cvar_squad = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in candidates_meta.items() if pid in cvar_ids]
            cvar_xi = sq.select_starting_xi(cvar_squad)
            cvar_totals_full = np.sum([sim_results[p.player_id].samples for p in cvar_xi.starters], axis=0) + sim_results[cvar_xi.captain.player_id].samples
            cvar_mean = float(cvar_totals_full.mean())
            cvar_cvars = {a: rb.compute_cvar(cvar_totals_full, a) for a in EVAL_ALPHAS}

            row = {
                "n_scenarios": n_scenarios, "seed": seed, "solve_time_sec": round(solve_time, 1),
                "cvar_squad_mean": cvar_mean, "cvar_squad_cvars": cvar_cvars,
                "diff_mean": cvar_mean - ev_mean,
                "diff_cvars": {str(a): cvar_cvars[a] - ev_cvars[a] for a in EVAL_ALPHAS},
            }
            results.append(row)
            print(f"n_scenarios={n_scenarios:5d} seed={seed}  solve={solve_time:6.1f}s  "
                  f"mean={cvar_mean:.2f} (diff {row['diff_mean']:+.2f})  " +
                  "  ".join(f"CVaR{a}_diff={row['diff_cvars'][str(a)]:+.2f}" for a in EVAL_ALPHAS))

            out = {"ev_baseline": {"mean": ev_mean, "cvars": {str(a): v for a, v in ev_cvars.items()}}, "sweep": results}
            (ARTIFACT_DIR / "scenario_sensitivity.json").write_text(json.dumps(out, indent=2))  # write incrementally

    print(f"\nWritten to {(ARTIFACT_DIR / 'scenario_sensitivity.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
