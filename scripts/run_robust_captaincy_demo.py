#!/usr/bin/env python3
"""Demonstrates the uncertainty-sensitive decision machinery (spec Parts
XXVI, XXVIII) on one real historical gameweek: 2022-23 GW20 (matching the
precedent of docs/phase2_milestone_report.md and docs/phase5_calibration_report.md).

Builds the full promoted-champion pipeline (team model, calibrated
minutes, shrinkage attacking allocation, Monte Carlo simulation) exactly
like replay.py, then:
  1. Selects a squad by pure expected value (existing optimizer).
  2. Selects a squad by CVaR (worst-20%-of-scenarios average) using a
     scenario-subsample of the SAME simulation draws — real correlated
     data, not an assumed uncertainty set.
  3. Reports captaincy analysis for both squads' top candidates across
     EV / risk-averse / ceiling selection modes.
  4. Reveals the real GW20 outcome and scores both squads against it —
     explicitly flagged as illustrative (n=1), not statistically
     conclusive, exactly like the original Phase 2 milestone's caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.optimization import captaincy as capt
from apex_fpl.optimization import robust as rb
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "robust_captaincy_demo"
SEASON, TARGET_GW = "2022-23", 20
LOOKBACK = 15
MINUTES_HALFLIFE = 3.0
ATTACKING_ALPHA = 10.0
N_SCENARIOS_FOR_CVAR = 400
CVAR_ALPHA = 0.2
SEED = 2026


def build_players_for_sim():
    """Thin wrapper around the shared scenario builder (see
    src/apex_fpl/backtesting/scenario_builder.py's module docstring for
    why this was consolidated out of a standalone copy here)."""
    data = build_gameweek_scenario_data(
        SEASON, TARGET_GW, lookback=LOOKBACK, minutes_halflife=MINUTES_HALFLIFE, attacking_alpha=ATTACKING_ALPHA,
    )
    return data.fixture_inputs, data.players_for_sim, data.candidates_meta, data.target_rows


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Robust optimization + captaincy risk demo: {SEASON} GW{TARGET_GW} ===\n")

    fixture_inputs, players_for_sim, candidates_meta, target_rows = build_players_for_sim()
    rules = scoring.load_scoring_rules("2026_27")
    print(f"Players entering simulation: {len(players_for_sim)}")
    sim_results = mc.simulate_gameweek(fixture_inputs, players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    n_sims = len(next(iter(sim_results.values())).samples)
    print(f"Simulations run: {n_sims}")

    # ---- EV squad (existing optimizer) ----
    ev_candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in candidates_meta.items()]
    ev_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    ev_xi = sq.select_starting_xi(ev_squad)
    print(f"\nEV squad: {len(ev_squad)} players, cost £{sum(p.price for p in ev_squad):.1f}m, mean EP={sum(p.expected_points for p in ev_squad):.2f}")
    print(f"EV captain: {candidates_meta[ev_xi.captain.player_id]['name']}")

    # ---- CVaR squad (robust optimizer) — scenario subsample for tractability ----
    rng = np.random.default_rng(SEED)
    scenario_idx = rng.choice(n_sims, size=min(N_SCENARIOS_FOR_CVAR, n_sims), replace=False)
    cvar_candidates = [
        rb.ScenarioPlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].samples[scenario_idx])
        for pid, m in candidates_meta.items()
    ]
    print(f"\nSolving CVaR squad (alpha={CVAR_ALPHA}, {len(scenario_idx)} scenarios)...")
    cvar_squad_sc = rb.select_squad_cvar(cvar_candidates, alpha=CVAR_ALPHA, budget=sq.BUDGET)
    cvar_squad_ids = {p.player_id for p in cvar_squad_sc}
    cvar_squad = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in candidates_meta.items() if pid in cvar_squad_ids]
    cvar_xi = sq.select_starting_xi(cvar_squad)
    print(f"CVaR squad: {len(cvar_squad)} players, cost £{sum(p.price for p in cvar_squad):.1f}m, mean EP={sum(p.expected_points for p in cvar_squad):.2f}")
    print(f"CVaR captain: {candidates_meta[cvar_xi.captain.player_id]['name']}")

    overlap = {p.player_id for p in ev_squad} & {p.player_id for p in cvar_squad}
    print(f"\nSquad overlap: {len(overlap)}/15 players in common")

    # ---- downside comparison on FULL simulation set (not just the CVaR subsample) ----
    ev_totals_full = np.sum([sim_results[p.player_id].samples for p in ev_xi.starters], axis=0) \
        + sim_results[ev_xi.captain.player_id].samples
    cvar_totals_full = np.sum([sim_results[p.player_id].samples for p in cvar_xi.starters], axis=0) \
        + sim_results[cvar_xi.captain.player_id].samples

    print("\n=== Downside comparison (evaluated on the FULL simulation set, not the CVaR subsample) ===")
    for alpha in (0.05, 0.1, 0.2):
        ev_cvar = rb.compute_cvar(ev_totals_full, alpha)
        cvar_cvar = rb.compute_cvar(cvar_totals_full, alpha)
        print(f"  alpha={alpha:.2f}: EV squad CVaR={ev_cvar:.2f}  CVaR squad CVaR={cvar_cvar:.2f}  (diff {cvar_cvar - ev_cvar:+.2f})")
    print(f"  Mean: EV squad={ev_totals_full.mean():.2f}  CVaR squad={cvar_totals_full.mean():.2f}  (diff {cvar_totals_full.mean() - ev_totals_full.mean():+.2f})")
    print(f"  Worst simulated outcome: EV squad={ev_totals_full.min():.1f}  CVaR squad={cvar_totals_full.min():.1f}")

    # ---- captaincy analysis ----
    print("\n=== Captaincy risk analysis (top 5 EV squad candidates by mean points) ===")
    starter_ids = [p.player_id for p in ev_xi.starters]
    profiles = {pid: capt.captain_profile(sim_results[pid]) for pid in starter_ids}
    top5 = sorted(profiles.values(), key=lambda p: -p.mean_points)[:5]
    for p in top5:
        name = candidates_meta[p.player_id]["name"]
        print(f"  {name:<22} mean={p.mean_points:5.2f} median={p.median_points:5.2f} std={p.std_points:5.2f} "
              f"p_blank={p.p_blank:.2f} p_10+={p.p_10_plus:.2f} p_20+={p.p_20_plus:.2f} p_no_appearance={p.p_no_appearance:.2f}")

    ev_pick = capt.select_captain_ev({pid: sim_results[pid] for pid in starter_ids})
    risk_averse_pick = capt.select_captain_risk_averse({pid: sim_results[pid] for pid in starter_ids}, quantile=0.25)
    ceiling_pick = capt.select_captain_ceiling({pid: sim_results[pid] for pid in starter_ids}, quantile=0.90)
    print(f"\nCaptain by mode: EV={candidates_meta[ev_pick]['name']}  "
          f"risk_averse(25th pct)={candidates_meta[risk_averse_pick]['name']}  "
          f"ceiling(90th pct)={candidates_meta[ceiling_pick]['name']}")

    # vice-captain joint fallback
    vc_id = sorted(ev_xi.starters, key=lambda p: -p.expected_points)[1].player_id
    bonus = capt.captain_bonus_points(sim_results[ev_pick], sim_results[vc_id])
    print(f"Captain+VC joint simulation: mean captaincy bonus={bonus.mean():.2f} "
          f"(vs captain alone mean={sim_results[ev_pick].samples.mean():.2f}) — "
          f"P(vice-captain activates)={float(np.mean(sim_results[ev_pick].minutes_samples <= 0)):.3f}")

    # ---- reveal actual GW20 results (illustrative only — n=1, not statistically conclusive) ----
    print("\n=== Revealing actual GW20 results (n=1, illustrative only — see docs/phase2_milestone_report.md's caveat) ===")
    actual = {r["element"]: int(r["total_points"]) for r in target_rows}

    def realized(xi):
        return sum(actual.get(p.player_id, 0) for p in xi.starters) + actual.get(xi.captain.player_id, 0)

    ev_realized = realized(ev_xi)
    cvar_realized = realized(cvar_xi)
    print(f"EV squad realized: {ev_realized}")
    print(f"CVaR squad realized: {cvar_realized}")

    summary = {
        "season": SEASON, "target_gw": TARGET_GW,
        "ev_squad": [p.player_id for p in ev_squad], "cvar_squad": [p.player_id for p in cvar_squad],
        "squad_overlap": len(overlap),
        "downside_comparison": {
            f"cvar_alpha_{a}": {"ev_squad": rb.compute_cvar(ev_totals_full, a), "cvar_squad": rb.compute_cvar(cvar_totals_full, a)}
            for a in (0.05, 0.1, 0.2)
        },
        "mean_comparison": {"ev_squad": float(ev_totals_full.mean()), "cvar_squad": float(cvar_totals_full.mean())},
        "captain_modes": {"ev": ev_pick, "risk_averse": risk_averse_pick, "ceiling": ceiling_pick},
        "realized_illustrative": {"ev_squad": ev_realized, "cvar_squad": cvar_realized},
    }
    (ARTIFACT_DIR / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'demo_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
