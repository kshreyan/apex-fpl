#!/usr/bin/env python3
"""Phase 10 real-data demonstration: a competitive-field Monte Carlo and
rank/percentile estimate for one real historical gameweek (2022-23 GW20
— the project's standard single-gameweek demonstration point, matching
Phase 2/Phase 6's precedent).

Builds a field of N_RIVALS synthetic rival squads from REAL 2022-23
GW20 ownership data (apex_fpl.field.ownership), simulates their scores
using the SAME per-player Monte Carlo samples already computed for the
user's own EV-optimal squad (apex_fpl.simulation.field), and reports:
- the user's estimated percentile within the field, per scenario
- the field's mean simulated score, cross-checked against an
  independent naive ownership-weighted estimate (a rough sanity check,
  not a validation against real ground truth — see field.py's
  naive_ownership_weighted_mean_score docstring for its own two
  offsetting biases)

Explicit, unresolved limitation stated honestly rather than glossed
over: there is no real "average_entry_score" or rank-distribution data
in this historical archive to validate the field simulation's absolute
scale against. The internal ownership-weighted cross-check below is the
only sanity check available in this pass — a genuinely independent
validation source (if one can be found) is a clear next step, not
something this demo claims to have already done.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.field import ownership as own
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import field as fld
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase10_field_simulation"
SEASON, TARGET_GW = "2022-23", 20
N_RIVALS = 2000
SEED = 2026


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase 10 field simulation demo: {SEASON} GW{TARGET_GW} ===\n")

    print("--- Building scenario data and running the Monte Carlo simulation ---")
    data = build_gameweek_scenario_data(SEASON, TARGET_GW)
    rules = scoring.load_scoring_rules("2026_27")
    sim_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)
    print(f"Players simulated: {len(sim_results)}")

    print("\n--- Loading real GW20 ownership data ---")
    total_managers = own.estimate_total_managers(SEASON)
    ownership_fractions = own.load_ownership_fractions(SEASON, TARGET_GW, total_managers=total_managers)
    print(f"Estimated total managers: {total_managers:,}")
    top5 = sorted(ownership_fractions.items(), key=lambda kv: -kv[1])[:5]
    for pid, frac in top5:
        name = data.candidates_meta.get(pid, {}).get("name", pid)
        print(f"  {name:<20} {frac * 100:5.1f}% owned")

    print("\n--- Selecting my squad (standard EV optimizer) ---")
    ev_candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in data.candidates_meta.items() if pid in sim_results]
    my_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    my_xi = sq.select_starting_xi(my_squad)
    my_samples = np.sum([sim_results[p.player_id].samples for p in my_xi.starters], axis=0) + sim_results[my_xi.captain.player_id].samples
    print(f"My squad mean simulated score: {my_samples.mean():.2f}")

    print(f"\n--- Sampling {N_RIVALS} synthetic rival squads from real ownership ---")
    rival_squads = fld.sample_synthetic_rival_squads(ownership_fractions, data.candidates_meta, n_rivals=N_RIVALS, seed=SEED)
    field_scores = fld.simulate_field_scores(rival_squads, sim_results, data.candidates_meta)
    print(f"Field mean simulated score (synthetic Monte Carlo): {field_scores.mean():.2f}")

    naive_mean = fld.naive_ownership_weighted_mean_score(ownership_fractions, sim_results)
    print(f"Field mean score (naive ownership-weighted cross-check, no bench/captaincy adjustment): {naive_mean:.2f}")

    print("\n--- My percentile within the field, per scenario ---")
    percentiles = fld.my_percentile_per_scenario(my_samples, field_scores)
    print(f"Mean percentile: {percentiles.mean():.3f}  (0=bottom of the field, 1=beats everyone)")
    print(f"P(top 10% of the field):    {(percentiles >= 0.90).mean():.3f}")
    print(f"P(top 25% of the field):    {(percentiles >= 0.75).mean():.3f}")
    print(f"P(bottom half of the field): {(percentiles < 0.50).mean():.3f}")

    actual_by_player = {r["element"]: int(r["total_points"]) for r in data.target_rows}
    my_actual = sum(actual_by_player.get(p.player_id, 0) for p in my_xi.starters) + actual_by_player.get(my_xi.captain.player_id, 0)
    print(f"\nMy squad's REAL realized GW20 score: {my_actual}")

    summary = {
        "season": SEASON, "target_gw": TARGET_GW, "n_rivals": N_RIVALS,
        "total_managers_estimate": total_managers,
        "my_squad_mean_simulated_score": float(my_samples.mean()),
        "field_mean_simulated_score": float(field_scores.mean()),
        "field_mean_naive_ownership_weighted": naive_mean,
        "my_mean_percentile": float(percentiles.mean()),
        "p_top10pct": float((percentiles >= 0.90).mean()),
        "p_top25pct": float((percentiles >= 0.75).mean()),
        "p_bottom_half": float((percentiles < 0.50).mean()),
        "my_actual_realized_score": my_actual,
        "top5_owned": [{"player_id": pid, "name": data.candidates_meta.get(pid, {}).get("name", pid), "ownership_fraction": frac} for pid, frac in top5],
    }
    (ARTIFACT_DIR / "field_simulation_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'field_simulation_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
