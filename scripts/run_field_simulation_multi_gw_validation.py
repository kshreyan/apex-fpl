#!/usr/bin/env python3
"""Block 2.8 (Phase 13 promotion schedule) — extends Phase 10's field/
rank simulator (apex_fpl.simulation.field) from its original single-
gameweek demonstration (2022-23 GW20 only) to the same 4-independent-
season, 8-gameweek walk-forward grid used throughout this project
(2020-21/2022-23/2023-24/2024-25, GW7/11/15/19/23/27/31/35) --
this is the FIRST of the two blocking items Block 2.8's own gate named
("far more field-simulation validation" and "a rank-aware selector that
doesn't exist yet"). This addresses the first; the second (a squad
selector optimizing for rank/percentile instead of raw EV) is real,
separate, substantially larger work, deliberately NOT attempted here --
there is no evidence base yet to design that selector's objective
against, which is exactly what this script starts building.

**What this can and cannot resolve.** Phase 10's report already
identified the single most important open item as EXTERNAL validation
against real average-score/rank data -- and already confirmed no such
source exists in this historical archive. That finding is not
re-litigated or re-checked here; it remains a genuine, unresolved gap.
What this DOES extend: (1) the field simulator's internal-consistency
check (its own Monte Carlo field-mean estimate vs. an independently-
computed naive ownership-weighted mean, which agreed within ~1.7% for
the single gameweek Phase 10 originally checked) -- does that agreement
hold systematically across 32 independent gameweeks, or was GW20 a
lucky draw? (2) a new, weaker but real and checkable relationship this
project hasn't looked at before: does the simulated percentile-within-
the-field correlate with the squad's REAL realized score that gameweek
-- not rank (no ground truth for that), but real points, as an
additional sanity signal that the percentile estimate is tracking
something genuine, not noise.

n_rivals reduced from Phase 10's original 2000 to 800 for this 32-
gameweek grid's runtime (a synthetic-field size, not a real-data
input -- timed at ~3.7s/500 rivals locally, well within reach at 800
across 32 observations); Phase 10's own report never claimed 2000 was
a validated-necessary sample size, just what a single demo could afford.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.field import ownership as own
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import field as fld
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "field_simulation_multi_gw_validation"
TEST_SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GAMEWEEKS = [7, 11, 15, 19, 23, 27, 31, 35]
N_RIVALS = 800
SEED = 2026
N_BOOTSTRAP = 5000
ALPHA = 0.05


def run_one_gameweek(season: str, gw: int) -> dict | None:
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None

    rules = scoring.load_scoring_rules("2026_27")
    sim_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)

    total_managers = own.estimate_total_managers(season)
    ownership_fractions = own.load_ownership_fractions(season, gw, total_managers=total_managers)

    ev_candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in data.candidates_meta.items() if pid in sim_results]
    my_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    my_xi = sq.select_starting_xi(my_squad)
    my_samples = np.sum([sim_results[p.player_id].samples for p in my_xi.starters], axis=0) + sim_results[my_xi.captain.player_id].samples

    rival_squads = fld.sample_synthetic_rival_squads(ownership_fractions, data.candidates_meta, n_rivals=N_RIVALS, seed=SEED)
    field_scores = fld.simulate_field_scores(rival_squads, sim_results, data.candidates_meta)
    naive_mean = fld.naive_ownership_weighted_mean_score(ownership_fractions, sim_results)
    percentiles = fld.my_percentile_per_scenario(my_samples, field_scores)

    actual_by_player = {r["element"]: int(r["total_points"]) for r in data.target_rows}
    my_actual = sum(actual_by_player.get(p.player_id, 0) for p in my_xi.starters) + actual_by_player.get(my_xi.captain.player_id, 0)

    field_mc_mean = float(field_scores.mean())
    return {
        "season": season, "gw": gw,
        "field_mc_mean": field_mc_mean, "naive_mean": naive_mean,
        "pct_diff": (field_mc_mean - naive_mean) / naive_mean if naive_mean else None,
        "my_mean_percentile": float(percentiles.mean()),
        "my_squad_real_score": my_actual,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = ALPHA) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(values.mean()), float(np.percentile(boot, lo_pct)), float(np.percentile(boot, hi_pct))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Field simulator multi-gameweek validation: {len(TEST_SEASONS)} seasons x {len(GAMEWEEKS)} gameweeks, n_rivals={N_RIVALS} ===\n")

    rows, blanks = [], []
    for season in TEST_SEASONS:
        for gw in GAMEWEEKS:
            result = run_one_gameweek(season, gw)
            if result is None:
                print(f"{season} GW{gw}: blank gameweek, skipped")
                blanks.append([season, gw])
                continue
            rows.append(result)
            print(f"{season} GW{gw:2d}: field_mc_mean={result['field_mc_mean']:.2f}  naive_mean={result['naive_mean']:.2f}  "
                  f"pct_diff={result['pct_diff']:+.4f}  my_percentile={result['my_mean_percentile']:.3f}  real_score={result['my_squad_real_score']}")

    pct_diffs = np.array([r["pct_diff"] for r in rows])
    mean_pd, lo_pd, hi_pd = bootstrap_ci(pct_diffs)

    percentiles_arr = np.array([r["my_mean_percentile"] for r in rows])
    real_scores_arr = np.array([r["my_squad_real_score"] for r in rows])
    corr = float(np.corrcoef(percentiles_arr, real_scores_arr)[0, 1])

    summary = {
        "n_gameweeks": len(rows), "n_blank": len(blanks), "blanks": blanks, "n_rivals": N_RIVALS,
        "internal_consistency_pct_diff": {"mean": mean_pd, "ci95": [lo_pd, hi_pd]},
        "corr_percentile_vs_real_score": corr,
        "rows": rows,
    }
    (ARTIFACT_DIR / "validation_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ({len(rows)} gameweeks, {len(blanks)} blank) ===")
    print(f"Internal-consistency pct_diff (field_mc_mean vs naive_mean, relative): mean={mean_pd:+.4f}  95% CI [{lo_pd:+.4f}, {hi_pd:+.4f}]")
    print(f"Correlation(my simulated percentile, my squad's real realized score): {corr:.3f}")
    print(f"\nWritten to {(ARTIFACT_DIR / 'validation_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
