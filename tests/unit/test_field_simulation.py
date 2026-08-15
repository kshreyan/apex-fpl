from __future__ import annotations

import numpy as np
import pytest

from apex_fpl.simulation import field as fld
from apex_fpl.simulation.monte_carlo import PlayerSimResult

POSITIONS = ["GK", "GK", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"]


def _candidates_meta(n_per_position=20):
    meta = {}
    for pos_i, pos in enumerate(["GK", "DEF", "MID", "FWD"]):
        for j in range(n_per_position):
            pid = f"{pos}{j}"
            meta[pid] = {"position": pos, "team": f"club_{pos_i}_{j % 5}"}
    return meta


def _sim_results(candidates_meta, n_scenarios=200, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    for pid in candidates_meta:
        mean = rng.uniform(2.0, 8.0)
        samples = np.clip(rng.normal(mean, 2.0, n_scenarios), 0, None)
        results[pid] = PlayerSimResult(player_id=pid, mean_points=float(samples.mean()), std_points=float(samples.std()), samples=samples)
    return results


def test_sample_synthetic_rival_squads_respects_quotas_and_no_duplicates():
    meta = _candidates_meta()
    ownership = {pid: 1.0 for pid in meta}  # uniform, so any valid draw is equally likely
    squads = fld.sample_synthetic_rival_squads(ownership, meta, n_rivals=20, seed=1)

    assert len(squads) == 20
    for squad in squads:
        assert len(squad) == 15
        assert len(set(squad)) == 15  # no duplicates within one squad
        counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for pid in squad:
            counts[meta[pid]["position"]] += 1
        assert counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def test_sampling_is_weighted_by_ownership():
    meta = _candidates_meta(n_per_position=10)
    ownership = {pid: 0.001 for pid in meta}
    star_mid = "MID0"
    ownership[star_mid] = 0.95  # near-universal ownership
    squads = fld.sample_synthetic_rival_squads(ownership, meta, n_rivals=500, seed=2)

    appearance_rate = sum(1 for squad in squads if star_mid in squad) / len(squads)
    assert appearance_rate > 0.85  # should appear in nearly every squad, given 95% real ownership


def test_insufficient_owned_candidates_raises():
    meta = {"GK0": {"position": "GK", "team": "clubA"}}  # only 1 GK candidate, quota needs 2
    with pytest.raises(ValueError):
        fld.sample_synthetic_rival_squads({"GK0": 1.0}, meta, n_rivals=5, seed=1)


def test_simulate_field_scores_matches_my_own_squad_when_rival_squad_is_identical():
    meta = _candidates_meta(n_per_position=1)  # exactly 4 candidates: GK0, DEF0, MID0, FWD0 -- not enough for a real squad, use a full pool instead
    meta = _candidates_meta(n_per_position=20)
    results = _sim_results(meta, seed=3)

    my_squad_ids = [f"GK{i}" for i in range(2)] + [f"DEF{i}" for i in range(5)] + [f"MID{i}" for i in range(5)] + [f"FWD{i}" for i in range(3)]
    field_scores = fld.simulate_field_scores([my_squad_ids], results, meta)

    from apex_fpl.optimization import squad as sq
    candidates = [sq.PlayerCandidate(pid, meta[pid]["position"], meta[pid]["team"], 0.0, results[pid].mean_points) for pid in my_squad_ids]
    xi = sq.select_starting_xi(candidates)
    expected = np.sum([results[p.player_id].samples for p in xi.starters], axis=0) + results[xi.captain.player_id].samples

    assert np.allclose(field_scores[0], expected)


def test_my_percentile_per_scenario_correctness():
    # 2 scenarios, 4 rivals
    field_scores = np.array([
        [10.0, 50.0],
        [20.0, 60.0],
        [30.0, 70.0],
        [40.0, 80.0],
    ])
    my_samples = np.array([25.0, 65.0])  # beats 2 of 4 rivals in scenario 0 (10,20), 2 of 4 in scenario 1 (50,60)
    pct = fld.my_percentile_per_scenario(my_samples, field_scores)
    assert np.allclose(pct, [0.5, 0.5])


def test_naive_ownership_weighted_mean_score_is_a_simple_weighted_sum():
    meta = _candidates_meta(n_per_position=2)
    results = _sim_results(meta, seed=4)
    ownership = {pid: 0.5 for pid in meta}
    expected = sum(0.5 * r.mean_points for r in results.values())
    assert abs(fld.naive_ownership_weighted_mean_score(ownership, results) - expected) < 1e-6
