from __future__ import annotations

import numpy as np
import pytest

from apex_fpl.optimization import rank_aware as ra
from apex_fpl.optimization import squad as sq
from apex_fpl.simulation.monte_carlo import PlayerSimResult


def _candidates_meta(n_per_position=6):
    meta = {}
    for pos_i, pos in enumerate(["GK", "DEF", "MID", "FWD"]):
        for j in range(n_per_position):
            pid = f"{pos}{j}"
            meta[pid] = {"position": pos, "team": f"club_{pos_i}_{j}", "price": 5.0}
    return meta


def _sim_results(candidates_meta, ep_by_id=None, n_scenarios=100, seed=0):
    rng = np.random.default_rng(seed)
    ep_by_id = ep_by_id or {}
    results = {}
    for pid in candidates_meta:
        mean = ep_by_id.get(pid, 5.0)
        samples = np.clip(rng.normal(mean, 1.0, n_scenarios), 0, None)
        results[pid] = PlayerSimResult(player_id=pid, mean_points=float(samples.mean()), std_points=float(samples.std()), samples=samples)
    return results


def _player_candidates(meta, sim_results):
    return [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points) for pid, m in meta.items()]


def test_generate_differential_candidates_swaps_the_most_owned_player_first():
    meta = _candidates_meta(n_per_position=6)
    sim_results = _sim_results(meta, seed=1)
    candidates = _player_candidates(meta, sim_results)
    ev_squad = sq.select_squad(candidates, budget=sq.BUDGET)

    # make the highest-EP MID (whichever select_squad actually picked) the most owned
    picked_mid = next(p for p in ev_squad if p.position == "MID")
    ownership = {pid: 0.05 for pid in meta}
    ownership[picked_mid.player_id] = 0.95

    diffs = ra.generate_differential_candidates(ev_squad, candidates, ownership, budget=sq.BUDGET, max_candidates=5, max_ev_loss_fraction=1.0)

    assert diffs  # at least one legal swap should exist given equal prices and generous EV tolerance
    first_squad, out_id, in_id = diffs[0]
    assert out_id == picked_mid.player_id  # most-owned squad member swapped first
    assert in_id != out_id
    assert {p.player_id for p in first_squad} == ({p.player_id for p in ev_squad} - {out_id}) | {in_id}


def test_generate_differential_candidates_respects_ev_loss_tolerance():
    meta = _candidates_meta(n_per_position=6)
    sim_results = _sim_results(meta, seed=2)
    candidates = _player_candidates(meta, sim_results)
    ev_squad = sq.select_squad(candidates, budget=sq.BUDGET)
    ev_squad_ids = {p.player_id for p in ev_squad}

    # every candidate OUTSIDE the squad has near-zero EP -- any swap costs far
    # more than 0.1% of squad EV, so no replacement should ever be accepted.
    starved_ep = {pid: (sim_results[pid].mean_points if pid in ev_squad_ids else 0.01) for pid in meta}
    starved_candidates = [sq.PlayerCandidate(pid, meta[pid]["position"], meta[pid]["team"], meta[pid]["price"], starved_ep[pid]) for pid in meta]
    starved_ev_squad = [c for c in starved_candidates if c.player_id in ev_squad_ids]

    ownership = {pid: (0.95 if pid == ev_squad[0].player_id else 0.05) for pid in meta}
    diffs = ra.generate_differential_candidates(starved_ev_squad, starved_candidates, ownership, budget=sq.BUDGET, max_candidates=5, max_ev_loss_fraction=0.001)

    assert diffs == []  # every legal replacement costs far more than 0.1% of squad EV


def test_generate_differential_candidates_respects_club_limit():
    meta = _candidates_meta(n_per_position=6)
    sim_results = _sim_results(meta, seed=3)

    # hand-construct a legal 15-man squad (not derived from select_squad,
    # so this test controls club composition directly): MID0-2 fill
    # "crowded_club" up to MAX_PER_CLUB, MID4/MID5 are on unique clubs.
    # MID3 (also crowded_club) is left OUT of the squad -- any swap that
    # would bring it in while crowded_club is still at 3 must be rejected.
    for i in range(4):
        meta[f"MID{i}"]["team"] = "crowded_club"
    candidates = _player_candidates(meta, sim_results)
    by_id = {c.player_id: c for c in candidates}
    ev_squad = (
        [by_id[f"GK{i}"] for i in range(2)]
        + [by_id[f"DEF{i}"] for i in range(5)]
        + [by_id["MID0"], by_id["MID1"], by_id["MID2"], by_id["MID4"], by_id["MID5"]]
        + [by_id[f"FWD{i}"] for i in range(3)]
    )
    ownership = {pid: 0.05 for pid in meta}
    ownership["MID3"] = 0.001  # lowest ownership of all -- would sort first if not filtered by the club cap

    diffs = ra.generate_differential_candidates(ev_squad, candidates, ownership, budget=sq.BUDGET, max_candidates=5, max_ev_loss_fraction=1.0)

    for new_squad, out_id, in_id in diffs:
        assert not (in_id == "MID3" and out_id != "MID0" and out_id != "MID1" and out_id != "MID2"), (
            "MID3 (crowded_club) can only legally replace an existing crowded_club member, not a unique-club one"
        )
        club_counts: dict[str, int] = {}
        for p in new_squad:
            club_counts[p.team] = club_counts.get(p.team, 0) + 1
        assert all(count <= sq.MAX_PER_CLUB for count in club_counts.values())


def test_select_rank_aware_squad_always_includes_the_max_ev_squad_as_a_candidate():
    meta = _candidates_meta(n_per_position=6)
    sim_results = _sim_results(meta, seed=4)
    candidates = _player_candidates(meta, sim_results)
    ownership = {pid: 0.5 for pid in meta}

    result = ra.select_rank_aware_squad(candidates, ownership, sim_results, meta, budget=sq.BUDGET, n_rivals=50, seed=1, max_candidates=3)

    assert result.candidates[0].label == "max_ev"
    assert result.candidates[0].swapped_out is None
    ev_squad = sq.select_squad(candidates, budget=sq.BUDGET)
    assert set(result.candidates[0].squad_ids) == {p.player_id for p in ev_squad}


def test_select_rank_aware_squad_selected_maximizes_the_target_metric():
    meta = _candidates_meta(n_per_position=6)
    sim_results = _sim_results(meta, seed=5)
    candidates = _player_candidates(meta, sim_results)
    ownership = {pid: 0.5 for pid in meta}

    result = ra.select_rank_aware_squad(candidates, ownership, sim_results, meta, budget=sq.BUDGET, n_rivals=50, seed=1, max_candidates=3, target_metric="p_top10pct")

    assert result.target_metric == "p_top10pct"
    best_value = max(c.p_top10pct for c in result.candidates)
    assert result.selected.p_top10pct == best_value


def _exact_quota_meta():
    """Exactly SQUAD_QUOTAS candidates per position -- select_squad has
    exactly one legal squad to pick and zero spare candidates for any
    position, so no swap can ever be generated."""
    meta = {}
    for pos, quota in sq.SQUAD_QUOTAS.items():
        for j in range(quota):
            meta[f"{pos}{j}"] = {"position": pos, "team": f"club_{pos}_{j}", "price": 5.0}
    return meta


def test_select_rank_aware_squad_falls_back_to_max_ev_when_no_differentials_exist():
    meta = _exact_quota_meta()
    sim_results = _sim_results(meta, seed=6)
    candidates = _player_candidates(meta, sim_results)
    ownership = {pid: 0.5 for pid in meta}

    result = ra.select_rank_aware_squad(candidates, ownership, sim_results, meta, budget=sq.BUDGET, n_rivals=50, seed=1, max_candidates=3)

    assert len(result.candidates) == 1
    assert result.selected.label == "max_ev"
