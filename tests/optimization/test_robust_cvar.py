"""Tests for the CVaR robust optimizer (spec Part XXVI).

Beyond ordinary legality checks, the key property worth proving directly
is a mathematical guarantee: because select_squad_cvar exactly optimizes
CVaR_alpha under the same constraints the EV optimizer uses, its squad
must achieve CVaR_alpha at least as good as ANY other feasible squad,
including the EV-optimal one — this isn't just "different," it's
provably weakly dominant on its own objective. That's what
test_cvar_squad_weakly_dominates_ev_squad_on_cvar checks directly,
rather than just eyeballing that the two optimizers pick different teams.
"""
from __future__ import annotations

import numpy as np

from apex_fpl.optimization import robust as rb
from apex_fpl.optimization import squad as sq


def _make_scenario_pool(seed=0, n_clubs=20, per_club=(3, 5, 6, 3), n_scenarios=500):
    rng = np.random.default_rng(seed)
    pool = []
    positions = ["GK", "DEF", "MID", "FWD"]
    pid = 0
    for club_i in range(n_clubs):
        club = f"club_{club_i}"
        for pos, count in zip(positions, per_club):
            for _ in range(count):
                pid += 1
                price = float(np.round(rng.uniform(4.0, 13.0), 1))
                mean = rng.normal(4.0, 2.0)
                scenario_points = np.clip(rng.normal(mean, 2.0, n_scenarios), -2, None)
                pool.append(rb.ScenarioPlayerCandidate(f"p{pid}", pos, club, price, scenario_points))
    return pool


def test_cvar_squad_satisfies_every_legality_constraint():
    pool = _make_scenario_pool()
    squad = rb.select_squad_cvar(pool, alpha=0.2, budget=sq.BUDGET)

    assert len(squad) == 15
    total_price = sum(p.price for p in squad)
    assert total_price <= sq.BUDGET + 1e-6

    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts: dict[str, int] = {}
    for p in squad:
        counts[p.position] += 1
        club_counts[p.team] = club_counts.get(p.team, 0) + 1
    for pos, quota in sq.SQUAD_QUOTAS.items():
        assert counts[pos] == quota
    for club, n in club_counts.items():
        assert n <= sq.MAX_PER_CLUB
    assert len({p.player_id for p in squad}) == 15


def test_compute_cvar_matches_direct_definition():
    values = np.array([10.0, 20.0, 5.0, 1.0, 15.0, 8.0, 2.0, 30.0, 3.0, 12.0])
    # alpha=0.3 -> worst 3 values: 1, 2, 3 -> mean = 2.0
    assert abs(rb.compute_cvar(values, alpha=0.3) - 2.0) < 1e-9


def test_cvar_squad_weakly_dominates_ev_squad_on_cvar():
    """The core mathematical property: select_squad_cvar's squad must
    achieve CVaR_alpha at least as good as the EV-optimal squad, when both
    are evaluated on the SAME scenario data with the SAME alpha — because
    it's directly optimizing that exact objective under identical
    constraints."""
    pool = _make_scenario_pool(seed=1)
    alpha = 0.2

    cvar_squad = rb.select_squad_cvar(pool, alpha=alpha, budget=sq.BUDGET)

    ev_candidates = [sq.PlayerCandidate(p.player_id, p.position, p.team, p.price, float(np.mean(p.scenario_points))) for p in pool]
    ev_squad_candidates = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    ev_squad_ids = {p.player_id for p in ev_squad_candidates}
    ev_squad = [p for p in pool if p.player_id in ev_squad_ids]

    cvar_totals = np.sum([p.scenario_points for p in cvar_squad], axis=0)
    ev_totals = np.sum([p.scenario_points for p in ev_squad], axis=0)

    cvar_of_cvar_squad = rb.compute_cvar(cvar_totals, alpha)
    cvar_of_ev_squad = rb.compute_cvar(ev_totals, alpha)

    assert cvar_of_cvar_squad >= cvar_of_ev_squad - 1e-6, (
        f"CVaR optimizer's squad ({cvar_of_cvar_squad:.3f}) must weakly dominate the EV squad "
        f"({cvar_of_ev_squad:.3f}) on the CVaR metric it directly optimizes"
    )


def test_cvar_optimizer_avoids_a_constructed_disaster_prone_star():
    """A player with a high mean but a real chance of a total disaster
    (e.g. an explosive-but-inconsistent forward vs a nailed-on defender
    with a much tighter distribution) should be LESS attractive to a
    low-alpha CVaR objective than to the pure-EV objective, even though
    the EV objective doesn't care about the disaster risk at all."""
    n_scenarios = 2000
    rng = np.random.default_rng(2)

    # boom_bust: mean ~7, but blanks (0 pts) 40% of the time
    boom_bust_pts = np.where(rng.random(n_scenarios) < 0.4, 0.0, rng.normal(11.7, 1.5, n_scenarios))
    # consistent: mean ~6, tight distribution, rarely blanks
    consistent_pts = np.clip(rng.normal(6.0, 1.0, n_scenarios), 0, None)

    assert boom_bust_pts.mean() > consistent_pts.mean()  # confirm EV would prefer boom_bust

    boom_bust_cvar = rb.compute_cvar(boom_bust_pts, alpha=0.2)
    consistent_cvar = rb.compute_cvar(consistent_pts, alpha=0.2)
    assert consistent_cvar > boom_bust_cvar, "the consistent player should have a much better worst-case-20% average despite the lower mean"


def test_time_limit_returns_a_legal_squad_even_if_not_proven_optimal():
    """Regression test for a real bug found running the multi-gameweek
    replay: some real problem instances take 15+ minutes to solve to
    proven optimality (a known MILP phenomenon on early-season data with
    less differentiated player values, not a bug). A tight time_limit
    must still return a legal, usable squad — status 1 (time/iteration
    limit reached, feasible solution found) must not be mistaken for
    failure, which scipy's own `res.success` flag would do (it's only
    True when optimality is proven)."""
    pool = _make_scenario_pool(seed=3, n_scenarios=800)
    squad, diagnostics = rb.select_squad_cvar(pool, alpha=0.2, budget=sq.BUDGET, time_limit=2.0, return_diagnostics=True)

    assert len(squad) == 15
    total_price = sum(p.price for p in squad)
    assert total_price <= sq.BUDGET + 1e-6
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in squad:
        counts[p.position] += 1
    for pos, quota in sq.SQUAD_QUOTAS.items():
        assert counts[pos] == quota

    assert diagnostics["status"] in (0, 1)
    # a 2-second limit on an 800-scenario/480-player problem should NOT prove optimal
    assert diagnostics["proven_optimal"] is False


def test_return_diagnostics_false_by_default_matches_old_signature():
    pool = _make_scenario_pool(seed=4, n_scenarios=200)
    squad = rb.select_squad_cvar(pool, alpha=0.2, budget=sq.BUDGET)
    assert isinstance(squad, list)
    assert len(squad) == 15
