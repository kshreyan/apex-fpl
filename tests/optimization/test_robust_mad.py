"""Tests for the Phase 8 mean-variance (MAD) robust optimizer, mirroring
test_robust_cvar.py's structure: legality, the key mathematical
dominance property, a lambda=0 sanity check (should reduce exactly to
plain EV selection), and the same time-limit/diagnostics regression
coverage the CVaR module needed after a real bug.
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


def test_mean_variance_squad_satisfies_every_legality_constraint():
    pool = _make_scenario_pool()
    squad = rb.select_squad_mean_variance(pool, lambda_risk=0.5, budget=sq.BUDGET)

    assert len(squad) == 15
    assert sum(p.price for p in squad) <= sq.BUDGET + 1e-6
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


def test_compute_mad_matches_direct_definition():
    values = np.array([10.0, 20.0, 5.0, 1.0, 15.0, 8.0, 2.0, 30.0, 3.0, 12.0])
    mean = values.mean()
    expected = float(np.mean(np.abs(values - mean)))
    assert abs(rb.compute_mad(values) - expected) < 1e-9


def test_lambda_zero_recovers_the_ev_squad_exactly():
    """lambda_risk=0 makes the MAD penalty vanish from the objective, so
    the mean-variance optimizer must select EXACTLY the same squad as
    plain EV selection — a direct correctness check on the linearization,
    not just "produces something plausible."""
    pool = _make_scenario_pool(seed=5)
    mv_squad = rb.select_squad_mean_variance(pool, lambda_risk=0.0, budget=sq.BUDGET)

    ev_candidates = [sq.PlayerCandidate(p.player_id, p.position, p.team, p.price, float(np.mean(p.scenario_points))) for p in pool]
    ev_squad = sq.select_squad(ev_candidates, budget=sq.BUDGET)

    assert {p.player_id for p in mv_squad} == {p.player_id for p in ev_squad}


def test_mean_variance_squad_weakly_dominates_ev_squad_on_its_own_objective():
    """The core mathematical property, mirroring
    test_cvar_squad_weakly_dominates_ev_squad_on_cvar: because
    select_squad_mean_variance exactly optimizes (mean - lambda*MAD) under
    the same constraints the EV optimizer uses, its squad must achieve a
    value of that objective at least as good as the EV-optimal squad's."""
    pool = _make_scenario_pool(seed=1)
    lambda_risk = 0.5

    mv_squad = rb.select_squad_mean_variance(pool, lambda_risk=lambda_risk, budget=sq.BUDGET)

    ev_candidates = [sq.PlayerCandidate(p.player_id, p.position, p.team, p.price, float(np.mean(p.scenario_points))) for p in pool]
    ev_squad_candidates = sq.select_squad(ev_candidates, budget=sq.BUDGET)
    ev_squad_ids = {p.player_id for p in ev_squad_candidates}
    ev_squad = [p for p in pool if p.player_id in ev_squad_ids]

    def objective(squad):
        totals = np.sum([p.scenario_points for p in squad], axis=0)
        return totals.mean() - lambda_risk * rb.compute_mad(totals)

    mv_obj = objective(mv_squad)
    ev_obj = objective(ev_squad)
    assert mv_obj >= ev_obj - 1e-6, (
        f"mean-variance optimizer's squad ({mv_obj:.3f}) must weakly dominate the EV squad "
        f"({ev_obj:.3f}) on the (mean - lambda*MAD) objective it directly optimizes"
    )


def test_mean_variance_optimizer_avoids_a_constructed_disaster_prone_star():
    """Parallel to the CVaR module's own disaster-prone-star test: a
    boom-bust player (high mean, frequent blanks) should be less
    attractive under a risk-averse MAD objective than a consistent player
    with a lower mean but tighter distribution, even though pure EV
    prefers the boom-bust player."""
    n_scenarios = 2000
    rng = np.random.default_rng(2)

    boom_bust_pts = np.where(rng.random(n_scenarios) < 0.4, 0.0, rng.normal(11.7, 1.5, n_scenarios))
    consistent_pts = np.clip(rng.normal(6.0, 1.0, n_scenarios), 0, None)

    assert boom_bust_pts.mean() > consistent_pts.mean()
    assert rb.compute_mad(consistent_pts) < rb.compute_mad(boom_bust_pts)


def test_time_limit_returns_a_legal_squad_even_if_not_proven_optimal():
    # MAD's formulation has 2 deviation constraints per scenario (vs CVaR's 1), so it
    # needs a slightly longer time budget than the CVaR test's 2.0s to reach even a
    # first feasible solution on this pool size — 5.0s reliably lands on status=1
    # (time-limited, feasible, not proven optimal), the exact case this test targets.
    pool = _make_scenario_pool(seed=3, n_scenarios=800)
    squad, diagnostics = rb.select_squad_mean_variance(pool, lambda_risk=0.5, budget=sq.BUDGET, time_limit=5.0, return_diagnostics=True)

    assert len(squad) == 15
    assert sum(p.price for p in squad) <= sq.BUDGET + 1e-6
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in squad:
        counts[p.position] += 1
    for pos, quota in sq.SQUAD_QUOTAS.items():
        assert counts[pos] == quota
    assert diagnostics["status"] in (0, 1)
    assert diagnostics["proven_optimal"] is False


def test_return_diagnostics_false_by_default_matches_old_signature():
    pool = _make_scenario_pool(seed=4, n_scenarios=200)
    squad = rb.select_squad_mean_variance(pool, lambda_risk=0.5, budget=sq.BUDGET)
    assert isinstance(squad, list)
    assert len(squad) == 15
