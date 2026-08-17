"""Independent legality verification for the squad optimizer (spec Part
XXIV / LXIV: "Every output is legal... independent legality tests pass").

Builds a realistic synthetic player pool (20 clubs x enough players per
position) and checks every constraint by direct inspection of the
optimizer's output — not by trusting the optimizer's own success flag.
"""
from __future__ import annotations

import numpy as np

from apex_fpl.optimization import squad as sq


def _make_pool(seed=0, n_clubs=20, per_club=(3, 5, 6, 3)):
    """per_club: (n_GK, n_DEF, n_MID, n_FWD) generated per club."""
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
                ep = float(np.round(rng.normal(4.0, 2.0), 2))
                pool.append(sq.PlayerCandidate(f"p{pid}", pos, club, price, max(ep, -2.0)))
    return pool


def test_selected_squad_satisfies_every_constraint():
    pool = _make_pool()
    squad = sq.select_squad(pool, budget=sq.BUDGET)

    assert len(squad) == 15
    total_price = sum(p.price for p in squad)
    assert total_price <= sq.BUDGET + 1e-6, f"budget violated: {total_price} > {sq.BUDGET}"

    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts: dict[str, int] = {}
    for p in squad:
        counts[p.position] += 1
        club_counts[p.team] = club_counts.get(p.team, 0) + 1

    for pos, quota in sq.SQUAD_QUOTAS.items():
        assert counts[pos] == quota, f"position {pos}: expected {quota}, got {counts[pos]}"

    for club, n in club_counts.items():
        assert n <= sq.MAX_PER_CLUB, f"club {club} has {n} players, exceeds max {sq.MAX_PER_CLUB}"

    assert len({p.player_id for p in squad}) == 15, "duplicate player selected"


def test_starting_xi_satisfies_formation_and_captain_legality():
    pool = _make_pool()
    squad = sq.select_squad(pool, budget=sq.BUDGET)
    xi = sq.select_starting_xi(squad)

    assert len(xi.starters) == sq.STARTING_XI_SIZE
    assert len(xi.bench) == 15 - sq.STARTING_XI_SIZE

    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in xi.starters:
        counts[p.position] += 1
    for pos in sq.STARTING_XI_MIN:
        assert sq.STARTING_XI_MIN[pos] <= counts[pos] <= sq.STARTING_XI_MAX[pos], (
            f"{pos}: {counts[pos]} starters violates [{sq.STARTING_XI_MIN[pos]}, {sq.STARTING_XI_MAX[pos]}]"
        )
    assert counts["GK"] == 1, "exactly one starting goalkeeper is required"

    assert xi.captain in xi.starters, "captain must be among the starting XI"
    starter_ids = {p.player_id for p in xi.starters}
    bench_ids = {p.player_id for p in xi.bench}
    assert starter_ids.isdisjoint(bench_ids)
    assert starter_ids | bench_ids == {p.player_id for p in squad}


def test_captain_is_the_highest_expected_points_starter():
    """Given captaincy just doubles points with no other constraint, the
    optimizer should always captain whoever it started with the highest
    expected points — verify this holds, not just assume the MILP got it right."""
    pool = _make_pool(seed=1)
    squad = sq.select_squad(pool, budget=sq.BUDGET)
    xi = sq.select_starting_xi(squad)
    best_starter = max(xi.starters, key=lambda p: p.expected_points)
    assert xi.captain.player_id == best_starter.player_id


def test_infeasible_budget_raises_rather_than_returning_illegal_squad():
    pool = _make_pool()
    try:
        sq.select_squad(pool, budget=1.0)  # impossibly small budget
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "an infeasible budget must raise, not silently return an over-budget squad"


# ---------------------------------------------------------- availability --
# Phase 13 Block 0: the correctness bug this project's own site named as
# its highest priority -- an unavailable player must never be selectable,
# and captaincy must be gated stricter than squad inclusion. See
# apex_fpl.serving.live_data.player_availability_probability's own
# docstring for the real-data verification behind these defaults.

def _make_pool_with_availability(seed=0, n_clubs=20, per_club=(3, 5, 6, 3), overrides=None):
    """Same shape as _make_pool, but lets specific player_ids get a
    non-default availability_probability via `overrides` (player_id ->
    probability), and gives the override target an inflated expected_points
    so any test failure (them getting picked/captained anyway) can't be
    explained by them being a weak candidate in the first place."""
    rng = np.random.default_rng(seed)
    overrides = overrides or {}
    pool = []
    positions = ["GK", "DEF", "MID", "FWD"]
    pid = 0
    for club_i in range(n_clubs):
        club = f"club_{club_i}"
        for pos, count in zip(positions, per_club):
            for _ in range(count):
                pid += 1
                player_id = f"p{pid}"
                price = float(np.round(rng.uniform(4.0, 13.0), 1))
                ep = float(np.round(rng.normal(4.0, 2.0), 2))
                availability = overrides.get(player_id, 1.0)
                if player_id in overrides:
                    ep = 50.0  # would dominate every selection if availability weren't gating it
                pool.append(sq.PlayerCandidate(player_id, pos, club, price, max(ep, -2.0), availability))
    return pool


def test_select_squad_never_includes_a_zero_availability_player():
    pool = _make_pool_with_availability(overrides={"p1": 0.0})
    squad = sq.select_squad(pool, budget=sq.BUDGET)
    assert "p1" not in {p.player_id for p in squad}


def test_select_squad_includes_a_doubtful_player_scaled_not_excluded():
    """A genuine, quantified doubt (not zero) is a real, if discounted,
    candidate -- it should still show up if its raw expected_points is
    high enough to clear the discount, proving the doubt is scaled into
    the objective rather than filtered out like a zero-chance player."""
    pool = _make_pool_with_availability(overrides={"p1": 0.75})
    squad = sq.select_squad(pool, budget=sq.BUDGET)
    assert "p1" in {p.player_id for p in squad}


def test_select_starting_xi_never_starts_a_zero_availability_squad_member():
    """Defense in depth: even if a zero-availability player somehow ends
    up in the 15 (select_starting_xi is called directly elsewhere, e.g.
    score.py's template-team baseline, bypassing select_squad's own
    filter), they must never be selected to start. Build a normal legal
    15 first, then swap ONE player for a same-position zero-availability
    candidate with an inflated expected_points -- preserves formation
    feasibility (still enough real candidates per position to fill
    STARTING_XI_MIN/MAX) while proving availability, not just weak
    expected_points, is what keeps them out."""
    pool = _make_pool(seed=2)
    squad = sq.select_squad(pool, budget=sq.BUDGET)
    target = squad[0]
    unavailable = sq.PlayerCandidate("p_unavailable", target.position, target.team, target.price, 999.0, 0.0)
    squad = [unavailable] + list(squad[1:])
    xi = sq.select_starting_xi(squad)
    assert unavailable.player_id not in {p.player_id for p in xi.starters}
    assert unavailable.player_id != xi.captain.player_id


def test_select_starting_xi_never_captains_anything_below_full_availability():
    """A quantified doubt (0.75) is startable but must never be captained
    — the whole point of CAPTAIN_MIN_AVAILABILITY. Give it an
    unambiguously dominant expected_points so the only thing that could
    stop it being captained is the availability gate itself."""
    pool = _make_pool(seed=3)
    squad = sq.select_squad(pool, budget=sq.BUDGET)
    doubtful = squad[0]
    squad = [sq.PlayerCandidate(doubtful.player_id, doubtful.position, doubtful.team, doubtful.price, 999.0, 0.75)] + list(squad[1:])
    xi = sq.select_starting_xi(squad)
    assert xi.captain.player_id != doubtful.player_id


def test_captain_min_availability_is_exactly_full_no_doubt_at_all():
    """Locks in the specific threshold: CAPTAIN_MIN_AVAILABILITY is 1.0,
    not some looser 'high enough' probability -- captaining any flagged
    doubt, however slight, is out of scope for this project's captaincy
    logic. If this constant ever changes, this test should force a
    conscious update, not silently keep passing."""
    assert sq.CAPTAIN_MIN_AVAILABILITY == 1.0
