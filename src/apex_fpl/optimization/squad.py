"""Smallest-legal squad optimizer (spec Part XXIV baseline).

Single-gameweek squad selection under budget/quota/club-limit constraints,
maximizing total expected points — the "smallest legal optimizer" the
Phase 2 milestone calls for. No transfers, no chips, no multi-gameweek
horizon, no robust/stochastic objective (those are Phase 7/8 work). Uses
scipy.optimize.milp (HiGHS-backed) rather than OR-Tools/PuLP — scipy
already covers this problem size, avoiding a heavier new dependency
(spec: don't introduce technology merely because it's on an allowed list).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
STARTING_XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI_SIZE = 11
MAX_PER_CLUB = 3
BUDGET = 100.0

# A player is only eligible to be CAPTAINED at full, undoubted
# availability -- not merely "likely." The captain's points are doubled,
# so a well-calibrated 75%-chance doubt is still a much larger expected
# loss as captain than as a squad pick: captaining a doubt is a strictly
# worse error than benching one. See PlayerCandidate.availability_
# probability's own docstring for how this is computed.
CAPTAIN_MIN_AVAILABILITY = 1.0


@dataclass(frozen=True)
class PlayerCandidate:
    player_id: str
    position: str  # GK/DEF/MID/FWD
    team: str
    price: float  # £m
    expected_points: float
    # 1.0 = no doubt (the default, and what every pre-Phase-13-Block-0
    # caller implicitly meant). See apex_fpl.serving.live_data.
    # player_availability_probability for how a real value is derived.
    # select_squad/select_starting_xi both scale on this and gate
    # captaincy eligibility on it -- it is not just informational.
    availability_probability: float = 1.0


def select_squad(players: list[PlayerCandidate], budget: float = BUDGET) -> list[PlayerCandidate]:
    # A player with zero chance of playing contributes nothing no matter
    # how the objective weighs them -- excluded from the candidate pool
    # entirely, not merely down-weighted, so they can never fill a squad
    # slot that a genuinely available player could have taken instead.
    players = [p for p in players if p.availability_probability > 0.0]
    n = len(players)
    ep = np.array([p.expected_points * p.availability_probability for p in players])
    price = np.array([p.price for p in players])

    constraints = [LinearConstraint(price, -np.inf, budget)]
    for pos, q in SQUAD_QUOTAS.items():
        row = np.array([1.0 if p.position == pos else 0.0 for p in players])
        constraints.append(LinearConstraint(row, q, q))
    for club in sorted({p.team for p in players}):
        row = np.array([1.0 if p.team == club else 0.0 for p in players])
        constraints.append(LinearConstraint(row, -np.inf, MAX_PER_CLUB))

    res = milp(c=-ep, constraints=constraints, integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"squad optimization infeasible: {res.message}")
    x = np.round(res.x).astype(int)
    return [players[i] for i in range(n) if x[i] == 1]


@dataclass(frozen=True)
class StartingXI:
    starters: list[PlayerCandidate]
    captain: PlayerCandidate
    bench: list[PlayerCandidate]


def select_starting_xi(squad: list[PlayerCandidate]) -> StartingXI:
    """Given a legal 15-player squad, choose 11 starters + 1 captain
    (points doubled) to maximize total expected points, subject to
    formation legality (element_types' squad_min_play/squad_max_play).

    Availability gates two separate things here, not one: a squad member
    with zero chance of playing can never be started (belt-and-braces --
    select_squad already excludes them from the 15 in the normal path,
    but this function is also called directly elsewhere, e.g. score.py's
    template-team baseline, so it enforces its own floor rather than
    trusting the caller). Captaincy is gated far more strictly, at
    CAPTAIN_MIN_AVAILABILITY (see its own module-level docstring) --
    every real caller in this codebase defaults availability_probability
    to 1.0, so this changes nothing for them."""
    n = len(squad)
    ep = np.array([p.expected_points * p.availability_probability for p in squad])
    n_vars = 2 * n  # [y_0..y_{n-1} (start), c_0..c_{n-1} (captain)]

    c = np.zeros(n_vars)
    c[:n] = -ep
    c[n:] = -ep  # captain's second copy = the doubling bonus

    constraints = []
    row = np.zeros(n_vars); row[:n] = 1.0
    constraints.append(LinearConstraint(row, STARTING_XI_SIZE, STARTING_XI_SIZE))
    row = np.zeros(n_vars); row[n:] = 1.0
    constraints.append(LinearConstraint(row, 1, 1))
    for i in range(n):
        row = np.zeros(n_vars)
        row[i] = -1.0
        row[n + i] = 1.0
        constraints.append(LinearConstraint(row, -np.inf, 0))  # c_i <= y_i
        if squad[i].availability_probability <= 0.0:
            zero_start = np.zeros(n_vars); zero_start[i] = 1.0
            constraints.append(LinearConstraint(zero_start, 0, 0))  # y_i == 0: can never start
        if squad[i].availability_probability < CAPTAIN_MIN_AVAILABILITY:
            zero_cap = np.zeros(n_vars); zero_cap[n + i] = 1.0
            constraints.append(LinearConstraint(zero_cap, 0, 0))  # c_i == 0: can never captain
    for pos in STARTING_XI_MIN:
        row = np.zeros(n_vars)
        for i, p in enumerate(squad):
            if p.position == pos:
                row[i] = 1.0
        constraints.append(LinearConstraint(row, STARTING_XI_MIN[pos], STARTING_XI_MAX[pos]))

    res = milp(c=c, constraints=constraints, integrality=np.ones(n_vars), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"starting XI optimization infeasible: {res.message}")
    x = np.round(res.x).astype(int)
    y, cap = x[:n], x[n:]

    starters = [squad[i] for i in range(n) if y[i] == 1]
    bench = [squad[i] for i in range(n) if y[i] == 0]
    captain = squad[int(np.argmax(cap))]
    return StartingXI(starters, captain, bench)
