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


@dataclass(frozen=True)
class PlayerCandidate:
    player_id: str
    position: str  # GK/DEF/MID/FWD
    team: str
    price: float  # £m
    expected_points: float


def select_squad(players: list[PlayerCandidate], budget: float = BUDGET) -> list[PlayerCandidate]:
    n = len(players)
    ep = np.array([p.expected_points for p in players])
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
    formation legality (element_types' squad_min_play/squad_max_play)."""
    n = len(squad)
    ep = np.array([p.expected_points for p in squad])
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
