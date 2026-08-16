"""Robust and stochastic squad optimization (spec Part XXVI; CVaR is
Phase "out-of-order" work, extended in Phase 8 with a second, structurally
different risk-aware variant).

Compares the deterministic expected-value objective (the existing
apex_fpl.optimization.squad optimizer) against distributionally-aware
objectives, all evaluated on REAL, CORRELATED Monte Carlo scenarios (not
an assumed or fabricated uncertainty set):

1. **CVaR** (Conditional Value at Risk, Rockafellar-Uryasev formulation):
   maximize the average outcome across the worst alpha-fraction of
   scenarios — cares ONLY about the tail, ignores all dispersion above
   it. Evaluated to a final NOT-PROMOTED decision on real multi-season
   data (docs/robust_captaincy_report.md) — real tail-protection signal,
   not yet significant at the decision level.
2. **Mean-variance / MAD** (Phase 8): maximize mean minus a
   risk-aversion-weighted Mean Absolute Deviation — a classic linear
   proxy for Markowitz variance-penalization (Konno & Yamazaki 1991),
   chosen over true variance specifically because variance is quadratic
   in the selection variables (would require a MIQP solver this project
   doesn't have) while MAD stays an EXACT linear MILP, reusing the same
   scipy.optimize.milp/HiGHS backend as everything else in this module.
   Structurally different from CVaR: MAD penalizes deviation on BOTH
   sides of the mean, symmetrically, not just the worst tail — a
   genuinely different notion of "risk," not a re-run of CVaR with a
   different name.

Both are EXACT MILPs alongside the existing squad-legality constraints —
not heuristics or approximations of their respective risk measures. Spec
Part XXVI is explicit that deterministic/robust/distributionally-robust/
stochastic tracks should be compared with evidence deciding the winner,
not assumed a priori — this module builds each as a genuine challenger,
not a replacement presumed to be better.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from apex_fpl.optimization.squad import BUDGET, MAX_PER_CLUB, SQUAD_QUOTAS


@dataclass(frozen=True)
class ScenarioPlayerCandidate:
    player_id: str
    position: str  # GK/DEF/MID/FWD
    team: str
    price: float  # £m
    scenario_points: np.ndarray  # shape (n_scenarios,) — real simulated points, one per scenario, aligned across all players (same scenario index = same simulated gameweek draw)


def compute_cvar(values: np.ndarray, alpha: float) -> float:
    """Direct, unambiguous CVaR: the average of the worst alpha-fraction
    of outcomes (lowest values), by sorting. Used to evaluate ANY squad's
    realized downside — not just the CVaR optimizer's own output — so
    comparisons between the EV and CVaR squads use one consistent
    definition rather than trusting each optimizer's internal objective
    value."""
    values = np.asarray(values, dtype=float)
    k = max(1, int(np.ceil(alpha * len(values))))
    worst = np.sort(values)[:k]
    return float(np.mean(worst))


def select_squad_cvar(
    players: list[ScenarioPlayerCandidate],
    alpha: float = 0.2,
    budget: float = BUDGET,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    return_diagnostics: bool = False,
) -> list[ScenarioPlayerCandidate] | tuple[list[ScenarioPlayerCandidate], dict]:
    """Selects a legal 15-player squad maximizing CVaR_alpha of total
    squad points across scenarios — the average outcome in the worst
    alpha-fraction of simulated gameweeks — rather than the mean.
    alpha=0.2 means "protect the worst 20% of simulated outcomes."

    Rockafellar-Uryasev formulation: introduces one free variable zeta
    (a VaR-like threshold) and one auxiliary variable u_s >= 0 per
    scenario, with u_s >= zeta - points_total_s. Maximizing
    zeta - (1/(alpha*S)) * sum(u_s) is provably equivalent to maximizing
    the exact CVaR — this is standard operations-research theory, not an
    approximation invented for this project.

    time_limit / mip_rel_gap: solve-time bounds for scipy's HiGHS backend.
    Real-data testing (docs/robust_captaincy_report.md) found solve time
    varies enormously and unpredictably by problem instance — some
    gameweeks took under a minute, one took 6+ minutes, and unbounded runs
    were observed taking 15+ minutes on early-season data with less
    differentiated player values (a known MILP phenomenon: many
    near-tied solutions make branch-and-bound search much harder, not a
    bug). A bounded solve accepts scipy's best FEASIBLE solution found
    within the limit rather than the provably optimal one — note this
    checks `res.status in (0, 1)`, NOT `res.success`, because scipy's
    `success` is only True when optimality is proven (status 0); a
    time-limited-but-feasible result (status 1) would otherwise be
    wrongly treated as a failure and raise, discarding a perfectly usable
    solution. Set return_diagnostics=True to see whether a given call hit
    the limit and how far its reported gap was from optimal.
    """
    n = len(players)
    s = len(players[0].scenario_points)
    for p in players:
        if len(p.scenario_points) != s:
            raise ValueError("all players must share the same number of scenarios (same simulate_gameweek() call)")

    points_matrix = np.array([p.scenario_points for p in players])  # (n, s)
    price = np.array([p.price for p in players])

    n_vars = n + s + 1
    zeta_idx = n + s

    c = np.zeros(n_vars)
    c[zeta_idx] = -1.0  # maximize zeta -> minimize -zeta
    c[n:n + s] = 1.0 / (alpha * s)  # minimize (1/(alpha*s)) * sum(u_s)

    constraints = []
    for si in range(s):
        row = np.zeros(n_vars)
        row[:n] = points_matrix[:, si]
        row[n + si] = 1.0
        row[zeta_idx] = -1.0
        constraints.append(LinearConstraint(row, 0, np.inf))  # u_s - zeta + sum(points*x) >= 0

    row = np.zeros(n_vars)
    row[:n] = price
    constraints.append(LinearConstraint(row, -np.inf, budget))

    for pos, q in SQUAD_QUOTAS.items():
        row = np.zeros(n_vars)
        for i, p in enumerate(players):
            if p.position == pos:
                row[i] = 1.0
        constraints.append(LinearConstraint(row, q, q))

    for club in sorted({p.team for p in players}):
        row = np.zeros(n_vars)
        for i, p in enumerate(players):
            if p.team == club:
                row[i] = 1.0
        constraints.append(LinearConstraint(row, -np.inf, MAX_PER_CLUB))

    integrality = np.zeros(n_vars)
    integrality[:n] = 1
    bounds = Bounds(
        lb=np.concatenate([np.zeros(n), np.zeros(s), [-np.inf]]),
        ub=np.concatenate([np.ones(n), np.full(s, np.inf), [np.inf]]),
    )

    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options or None)
    if res.status not in (0, 1) or res.x is None:
        raise RuntimeError(f"CVaR squad optimization infeasible: {res.message}")

    x = np.round(res.x[:n]).astype(int)
    squad = [players[i] for i in range(n) if x[i] == 1]

    if return_diagnostics:
        diagnostics = {
            "status": int(res.status), "proven_optimal": res.status == 0,
            "mip_gap": getattr(res, "mip_gap", None), "message": res.message,
        }
        return squad, diagnostics
    return squad


def compute_mad(values: np.ndarray) -> float:
    """Mean Absolute Deviation from the mean — the direct, unambiguous
    definition, used to evaluate ANY squad's dispersion (not just the
    mean-variance optimizer's own output), matching the same
    "evaluate everyone the same way" discipline as compute_cvar above."""
    values = np.asarray(values, dtype=float)
    return float(np.mean(np.abs(values - values.mean())))


def select_squad_mean_variance(
    players: list[ScenarioPlayerCandidate],
    lambda_risk: float,
    budget: float = BUDGET,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    node_limit: int | None = None,
    return_diagnostics: bool = False,
) -> list[ScenarioPlayerCandidate] | tuple[list[ScenarioPlayerCandidate], dict]:
    """Selects a legal 15-player squad maximizing mean squad points minus
    `lambda_risk` times the Mean Absolute Deviation of squad points across
    scenarios — a linear (MAD) proxy for classic Markowitz mean-variance
    optimization. lambda_risk=0 recovers plain EV selection exactly;
    larger lambda_risk trades expected points for a tighter, more
    predictable distribution (penalizing dispersion on BOTH sides of the
    mean, unlike CVaR's tail-only focus).

    Exact linearization: because both the scenario-total-points function
    and the squad's mean are LINEAR in the selection variables x (mean
    per-player scenario points are fixed constants once scenarios are
    drawn), `dev_s >= points_s(x) - mean(x)` and `dev_s >= mean(x) -
    points_s(x)` are both genuinely linear constraints in x — no
    approximation of the MAD objective itself is involved, only the
    (standard, well-established) choice of MAD over true variance to stay
    solvable by a linear MILP solver.
    """
    n = len(players)
    s = len(players[0].scenario_points)
    for p in players:
        if len(p.scenario_points) != s:
            raise ValueError("all players must share the same number of scenarios (same simulate_gameweek() call)")

    points_matrix = np.array([p.scenario_points for p in players])  # (n, s)
    avg_points = points_matrix.mean(axis=1)  # (n,) — each player's own mean, used as mean(x)'s linear coefficients
    price = np.array([p.price for p in players])

    n_vars = n + s  # x_0..x_{n-1} (select), dev_0..dev_{s-1} (>=0)

    c = np.zeros(n_vars)
    c[:n] = -avg_points  # maximize mean(x) -> minimize -mean(x)
    c[n:n + s] = lambda_risk / s  # minimize lambda_risk * (1/s) * sum(dev_s)

    constraints = []
    for si in range(s):
        row = np.zeros(n_vars)
        row[:n] = avg_points - points_matrix[:, si]  # dev_s >= points_s(x) - mean(x)  <=>  dev_s - (points_s - avg)@x >= 0
        row[n + si] = 1.0
        constraints.append(LinearConstraint(row, 0, np.inf))
        row2 = np.zeros(n_vars)
        row2[:n] = points_matrix[:, si] - avg_points  # dev_s >= mean(x) - points_s(x)
        row2[n + si] = 1.0
        constraints.append(LinearConstraint(row2, 0, np.inf))

    row = np.zeros(n_vars)
    row[:n] = price
    constraints.append(LinearConstraint(row, -np.inf, budget))

    for pos, q in SQUAD_QUOTAS.items():
        row = np.zeros(n_vars)
        for i, p in enumerate(players):
            if p.position == pos:
                row[i] = 1.0
        constraints.append(LinearConstraint(row, q, q))

    for club in sorted({p.team for p in players}):
        row = np.zeros(n_vars)
        for i, p in enumerate(players):
            if p.team == club:
                row[i] = 1.0
        constraints.append(LinearConstraint(row, -np.inf, MAX_PER_CLUB))

    integrality = np.zeros(n_vars)
    integrality[:n] = 1
    bounds = Bounds(
        lb=np.concatenate([np.zeros(n), np.zeros(s)]),
        ub=np.concatenate([np.ones(n), np.full(s, np.inf)]),
    )

    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap
    if node_limit is not None:
        options["node_limit"] = node_limit

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options or None)
    if res.status not in (0, 1) or res.x is None:
        raise RuntimeError(f"mean-variance squad optimization infeasible: {res.message}")

    x = np.round(res.x[:n]).astype(int)
    squad = [players[i] for i in range(n) if x[i] == 1]

    if return_diagnostics:
        diagnostics = {
            "status": int(res.status), "proven_optimal": res.status == 0,
            "mip_gap": getattr(res, "mip_gap", None), "message": res.message,
        }
        return squad, diagnostics
    return squad
