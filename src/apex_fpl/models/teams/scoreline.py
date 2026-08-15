"""Dixon-Coles scoreline probability matrix.

Ported from the FIFA World Cup predictor project's models.py (`dc_tau`,
`score_matrix`) — docs/world_cup_transfer_audit.md classification A.
`RHO_DEFAULT` is the World Cup repo's own MLE-fitted value for
international football, used here as a documented placeholder, NOT refit
for club football — refitting on real Premier League data is Phase 4
ablation work.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

RHO_DEFAULT = -0.04
MAX_GOALS = 7


def dc_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1 - lh * la * rho
    if i == 0 and j == 1:
        return 1 + lh * rho
    if i == 1 and j == 0:
        return 1 + la * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def score_matrix(lh: float, la: float, rho: float = RHO_DEFAULT, max_goals: int = MAX_GOALS) -> np.ndarray:
    lh = max(lh, 1e-3)
    la = max(la, 1e-3)
    ph = poisson.pmf(np.arange(max_goals + 1), lh)
    pa = poisson.pmf(np.arange(max_goals + 1), la)
    m = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            m[i, j] *= dc_tau(i, j, lh, la, rho)
    return m / m.sum()


def clean_sheet_prob(m: np.ndarray, side: str) -> float:
    """P(team on `side` concedes 0), side in {'home','away'}."""
    if side == "home":
        return float(m[:, 0].sum())  # away scores 0
    return float(m[0, :].sum())  # home scores 0
