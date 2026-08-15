"""Hyperparameter tuning for the attack/defense team model + Dixon-Coles
rho refit (spec Part XLV: search must occur strictly within
training/validation boundaries — never touching the outer test fold this
tournament will ultimately score the model on).

fit_rho() is ported from the World Cup predictor project's
models.py::fit_rho (docs/world_cup_transfer_audit.md classification A).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from apex_fpl.evaluation import metrics as em
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl


def _wdl_from_matrix(m: np.ndarray) -> np.ndarray:
    home = np.tril(m, -1).sum()
    away = np.triu(m, 1).sum()
    draw = np.trace(m)
    return np.array([home, draw, away])


def _predict(model: ad.AttackDefenseModel, fixtures: list[ad.Fixture]):
    lh, la, outcomes = [], [], []
    for f in fixtures:
        if f.home_score is None:
            continue
        eh, ea = model.expected_goals(f.home_team, f.away_team, f.date)
        lh.append(eh)
        la.append(ea)
        outcomes.append("H" if f.home_score > f.away_score else ("A" if f.home_score < f.away_score else "D"))
    return np.array(lh), np.array(la), outcomes


def fit_rho(lh: np.ndarray, la: np.ndarray, home_scores, away_scores, sample: int = 4000, seed: int = 0) -> float:
    """MLE fit of the Dixon-Coles low-score correlation parameter."""
    n = len(lh)
    idx = np.random.RandomState(seed).choice(n, min(sample, n), replace=False)
    lh_s, la_s = lh[idx], la[idx]
    hs_s = np.asarray(home_scores)[idx]
    as_s = np.asarray(away_scores)[idx]

    def negll(rho: float) -> float:
        ll = 0.0
        for l1, l2, i, j in zip(lh_s, la_s, hs_s, as_s):
            i = int(min(i, sl.MAX_GOALS))
            j = int(min(j, sl.MAX_GOALS))
            base = poisson.pmf(i, max(l1, 1e-3)) * poisson.pmf(j, max(l2, 1e-3))
            base *= sl.dc_tau(i, j, l1, l2, rho)
            ll += np.log(max(base, 1e-12))
        return -ll

    res = minimize_scalar(negll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


@dataclass(frozen=True)
class TunedConstants:
    k_base: float
    halflife_days: float
    rho: float
    inner_val_log_loss: float  # the score the winning (k_base, halflife) combo achieved on the inner validation split


def grid_search(
    train_fixtures: list[ad.Fixture],
    val_fixtures: list[ad.Fixture],
    k_base_grid: list[float],
    halflife_grid: list[float],
) -> TunedConstants:
    """Inner-fold hyperparameter search. Fits the attack/defense model on
    train_fixtures under every (k_base, halflife) combination, scores each
    on val_fixtures by log loss of the resulting Poisson-only [rho=0]
    match-outcome probabilities, and keeps the best. rho is then refit
    separately by MLE using the WINNING model's predictions on
    train_fixtures only — val_fixtures and any outer test fold are never
    touched by the rho fit either.
    """
    best = None
    for k_base in k_base_grid:
        for halflife in halflife_grid:
            model = ad.fit(train_fixtures, k_base=k_base, halflife_days=halflife)
            lh, la, outcomes = _predict(model, val_fixtures)
            if len(outcomes) == 0:
                continue
            wdl = np.array([_wdl_from_matrix(sl.score_matrix(h, a, rho=0.0)) for h, a in zip(lh, la)])
            ll = em.log_loss(wdl, outcomes)
            if best is None or ll < best[0]:
                best = (ll, k_base, halflife)

    if best is None:
        raise ValueError("grid_search(): no validation fixtures had a completed score — cannot tune")
    inner_ll, k_base, halflife = best

    final_model = ad.fit(train_fixtures, k_base=k_base, halflife_days=halflife)
    lh, la, _ = _predict(final_model, train_fixtures)
    home_scores = [f.home_score for f in train_fixtures if f.home_score is not None]
    away_scores = [f.away_score for f in train_fixtures if f.away_score is not None]
    rho = fit_rho(lh, la, home_scores, away_scores)

    return TunedConstants(k_base=k_base, halflife_days=halflife, rho=rho, inner_val_log_loss=round(inner_ll, 4))
