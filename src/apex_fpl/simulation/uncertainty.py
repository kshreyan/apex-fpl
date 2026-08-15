r"""Uncertainty decomposition for simulated player point distributions
(spec Part XVIII).

Spec Part XVIII names eight sources of uncertainty (aleatoric, selection,
minutes, role, parameter, model, data, schedule) and warns against
collapsing everything into a single expected_points number. This module
does NOT attempt all eight — several (role, data, schedule) have no
evidence-backed way to be computed from what this project has built. It
implements the two that can be computed honestly from what the Monte
Carlo simulator (src/apex_fpl/simulation/monte_carlo.py) already produces,
plus a third computed from a genuinely separate source (Phase 4a's two
evaluated team models).

## Selection/minutes vs aleatoric — law of total variance

Var(points) = Var(E[points | played_state]) + E[Var(points | played_state)]
               \_____________________________/   \_______________________/
                "selection/minutes uncertainty":   "aleatoric uncertainty":
                 uncertainty from NOT KNOWING       given the player played
                 whether/how much the player        that much, how random
                 will play at all                   is the scoring outcome

played_state buckets: 0 minutes / 1-59 minutes ("sub appearance") / 60+
minutes — the same three states the minutes model and scoring engine
already distinguish (appearance-points threshold, clean-sheet-eligibility
threshold), so this decomposition uses categories the rest of the
pipeline already treats as meaningful, not an arbitrary binning choice.

## Model (parameter) uncertainty — champion-vs-challenger disagreement

Rather than inventing a "model uncertainty" number with no evidence
behind it, this is computed directly from Phase 4a's actual tournament
result: the disagreement in expected-goals between champion_unfit and
challenger_tuned (both real, evaluated models — docs/phase4_tournament_report.md)
for the same fixture. This directly answers spec Part XVIII's own framing
of model uncertainty: "how much do credible models disagree?"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apex_fpl.simulation.monte_carlo import PlayerSimResult


@dataclass(frozen=True)
class VarianceDecomposition:
    total_variance: float
    selection_minutes_variance: float  # between-played-state variance
    aleatoric_variance: float  # mean of within-played-state variance
    selection_minutes_share: float  # fraction of total variance
    aleatoric_share: float
    state_means: dict  # played_state -> mean points, for inspection
    state_probs: dict  # played_state -> P(that state), for inspection


def _played_state(minutes: np.ndarray) -> np.ndarray:
    """Returns an integer state per sample: 0 = no appearance, 1 = sub
    appearance (1-59 min), 2 = full appearance (60+ min)."""
    return np.where(minutes >= 60, 2, np.where(minutes > 0, 1, 0))


def decompose_variance(result: PlayerSimResult) -> VarianceDecomposition:
    if result.minutes_samples is None:
        raise ValueError("PlayerSimResult has no minutes_samples — re-run simulate_gameweek with the current version")

    points = result.samples
    states = _played_state(result.minutes_samples)
    total_var = float(np.var(points))

    state_means, state_vars, state_probs = {}, {}, {}
    within_state_var_sum = 0.0
    between_state_var = 0.0
    grand_mean = float(np.mean(points))

    for s in (0, 1, 2):
        mask = states == s
        n_s = int(mask.sum())
        if n_s == 0:
            continue
        pts_s = points[mask]
        mean_s = float(np.mean(pts_s))
        var_s = float(np.var(pts_s)) if n_s > 1 else 0.0
        prob_s = n_s / len(points)
        state_means[s] = mean_s
        state_vars[s] = var_s
        state_probs[s] = prob_s
        within_state_var_sum += prob_s * var_s
        between_state_var += prob_s * (mean_s - grand_mean) ** 2

    label_map = {0: "no_appearance", 1: "sub_appearance", 2: "full_appearance"}
    state_means_labelled = {label_map[k]: v for k, v in state_means.items()}
    state_probs_labelled = {label_map[k]: v for k, v in state_probs.items()}

    denom = total_var if total_var > 1e-12 else 1.0
    return VarianceDecomposition(
        total_variance=total_var,
        selection_minutes_variance=between_state_var,
        aleatoric_variance=within_state_var_sum,
        selection_minutes_share=round(between_state_var / denom, 4),
        aleatoric_share=round(within_state_var_sum / denom, 4),
        state_means=state_means_labelled,
        state_probs=state_probs_labelled,
    )


@dataclass(frozen=True)
class ModelDisagreement:
    champion_expected_goals: float
    challenger_expected_goals: float
    absolute_disagreement: float
    relative_disagreement: float  # |diff| / champion value, undefined-safe


def model_disagreement(champion_expected_goals: float, challenger_expected_goals: float) -> ModelDisagreement:
    diff = abs(champion_expected_goals - challenger_expected_goals)
    denom = champion_expected_goals if champion_expected_goals > 1e-9 else 1e-9
    return ModelDisagreement(
        champion_expected_goals=champion_expected_goals,
        challenger_expected_goals=challenger_expected_goals,
        absolute_disagreement=round(diff, 4),
        relative_disagreement=round(diff / denom, 4),
    )
