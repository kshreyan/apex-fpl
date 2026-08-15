"""Decision-focused learning (Phase 11; spec Part XXVII).

Every model built through Phase 10 is fit to minimize a PREDICTION loss
(log-loss for minutes, Poisson-NLL for attacking allocation, MAE/R2 for
BPS) — the standard "predict-then-optimize" pipeline: fit the best
predictor you can, then feed its output to the squad optimizer. Track A
below IS that pipeline, unchanged.

Decision-focused learning's core idea is that the best predictor for its
OWN loss isn't always the best input to a DOWNSTREAM DECISION — a model
can be tuned to directly maximize decision quality (here: realized squad
points) instead of prediction accuracy. A full implementation would
differentiate through the squad-selection MILP itself (e.g. SPO+,
Elmachtoub & Grigas 2017) — a substantial, fragile undertaking distinct
from everything else built this project. This module implements a
smaller, tractable, well-understood instance of the same idea instead:
**shrinkage**.

Shrinking each player's predicted EP toward their position's median EP
is a classic technique for making a NOISY estimator's downstream
decisions more robust, even though it makes the estimator itself
"worse" by a pointwise-accuracy measure (it deliberately biases
predictions toward the mean). The intuition: an optimizer chasing raw
point estimates over-trusts high-variance/low-sample predictions (a
player who got lucky in a small recent sample looks great by mean EP,
even though that estimate is unreliable) — pulling everyone toward a
robust central value reduces how much the SELECTION decision is driven
by prediction noise specifically. Whether this actually helps FPL squad
selection on real data — rather than just sounding plausible — is
exactly what scripts/run_phase11_decision_focused_tournament.py tests,
following spec Part XXVII's Track A (prediction-focused, unmodified) vs
Track B (decision-focused, shrinkage TUNED directly on realized decision
regret, not prediction loss) vs Track C (a simple hybrid average of the
two) design, evaluated on held-out decision regret across the same 4
independent seasons used throughout this project.

shrinkage=1.0 is the identity transform (Track B collapses to Track A
exactly) — a real correctness property, not just a default.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionFocusedAdjustment:
    shrinkage: float  # 1.0 = no shrinkage (identity), 0.0 = collapse every player to their position's median EP


def apply_shrinkage(ep_by_id: dict[str, float], position_by_id: dict[str, str], shrinkage: float) -> dict[str, float]:
    """adjusted = median_position + shrinkage * (raw - median_position),
    computed independently per position (a GK's EP is only shrunk toward
    the GK median, not pooled across positions with very different point
    scales)."""
    by_position: dict[str, list[float]] = {}
    for pid, ep in ep_by_id.items():
        by_position.setdefault(position_by_id[pid], []).append(ep)
    medians = {pos: statistics.median(vals) for pos, vals in by_position.items()}
    return {pid: medians[position_by_id[pid]] + shrinkage * (ep - medians[position_by_id[pid]]) for pid, ep in ep_by_id.items()}


def hybrid_ep(ep_a: dict[str, float], ep_b: dict[str, float], weight_b: float = 0.5) -> dict[str, float]:
    """Track C: a simple linear blend of two EP dicts over their shared
    player ids."""
    shared = set(ep_a) & set(ep_b)
    return {pid: (1 - weight_b) * ep_a[pid] + weight_b * ep_b[pid] for pid in shared}


def tune_shrinkage(
    tuning_gameweeks: list[tuple[dict[str, dict], dict[str, float], dict[str, int]]],
    shrinkage_grid: list[float],
    select_squad_fn,
    select_starting_xi_fn,
    player_candidate_cls,
) -> float:
    """Picks the shrinkage value maximizing TOTAL REALIZED squad points
    summed across every tuning gameweek — a genuinely decision-focused
    objective (realized decision quality), not a prediction-loss metric.
    Each tuning gameweek is `(candidates_meta, ep_by_id, actual_by_player)`,
    exactly matching the real data this project's replay scripts already
    assemble (see scripts/run_phase11_decision_focused_tournament.py).

    select_squad_fn/select_starting_xi_fn/player_candidate_cls are passed
    in (rather than imported directly) to avoid a decision-focused
    LEARNING module depending on a specific optimizer's dataclass -- the
    caller supplies apex_fpl.optimization.squad's real functions."""
    best_shrinkage, best_total = None, float("-inf")
    for shrinkage in shrinkage_grid:
        total = 0.0
        for candidates_meta, ep_by_id, actual_by_player in tuning_gameweeks:
            position_by_id = {pid: m["position"] for pid, m in candidates_meta.items()}
            adjusted = apply_shrinkage(ep_by_id, position_by_id, shrinkage)
            candidates = [player_candidate_cls(pid, m["position"], m["team"], m["price"], adjusted[pid]) for pid, m in candidates_meta.items() if pid in adjusted]
            squad = select_squad_fn(candidates)
            xi = select_starting_xi_fn(squad)
            total += sum(actual_by_player.get(p.player_id, 0) for p in xi.starters) + actual_by_player.get(xi.captain.player_id, 0)
        if total > best_total:
            best_shrinkage, best_total = shrinkage, total
    return best_shrinkage
