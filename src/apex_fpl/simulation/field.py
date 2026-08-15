"""Competitive-field Monte Carlo simulation (Phase 10; spec Parts
XXIII/XXXI/XXXII — "Rank utility / field Monte Carlo").

Rather than modeling millions of individual real rivals (impossible — no
data source exposes other managers' actual squads), this builds a
FIELD OF SYNTHETIC RIVALS drawn from REAL per-player ownership data
(`apex_fpl.field.ownership`): each synthetic rival's squad is sampled
player-by-player, weighted by real historical ownership share within
each position. This is a genuine, data-grounded approximation of the
competitive field's aggregate composition, not a fabricated one — but it
IS an approximation, with explicit, stated limitations:

- Synthetic squads are NOT budget-constrained (ownership alone doesn't
  reconstruct who could afford what) — a real field's squads are
  cheaper on average than pure ownership-weighted sampling would imply,
  since ownership already reflects real managers' budget tradeoffs, but
  this sampling doesn't re-derive that constraint from scratch.
- Real cross-player ownership CORRELATION (e.g. managers who own one
  premium midfielder disproportionately also own a specific premium
  forward, because of shared budget logic or "template" squads) is not
  modeled — each position is sampled independently.
- Each synthetic rival is assumed to pick their starting XI/captain via
  the SAME EV-optimal logic used for the user's own squad throughout
  this project (`apex_fpl.optimization.squad.select_starting_xi`) — not
  a claim that real rivals are all perfectly optimal, but the most
  defensible non-arbitrary default given no data on how sub-optimal a
  typical rival's actual choices are.

Crucially, a synthetic rival's simulated score is computed by summing
their squad's PER-SCENARIO samples from the SAME Monte Carlo simulation
already run for the user's own squad (`apex_fpl.simulation.monte_carlo`)
— not independently redrawn. This correctly correlates the user's score
and the field's score within each simulated scenario (shared players,
shared match outcomes), which is essential for a meaningful rank/
percentile estimate: comparing two INDEPENDENTLY drawn distributions
would overstate uncertainty about relative rank whenever the user and
the field share players (the normal case).
"""
from __future__ import annotations

import numpy as np

from apex_fpl.optimization import squad as sq
from apex_fpl.simulation.monte_carlo import PlayerSimResult


def sample_synthetic_rival_squads(
    ownership_fractions: dict[str, float],
    candidates_meta: dict[str, dict],
    n_rivals: int,
    quotas: dict[str, int] | None = None,
    seed: int = 2026,
) -> list[list[str]]:
    """Draws `n_rivals` synthetic 15-player squads, sampling WITHOUT
    replacement within each squad (a real manager can't own a player
    twice) but WITH replacement ACROSS different rival squads (the same
    highly-owned player legitimately appears in most rivals' squads,
    exactly as in reality), weighted by real ownership share within each
    position."""
    quotas = quotas or sq.SQUAD_QUOTAS
    rng = np.random.default_rng(seed)

    by_position: dict[str, list[str]] = {}
    for pid, meta in candidates_meta.items():
        if pid not in ownership_fractions:
            continue
        by_position.setdefault(meta["position"], []).append(pid)

    weights_by_position: dict[str, np.ndarray] = {}
    for pos, ids in by_position.items():
        w = np.array([max(ownership_fractions[pid], 1e-9) for pid in ids])
        weights_by_position[pos] = w / w.sum()

    for pos, quota in quotas.items():
        if pos not in by_position or len(by_position[pos]) < quota:
            raise ValueError(f"not enough owned candidates at position {pos} to fill a squad (need {quota})")

    squads = []
    for _ in range(n_rivals):
        squad_ids = []
        for pos, quota in quotas.items():
            ids = by_position[pos]
            probs = weights_by_position[pos]
            chosen = rng.choice(len(ids), size=quota, replace=False, p=probs)
            squad_ids.extend(ids[i] for i in chosen)
        squads.append(squad_ids)
    return squads


def simulate_field_scores(
    rival_squads: list[list[str]],
    sim_results: dict[str, PlayerSimResult],
    candidates_meta: dict[str, dict],
) -> np.ndarray:
    """Returns an (n_rivals, n_scenarios) matrix of simulated total
    points, one row per synthetic rival, using the EV-optimal starting
    XI/captain choice and summing REAL per-scenario samples (see module
    docstring for why this must reuse samples rather than redraw)."""
    n_scenarios = len(next(iter(sim_results.values())).samples)
    scores = np.zeros((len(rival_squads), n_scenarios))
    for i, squad_ids in enumerate(rival_squads):
        candidates = [
            sq.PlayerCandidate(pid, candidates_meta[pid]["position"], candidates_meta[pid]["team"], 0.0, sim_results[pid].mean_points)
            for pid in squad_ids
        ]
        xi = sq.select_starting_xi(candidates)
        total = np.sum([sim_results[p.player_id].samples for p in xi.starters], axis=0)
        total = total + sim_results[xi.captain.player_id].samples
        scores[i] = total
    return scores


def my_percentile_per_scenario(my_samples: np.ndarray, field_scores: np.ndarray) -> np.ndarray:
    """For each scenario, the fraction of the synthetic field the user's
    squad beats (a percentile in [0,1]) — computed scenario-by-scenario
    so it correctly reflects correlation between the user and the field
    within each simulated world, not an independent comparison of two
    separately-summarized distributions (e.g. just comparing means)."""
    n_rivals = field_scores.shape[0]
    beaten = (field_scores < my_samples[np.newaxis, :]).sum(axis=0)
    return beaten / n_rivals


def naive_ownership_weighted_mean_score(ownership_fractions: dict[str, float], sim_results: dict[str, PlayerSimResult]) -> float:
    """A simple, INDEPENDENT cross-check for the field Monte Carlo's mean
    score, not a substitute for it: the ownership-weighted sum of each
    player's own mean EP. Two known, PARTIALLY OFFSETTING biases, stated
    honestly rather than corrected for (no data exists to correct them
    properly): it excludes any captaincy bonus (real per-player
    captaincy rates aren't in this dataset, so this under-counts), and it
    doesn't respect any real starting-XI/bench split (ownership doesn't
    say who a manager benches, so this over-counts by including some
    players who were actually benched by their owners). Useful as a
    rough sanity-check magnitude, not a precise ground truth."""
    return float(sum(frac * sim_results[pid].mean_points for pid, frac in ownership_fractions.items() if pid in sim_results))
