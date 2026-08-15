"""Captaincy risk analysis (spec Part XXVIII).

Uses the real per-simulation samples the Monte Carlo simulator already
produces (src/apex_fpl/simulation/monte_carlo.py) — every probability
below is a genuine empirical frequency across actual simulation draws,
not a fabricated or assumed distribution.

Explicitly NOT computed here (spec Part XXVIII names these too):
effective ownership, expected rank gain/loss. Both require an
ownership/field model (spec Parts XXIII/XXXII) this project has not built
— see docs/fpl_gap_analysis.md. Reporting them would mean fabricating
numbers with no evidence behind them, which the spec explicitly forbids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apex_fpl.simulation.monte_carlo import PlayerSimResult


@dataclass(frozen=True)
class CaptainProfile:
    player_id: str
    mean_points: float
    median_points: float
    std_points: float
    p_blank: float  # P(points <= 1)
    p_2_plus: float
    p_6_plus: float
    p_10_plus: float
    p_15_plus: float
    p_20_plus: float
    p_no_appearance: float


def captain_profile(result: PlayerSimResult) -> CaptainProfile:
    pts = result.samples
    minutes = result.minutes_samples
    return CaptainProfile(
        player_id=result.player_id,
        mean_points=float(np.mean(pts)),
        median_points=float(np.median(pts)),
        std_points=float(np.std(pts)),
        p_blank=float(np.mean(pts <= 1)),
        p_2_plus=float(np.mean(pts >= 2)),
        p_6_plus=float(np.mean(pts >= 6)),
        p_10_plus=float(np.mean(pts >= 10)),
        p_15_plus=float(np.mean(pts >= 15)),
        p_20_plus=float(np.mean(pts >= 20)),
        p_no_appearance=float(np.mean(minutes <= 0)) if minutes is not None else float("nan"),
    )


def captain_bonus_points(captain: PlayerSimResult, vice: PlayerSimResult) -> np.ndarray:
    """Per-simulation EXTRA points contributed by captaincy (the doubling
    bonus — captain's base points are already counted once in the
    starting XI sum elsewhere, this is only the additional copy).

    Applies the real auto-vice-captain fallback rule: if the captain gets
    0 minutes, the vice-captain's points become the bonus instead; if
    the vice-captain ALSO gets 0 minutes, the armband bonus is wasted
    (0), matching official FPL rules rather than assuming the vice always
    saves the day.

    Captain and vice MUST come from the same simulate_gameweek() call so
    their samples share simulation draw indices — this is what lets any
    real correlation between them (same match, same team) come through
    correctly instead of being assumed away by treating them as
    independent."""
    n = len(captain.samples)
    if len(vice.samples) != n:
        raise ValueError("captain and vice-captain must come from the same simulate_gameweek() call (mismatched sample counts)")
    captain_played = captain.minutes_samples > 0
    vice_played = vice.minutes_samples > 0
    return np.where(captain_played, captain.samples, np.where(vice_played, vice.samples, 0.0))


def select_captain_ev(candidates: dict[str, PlayerSimResult]) -> str:
    """Pure expected-value captaincy ('EV mode', spec Part XXVIII) —
    matches the existing squad optimizer's captain choice, exposed here
    for direct comparison against the risk-aware modes below."""
    return max(candidates, key=lambda pid: candidates[pid].mean_points)


def select_captain_risk_averse(candidates: dict[str, PlayerSimResult], quantile: float = 0.25) -> str:
    """'Rank protection' mode: maximize a LOWER quantile of the points
    distribution rather than the mean — favors a safer floor even at a
    slightly lower average, appropriate when protecting an existing lead
    matters more than chasing extra points."""
    return max(candidates, key=lambda pid: float(np.percentile(candidates[pid].samples, quantile * 100)))


def select_captain_ceiling(candidates: dict[str, PlayerSimResult], quantile: float = 0.90) -> str:
    """'Rank chase' / differential mode: maximize a HIGH quantile —
    appropriate when behind and needing a spike to close a rank gap,
    where a high-variance high-ceiling pick can be preferable to a safe
    one even at a lower mean."""
    return max(candidates, key=lambda pid: float(np.percentile(candidates[pid].samples, quantile * 100)))
