"""Gameweek Monte Carlo simulation (spec Parts XVI-XIX), Phase 2 baseline scope.

Simulates, per gameweek, for every player: minutes (from the naive
start-rate model), goals/assists (from the proportional-allocation
model conditioned on the team model's simulated scoreline), and clean
sheets (from that same simulated scoreline) — then scores each simulated
outcome with the deterministic FPL scoring engine.

What this baseline deliberately does NOT do (all later phases, per
research/research_plan.md):
  - No BPS/bonus simulation (Phase 6, needs event-level data we don't have).
  - No defensive contributions (same reason).
  - No saves/cards/own-goals/penalties simulation (would need shot- and
    foul-level match data this baseline has no source for).
  - No cross-player correlation beyond what's already implied by sharing
    one team's simulated scoreline (e.g. two teammates' clean sheets are
    correlated because they share a match draw; two players' goals are
    NOT modelled as competing for a fixed team-goal total — Part XVI's
    full joint/BPS-consistent simulation is Phase 6 work).

Convergence (spec Part XIX): simulates in batches, stopping when the
largest change in any player's mean simulated points falls below `tol`
between batches, rather than picking a simulation count arbitrarily.

Performance: scoring is computed as vectorized numpy array arithmetic
across an entire batch for a player at once (not a Python-level call to
score_player_gameweek per sample) — see
tests/unit/test_monte_carlo.py::test_vectorized_scoring_matches_scoring_engine
for the property test proving this fast path never drifts from the
authoritative deterministic scoring engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apex_fpl.models.minutes.baseline import MinutesForecast

POSITION_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


@dataclass(frozen=True)
class PlayerInput:
    player_id: str
    team: str
    position: str
    minutes_forecast: MinutesForecast
    expected_goals: float
    expected_assists: float


@dataclass(frozen=True)
class FixtureInput:
    home_team: str
    away_team: str
    score_matrix: np.ndarray  # from apex_fpl.models.teams.scoreline.score_matrix


@dataclass
class PlayerSimResult:
    player_id: str
    mean_points: float
    std_points: float
    samples: np.ndarray = field(repr=False)
    minutes_samples: np.ndarray = field(repr=False, default=None)  # parallel array; needed for uncertainty decomposition (spec Part XVIII)


def _vectorized_points(position: str, minutes: np.ndarray, goals: np.ndarray, assists: np.ndarray,
                        clean_sheet: np.ndarray, goals_conceded: np.ndarray, rules: dict) -> np.ndarray:
    """Must stay exactly equivalent to scoring.score_player_gameweek for
    every combination of inputs — proven by
    tests/unit/test_monte_carlo.py::test_vectorized_scoring_matches_scoring_engine.
    In particular this function does NOT trust the caller to have already
    gated `clean_sheet` by the minutes threshold (score_player_gameweek
    doesn't trust its caller either — it checks minutes internally), so it
    re-checks that here rather than assuming simulate_gameweek's own
    pre-gating is the only caller that will ever exist."""
    pos = POSITION_MAP[position]
    pts = np.where(minutes >= 60, rules["appearance"]["at_least_60_min"],
           np.where(minutes > 0, rules["appearance"]["under_60_min"], 0)).astype(float)
    pts += goals * rules["goals_scored"][pos]
    pts += assists * rules["assist"]
    cs_eligible = clean_sheet & (minutes >= rules["clean_sheet_min_minutes"])
    pts += np.where(cs_eligible, rules["clean_sheet"][pos], 0)
    gc_rate = rules["goals_conceded"][pos]
    if gc_rate != 0:
        pts += (goals_conceded // rules["goals_conceded_divisor"]) * gc_rate
    return pts


def simulate_gameweek(
    fixtures: list[FixtureInput],
    players: list[PlayerInput],
    rules: dict,
    tol: float = 0.05,
    batch: int = 2000,
    max_sims: int = 50000,
    seed: int = 2026,
) -> dict[str, PlayerSimResult]:
    rng = np.random.default_rng(seed)
    team_fixture = {}
    for fx in fixtures:
        team_fixture[fx.home_team] = (fx, "home")
        team_fixture[fx.away_team] = (fx, "away")

    accum: dict[str, list[np.ndarray]] = {p.player_id: [] for p in players}
    minutes_accum: dict[str, list[np.ndarray]] = {p.player_id: [] for p in players}
    prev_means: dict[str, float] | None = None
    total_sims = 0

    while total_sims < max_sims:
        n = batch
        fixture_draws = {}
        for fx in fixtures:
            flat = fx.score_matrix.flatten()
            idx = rng.choice(len(flat), size=n, p=flat)
            gh = idx // fx.score_matrix.shape[1]
            ga = idx % fx.score_matrix.shape[1]
            fixture_draws[id(fx)] = (gh, ga)

        for p in players:
            fx, side = team_fixture[p.team]
            gh, ga = fixture_draws[id(fx)]
            goals_against = ga if side == "home" else gh

            p_app = max(p.minutes_forecast.p_appearance, 1e-6)
            u = rng.random(n)
            minutes = np.where(
                u < p.minutes_forecast.p_60_plus, p.minutes_forecast.expected_minutes_if_played,
                np.where(u < p.minutes_forecast.p_appearance, rng.uniform(1, 59, n), 0.0),
            )
            played = minutes > 0

            lam_g = p.expected_goals / p_app
            lam_a = p.expected_assists / p_app
            goals = np.where(played, rng.poisson(lam_g, n), 0)
            assists = np.where(played, rng.poisson(lam_a, n), 0)
            clean_sheet = played & (minutes >= 60) & (goals_against == 0)
            goals_conceded = np.where(played, goals_against, 0)

            pts = _vectorized_points(p.position, minutes, goals, assists, clean_sheet, goals_conceded, rules)
            accum[p.player_id].append(pts)
            minutes_accum[p.player_id].append(minutes)

        total_sims += n
        means_now = {pid: float(np.mean(np.concatenate(v))) for pid, v in accum.items()}
        if prev_means is not None:
            max_delta = max(abs(means_now[pid] - prev_means[pid]) for pid in means_now)
            if max_delta < tol:
                prev_means = means_now
                break
        prev_means = means_now

    results = {}
    for pid, chunks in accum.items():
        samples = np.concatenate(chunks)
        minutes_samples = np.concatenate(minutes_accum[pid])
        results[pid] = PlayerSimResult(pid, float(samples.mean()), float(samples.std()), samples, minutes_samples)
    return results
