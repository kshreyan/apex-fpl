"""Joint, within-match Monte Carlo simulation with correlated goal/assist
allocation and ranked BPS-based bonus points (spec Parts XIV, XVI).

A challenger to the Phase 2 baseline simulator
(src/apex_fpl/simulation/monte_carlo.py) — kept separate rather than
modifying that well-tested, production-used module in place. Two
structural improvements:

1. Player goals/assists are no longer independent per-player Poisson
   draws. Each scenario allocates the TEAM's own simulated goal total
   multinomially across its players (spec Part XV: "Maintain consistency:
   sum(player_goals) == team_goals inside every simulated match" — the
   baseline simulator does not guarantee this; this one does, by
   construction — see test_joint_simulator.py's reconciliation test).
   Assists are allocated per-goal, conditional on who scored, creating a
   genuine positive scorer-assister correlation.

2. Bonus points are no longer absent (a documented Phase 2 gap). Each
   player's BPS is predicted by the reduced-form model in
   apex_fpl.models.bonus.bps_model, with residual noise added per
   scenario, then RANKED jointly among all match participants using the
   official FPL tie-break rule (ties share points upward — a 2-way tie
   for 1st both score 3, the next player gets 1, not 2 — verified against
   premierleague.com/en/news/106533, not assumed from memory). This makes
   bonus a genuinely zero-sum-like within-match competition, matching
   spec Part XIV directly, rather than an independent per-player draw.

Cost of both improvements: per-scenario Python-level allocation loops
(not fully vectorized like the baseline), so this module targets a
smaller, FIXED scenario count rather than convergence-based stopping — a
deliberate, documented simplification, not an oversight.

What this still does NOT do (same documented gaps as the baseline):
saves/cards/own-goals/penalties are not simulated as stochastic events
(their BPS contribution is fixed at 0 in the feature vector fed to the
BPS model, since we have no shot/foul-level source to simulate them
from).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apex_fpl.models.bonus.bps_model import FEATURE_NAMES, PositionBPSModel
from apex_fpl.models.minutes.baseline import MinutesForecast

POSITION_MAP = {"GK": "GK", "GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
P_GOAL_ASSISTED_DEFAULT = 0.68  # documented estimate; see fit_p_goal_assisted() to derive it from real data instead of assuming it
NON_PARTICIPANT_BPS_SENTINEL = -1e18
NON_PARTICIPANT_THRESHOLD = -1e17  # anything below this is treated as "did not play, cannot rank for bonus"


@dataclass(frozen=True)
class JointPlayerInput:
    player_id: str
    team: str
    position: str
    minutes_forecast: MinutesForecast
    goal_share: float
    assist_share: float


@dataclass(frozen=True)
class JointFixtureInput:
    home_team: str
    away_team: str
    score_matrix: np.ndarray


@dataclass
class JointPlayerSimResult:
    player_id: str
    mean_points: float
    std_points: float
    samples: np.ndarray = field(repr=False)
    minutes_samples: np.ndarray = field(repr=False)
    goals_samples: np.ndarray = field(repr=False)
    assists_samples: np.ndarray = field(repr=False)
    bonus_samples: np.ndarray = field(repr=False)


def fit_p_goal_assisted(rows: list[dict]) -> float:
    """Real-data estimate of P(a goal is assisted), from any historical
    merged_gw-style row set: total assists / total goals scored,
    league-wide. An approximation (doesn't perfectly separate penalties/
    own goals, which are typically unassisted) but grounded in real data
    rather than an arbitrary constant."""
    total_goals = sum(int(r["goals_scored"]) for r in rows)
    total_assists = sum(int(r["assists"]) for r in rows)
    if total_goals == 0:
        return P_GOAL_ASSISTED_DEFAULT
    return min(1.0, total_assists / total_goals)


def _competition_rank_bonus(bps_column: np.ndarray, player_ids: list[str]) -> dict[str, int]:
    """Standard competition ("1224") ranking: ties share the higher bonus
    tier, and the next distinct value jumps by the number of tied
    players — e.g. a 2-way tie for 1st gives both players 3, and the next
    player gets 1 (not 2), matching the official rule.

    Non-participants (sentinel BPS <= NON_PARTICIPANT_THRESHOLD) are
    excluded from ranking ENTIRELY, not just relied upon to sort last —
    with fewer than 3 real participants, a naive rank-by-sort-position
    would otherwise hand a non-participant "rank 3" and a real bonus
    point, which is exactly the bug this guard exists to prevent (caught
    by test_non_appearing_player_never_gets_bonus)."""
    participant_mask = bps_column > NON_PARTICIPANT_THRESHOLD
    participant_idx = np.where(participant_mask)[0]
    if len(participant_idx) == 0:
        return {}
    participant_vals = bps_column[participant_idx]

    order = np.argsort(-participant_vals)
    sorted_vals = participant_vals[order]
    ranks = np.empty(len(sorted_vals), dtype=int)
    ranks[0] = 1
    for i in range(1, len(sorted_vals)):
        ranks[i] = ranks[i - 1] if sorted_vals[i] == sorted_vals[i - 1] else i + 1
    bonus_map = {1: 3, 2: 2, 3: 1}
    out = {}
    for i, local_pos in enumerate(order):
        orig_idx = participant_idx[local_pos]
        r = ranks[i]
        if r in bonus_map:
            out[player_ids[orig_idx]] = bonus_map[r]
    return out


def simulate_gameweek_joint(
    fixtures: list[JointFixtureInput],
    players: list[JointPlayerInput],
    rules: dict,
    bps_models: dict[str, PositionBPSModel],
    n_scenarios: int = 4000,
    p_goal_assisted: float = P_GOAL_ASSISTED_DEFAULT,
    seed: int = 2026,
) -> dict[str, JointPlayerSimResult]:
    from apex_fpl.simulation.monte_carlo import _vectorized_points  # reuse the authoritative core-scoring function

    rng = np.random.default_rng(seed)
    n = n_scenarios

    players_by_team: dict[str, list[JointPlayerInput]] = {}
    for p in players:
        players_by_team.setdefault(p.team, []).append(p)

    team_fixture_side = {}
    for fx in fixtures:
        team_fixture_side[fx.home_team] = (fx, "home")
        team_fixture_side[fx.away_team] = (fx, "away")

    fixture_goals = {}
    for fx in fixtures:
        flat = fx.score_matrix.flatten()
        idx = rng.choice(len(flat), size=n, p=flat)
        gh = idx // fx.score_matrix.shape[1]
        ga = idx % fx.score_matrix.shape[1]
        fixture_goals[id(fx)] = (gh, ga)

    minutes_samples: dict[str, np.ndarray] = {}
    for p in players:
        u = rng.random(n)
        m = np.where(
            u < p.minutes_forecast.p_60_plus, p.minutes_forecast.expected_minutes_if_played,
            np.where(u < p.minutes_forecast.p_appearance, rng.uniform(1, 59, n), 0.0),
        )
        minutes_samples[p.player_id] = m

    goals_samples: dict[str, np.ndarray] = {p.player_id: np.zeros(n, dtype=int) for p in players}
    assists_samples: dict[str, np.ndarray] = {p.player_id: np.zeros(n, dtype=int) for p in players}

    for team, team_players in players_by_team.items():
        if team not in team_fixture_side:
            continue
        fx, side = team_fixture_side[team]
        gh, ga = fixture_goals[id(fx)]
        team_goals_per_scenario = gh if side == "home" else ga

        ids = [p.player_id for p in team_players]
        raw_goal_shares = np.array([p.goal_share for p in team_players])
        goal_shares = raw_goal_shares / raw_goal_shares.sum() if raw_goal_shares.sum() > 0 else np.full(len(ids), 1.0 / len(ids))
        raw_assist_shares = np.array([p.assist_share for p in team_players])

        for s in range(n):
            tg = int(team_goals_per_scenario[s])
            if tg == 0:
                continue
            scorer_positions = rng.choice(len(ids), size=tg, p=goal_shares)
            for scorer_pos in scorer_positions:
                goals_samples[ids[scorer_pos]][s] += 1
                if rng.random() < p_goal_assisted and len(ids) > 1:
                    mask = np.ones(len(ids), dtype=bool)
                    mask[scorer_pos] = False
                    probs = raw_assist_shares[mask]
                    probs = probs / probs.sum() if probs.sum() > 0 else np.full(mask.sum(), 1.0 / mask.sum())
                    assister_pos = rng.choice(np.where(mask)[0], p=probs)
                    assists_samples[ids[assister_pos]][s] += 1

    clean_sheet_samples: dict[str, np.ndarray] = {}
    goals_conceded_samples: dict[str, np.ndarray] = {}
    for p in players:
        fx, side = team_fixture_side[p.team]
        gh, ga = fixture_goals[id(fx)]
        goals_against = ga if side == "home" else gh
        played = minutes_samples[p.player_id] > 0
        clean_sheet_samples[p.player_id] = played & (minutes_samples[p.player_id] >= 60) & (goals_against == 0)
        goals_conceded_samples[p.player_id] = np.where(played, goals_against, 0)

    bps_samples: dict[str, np.ndarray] = {}
    for p in players:
        pos = POSITION_MAP[p.position]
        model = bps_models.get(pos)
        played = minutes_samples[p.player_id] > 0
        if model is None:
            bps_samples[p.player_id] = np.where(played, 0.0, -1e9)
            continue
        played60 = (minutes_samples[p.player_id] >= 60).astype(float)
        under60 = ((minutes_samples[p.player_id] > 0) & (minutes_samples[p.player_id] < 60)).astype(float)
        g = goals_samples[p.player_id].astype(float)
        a = assists_samples[p.player_id].astype(float)
        cs = clean_sheet_samples[p.player_id].astype(float)
        gc = goals_conceded_samples[p.player_id].astype(float)
        zeros = np.zeros(n)
        x = np.column_stack([played60, under60, g, a, cs, zeros, gc, zeros, zeros, zeros, zeros, zeros])
        coef = np.array([model.coefficients[name] for name in FEATURE_NAMES])
        mean_bps = model.intercept + x @ coef
        noise = rng.normal(0, model.residual_std, n)
        raw_bps = mean_bps + noise
        bps_samples[p.player_id] = np.where(played, raw_bps, NON_PARTICIPANT_BPS_SENTINEL)

    bonus_samples: dict[str, np.ndarray] = {p.player_id: np.zeros(n, dtype=int) for p in players}
    for fx in fixtures:
        match_player_ids = [p.player_id for p in players if p.team in (fx.home_team, fx.away_team)]
        if not match_player_ids:
            continue
        bps_matrix = np.array([bps_samples[pid] for pid in match_player_ids])
        for s in range(n):
            col = bps_matrix[:, s]
            if np.all(col <= NON_PARTICIPANT_THRESHOLD):
                continue  # nobody played (shouldn't happen for a real fixture, defensive guard)
            awarded = _competition_rank_bonus(col, match_player_ids)
            for pid, pts in awarded.items():
                bonus_samples[pid][s] = pts

    results = {}
    for p in players:
        minutes = minutes_samples[p.player_id]
        goals = goals_samples[p.player_id]
        assists = assists_samples[p.player_id]
        cs = clean_sheet_samples[p.player_id]
        gc = goals_conceded_samples[p.player_id]
        bonus = bonus_samples[p.player_id]
        pts = _vectorized_points(p.position, minutes, goals, assists, cs, gc, rules) + bonus
        results[p.player_id] = JointPlayerSimResult(
            player_id=p.player_id, mean_points=float(pts.mean()), std_points=float(pts.std()),
            samples=pts, minutes_samples=minutes, goals_samples=goals, assists_samples=assists, bonus_samples=bonus,
        )
    return results
