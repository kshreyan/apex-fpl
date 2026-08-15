"""Mandatory baseline team models (spec Part XXXVI).

These exist so a candidate model has to beat something simpler before
being taken seriously — per the spec's own instruction that "a model
improving RMSE by 0.01 but adding huge operational complexity should
usually not replace the champion," and the general research-methodology
requirement to always compare against naive baselines, not just against
the previous champion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from apex_fpl.models.teams.attack_defense import Fixture


@dataclass
class ConstantStrengthModel:
    """No team-skill signal at all: every match gets the league-average
    home/away goal rate from training data, regardless of opponent. If a
    sophisticated model can't clearly beat this, something is wrong with
    it (or the sport is more homogeneous than assumed — worth knowing
    either way)."""
    home_goals_avg: float
    away_goals_avg: float

    def expected_goals(self, home_team: str, away_team: str, at_date: datetime) -> tuple[float, float]:
        return self.home_goals_avg, self.away_goals_avg


def fit_constant(fixtures: list[Fixture]) -> ConstantStrengthModel:
    completed = [f for f in fixtures if f.home_score is not None]
    if not completed:
        raise ValueError("fit_constant() requires at least one completed fixture")
    return ConstantStrengthModel(
        home_goals_avg=float(np.mean([f.home_score for f in completed])),
        away_goals_avg=float(np.mean([f.away_score for f in completed])),
    )


@dataclass
class PreviousSeasonAverageModel:
    """Per-team average goals for/against over the training window,
    ignoring opponent-specific adjustment beyond a simple average of "my
    attack" and "their defensive weakness," and ignoring any within-window
    trend. A step up from ConstantStrengthModel but still much simpler
    than the attack/defense model's online, decayed, opponent-adjusted
    ratings."""
    team_gf: dict = field(default_factory=dict)
    team_ga: dict = field(default_factory=dict)
    league_home_avg: float = 1.4
    league_away_avg: float = 1.1

    def expected_goals(self, home_team: str, away_team: str, at_date: datetime) -> tuple[float, float]:
        h_gf = self.team_gf.get(home_team, self.league_home_avg)
        a_ga = self.team_ga.get(away_team, self.league_away_avg)
        a_gf = self.team_gf.get(away_team, self.league_away_avg)
        h_ga = self.team_ga.get(home_team, self.league_home_avg)
        eh = (h_gf + a_ga) / 2
        ea = (a_gf + h_ga) / 2
        return max(eh, 0.05), max(ea, 0.05)


def fit_previous_season_average(fixtures: list[Fixture]) -> PreviousSeasonAverageModel:
    completed = [f for f in fixtures if f.home_score is not None]
    if not completed:
        raise ValueError("fit_previous_season_average() requires at least one completed fixture")
    gf: dict[str, list[float]] = {}
    ga: dict[str, list[float]] = {}
    for f in completed:
        gf.setdefault(f.home_team, []).append(f.home_score)
        ga.setdefault(f.home_team, []).append(f.away_score)
        gf.setdefault(f.away_team, []).append(f.away_score)
        ga.setdefault(f.away_team, []).append(f.home_score)
    return PreviousSeasonAverageModel(
        team_gf={t: float(np.mean(v)) for t, v in gf.items()},
        team_ga={t: float(np.mean(v)) for t, v in ga.items()},
        league_home_avg=float(np.mean([f.home_score for f in completed])),
        league_away_avg=float(np.mean([f.away_score for f in completed])),
    )
