"""Time-decayed attack/defense team-strength ratings.

Ported and adapted from the FIFA World Cup predictor project's
time_decay_elo.py (docs/world_cup_transfer_audit.md classifies the
mechanism A — directly reusable — and the constants B — need refitting).
Tournament-importance weighting is dropped since every Premier League
fixture carries equal competitive weight (unlike a World Cup mixing
friendlies, qualifiers, and the tournament itself).

K_BASE_DEFAULT and HALFLIFE_DAYS_DEFAULT are the Phase 2/3 baseline's
values, carried over UNFIT from the World Cup repo's international-
football fit. They are now `fit()` parameters (not module-level globals)
specifically so Phase 4's hyperparameter search
(src/apex_fpl/models/teams/tuning.py) can try different values per fold
without mutating shared state — a model's fitted constants live on the
AttackDefenseModel instance it produced, not in module globals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

K_BASE_DEFAULT = 0.045
HALFLIFE_DAYS_DEFAULT = 380.0
GOAL_CLIP = (0.05, 6.0)


@dataclass
class Fixture:
    date: datetime
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None


@dataclass
class AttackDefenseModel:
    mu: float
    home_adv: float
    k_base: float = K_BASE_DEFAULT
    halflife_days: float = HALFLIFE_DAYS_DEFAULT
    attack: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)
    last_played: dict = field(default_factory=dict)

    def _decayed(self, team: str, at_date: datetime) -> tuple[float, float]:
        a = self.attack.get(team, 0.0)
        d = self.defense.get(team, 0.0)
        if team in self.last_played:
            days = (at_date - self.last_played[team]).days
            if days > 0:
                f = 0.5 ** (days / self.halflife_days)
                a *= f
                d *= f
        return a, d

    def expected_goals(self, home_team: str, away_team: str, at_date: datetime) -> tuple[float, float]:
        a_h, d_h = self._decayed(home_team, at_date)
        a_a, d_a = self._decayed(away_team, at_date)
        exp_h = np.exp(self.mu + self.home_adv + a_h - d_a)
        exp_a = np.exp(self.mu + a_a - d_h)
        return (
            float(np.clip(exp_h, *GOAL_CLIP)),
            float(np.clip(exp_a, *GOAL_CLIP)),
        )

    def _update(self, fixture: Fixture) -> None:
        a_h, d_h = self._decayed(fixture.home_team, fixture.date)
        a_a, d_a = self._decayed(fixture.away_team, fixture.date)
        exp_h, exp_a = self.expected_goals(fixture.home_team, fixture.away_team, fixture.date)
        eh = (fixture.home_score - exp_h) / np.sqrt(exp_h + 1.0)
        ea = (fixture.away_score - exp_a) / np.sqrt(exp_a + 1.0)
        self.attack[fixture.home_team] = float(np.clip(a_h + self.k_base * eh, -1.2, 1.2))
        self.defense[fixture.away_team] = float(np.clip(d_a - self.k_base * eh, -1.2, 1.2))
        self.attack[fixture.away_team] = float(np.clip(a_a + self.k_base * ea, -1.2, 1.2))
        self.defense[fixture.home_team] = float(np.clip(d_h - self.k_base * ea, -1.2, 1.2))
        self.last_played[fixture.home_team] = fixture.date
        self.last_played[fixture.away_team] = fixture.date


def fit(
    fixtures: list[Fixture],
    k_base: float = K_BASE_DEFAULT,
    halflife_days: float = HALFLIFE_DAYS_DEFAULT,
) -> AttackDefenseModel:
    """Fit ratings from CHRONOLOGICALLY SORTED fixtures. The caller is
    responsible for ensuring `fixtures` contains only matches strictly
    before whatever deadline this model will be used to predict past —
    this function has no knowledge of "the future" and will happily leak
    if handed fixtures from after a target gameweek."""
    completed = [f for f in fixtures if f.home_score is not None]
    if not completed:
        raise ValueError("fit() requires at least one completed fixture")

    all_goals = [f.home_score for f in completed] + [f.away_score for f in completed]
    mu = float(np.log(max(float(np.mean(all_goals)), 0.3)))
    home_mean = max(float(np.mean([f.home_score for f in completed])), 0.3)
    away_mean = max(float(np.mean([f.away_score for f in completed])), 0.3)
    home_adv = float(np.clip(np.log(home_mean) - np.log(away_mean), 0.0, 0.5))

    model = AttackDefenseModel(mu=mu, home_adv=home_adv, k_base=k_base, halflife_days=halflife_days)
    for f in fixtures:
        if f.home_score is None:
            continue
        model._update(f)
    return model
