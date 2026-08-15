"""Reduced-form Bonus Points System (BPS) model (spec Part XIV).

The spec's own vision for BPS is a bottom-up simulation from underlying
match actions (tackles, blocks, interceptions, recoveries, shot locations
etc.) — this project has no source for that event-level data
(docs/fpl_gap_analysis.md: "Defensive contributions... Goalkeepers...
requires per-match CBIT/CBIRT action counts — not available from
bootstrap-static; needs an event-level source"). The full BPS
point-per-action weight matrix is also an explicitly unresolved gap
(configs/seasons/2026_27.yaml's `unresolved_gaps`), since bootstrap-static
doesn't expose it either.

What real data DOES give us: `bps` (the actual awarded BPS score) and the
outcome-level events that plausibly drive most of it (goals, assists,
clean sheets, saves, minutes, cards, goals conceded) for every historical
player-gameweek. This module fits a per-position linear regression of bps
on those events — an honest, evidence-based approximation of the
*aggregate effect* of the true action-level formula, not a claim to have
reverse-engineered the exact official weights. This is precisely the
`configs/seasons/2026_27.yaml` resolution_plan's own suggestion: "regress
against clearances_blocks_interceptions/tackles/recoveries/saves/goals/
etc. to recover implied per-action weights" — done here with the subset
of those regressors we actually have for historical seasons (goals-based
events; CBI/tackles/recoveries are not present in the historical archive
for the seasons used, since defensive contributions is a 2025/26+
mechanic).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

POSITIONS = ("GK", "DEF", "MID", "FWD")
FEATURE_NAMES = [
    "played_60plus", "played_under_60", "goals_scored", "assists", "clean_sheet",
    "saves", "goals_conceded", "yellow_cards", "red_cards", "own_goals",
    "penalties_saved", "penalties_missed",
]


def _row_to_features(row: dict) -> list[float]:
    minutes = int(row["minutes"])
    return [
        1.0 if minutes >= 60 else 0.0,
        1.0 if 0 < minutes < 60 else 0.0,
        float(row["goals_scored"]),
        float(row["assists"]),
        float(row["clean_sheets"]),
        float(row["saves"]),
        float(row["goals_conceded"]),
        float(row["yellow_cards"]),
        float(row["red_cards"]),
        float(row["own_goals"]),
        float(row["penalties_saved"]),
        float(row["penalties_missed"]),
    ]


@dataclass(frozen=True)
class PositionBPSModel:
    position: str
    coefficients: dict[str, float]
    intercept: float
    residual_std: float
    n_train: int

    def predict_mean(self, events: dict[str, float]) -> float:
        x = np.array([events.get(name, 0.0) for name in FEATURE_NAMES])
        coef = np.array([self.coefficients[name] for name in FEATURE_NAMES])
        return float(self.intercept + coef @ x)


def fit_bps_models(rows: list[dict]) -> dict[str, PositionBPSModel]:
    """rows: raw merged_gw.csv-style dicts (any real historical season)."""
    by_position: dict[str, list[dict]] = {p: [] for p in POSITIONS}
    for r in rows:
        pos = r.get("position")
        if pos in by_position and int(r["minutes"]) > 0:  # only players who actually appeared have a meaningful BPS
            by_position[pos].append(r)

    models = {}
    for pos, pos_rows in by_position.items():
        if len(pos_rows) < 20:
            continue
        X = np.array([_row_to_features(r) for r in pos_rows])
        y = np.array([float(r["bps"]) for r in pos_rows])
        reg = LinearRegression().fit(X, y)
        residuals = y - reg.predict(X)
        models[pos] = PositionBPSModel(
            position=pos,
            coefficients=dict(zip(FEATURE_NAMES, reg.coef_.tolist())),
            intercept=float(reg.intercept_),
            residual_std=float(residuals.std()),
            n_train=len(pos_rows),
        )
    return models


def evaluate_bps_models(models: dict[str, PositionBPSModel], test_rows: list[dict]) -> dict[str, dict]:
    """Held-out evaluation: MAE and R^2 per position on genuinely unseen rows."""
    by_position: dict[str, list[dict]] = {p: [] for p in POSITIONS}
    for r in test_rows:
        pos = r.get("position")
        if pos in by_position and int(r["minutes"]) > 0:
            by_position[pos].append(r)

    results = {}
    for pos, pos_rows in by_position.items():
        if pos not in models or not pos_rows:
            continue
        model = models[pos]
        y_true = np.array([float(r["bps"]) for r in pos_rows])
        y_pred = np.array([model.predict_mean(dict(zip(FEATURE_NAMES, _row_to_features(r)))) for r in pos_rows])
        mae = float(np.mean(np.abs(y_true - y_pred)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results[pos] = {"n": len(pos_rows), "mae": round(mae, 2), "r2": round(r2, 4)}
    return results
