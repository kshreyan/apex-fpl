#!/usr/bin/env python3
"""Phase 4 model tournament: team-goal forecasting (spec Parts VI, XXXVII).

Nested temporal cross-validation across 6 real seasons (2019-20..2024-25):
each outer fold trains on all prior seasons, tunes hyperparameters on the
most recent training season ALONE (inner validation — never touching the
outer test season), then walks forward gameweek-by-gameweek through the
outer test season exactly like the Phase 3 replay (refitting on cumulative
history before each gameweek, never using that gameweek's own results as
input).

Candidates:
  - champion_unfit      : attack/defense model, Phase 2/3's unfit constants (K_BASE=0.045, HALFLIFE=380, rho=-0.04)
  - challenger_tuned     : same mechanism, constants grid-searched per fold on inner-validation data only
  - baseline_constant    : no team-skill signal at all (spec Part XXXVI mandatory baseline)
  - baseline_prev_season_avg : per-team average goals for/against, no decay/opponent-adjustment

Evaluated with proper scoring rules (log loss, RPS, Brier, accuracy, ECE,
goals MAE) on every real match in all 3 outer test seasons, then a
block-bootstrap (blocked by season+gameweek, per spec Part XXXIX) on the
challenger-vs-champion log-loss difference to decide promotion under the
criteria in spec Part LXI.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.evaluation import metrics as em
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import baselines
from apex_fpl.models.teams import scoreline as sl
from apex_fpl.models.teams import tuning

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase4_tournament"

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
K_BASE_GRID = [0.02, 0.03, 0.045, 0.06, 0.08, 0.12]
HALFLIFE_GRID = [90.0, 180.0, 380.0, 730.0]
CANDIDATE_NAMES = ["champion_unfit", "challenger_tuned", "baseline_constant", "baseline_prev_season_avg"]
N_BOOTSTRAP = 5000
SEED = 2026


def _to_fixture_objs(raw_rows) -> list[ad.Fixture]:
    return [ad.Fixture(r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"]) for r in raw_rows]


def run_outer_fold(train_seasons: list[str], tuning_val_season: str, test_season: str) -> dict:
    print(f"\n--- Outer fold: train={train_seasons + [tuning_val_season]}, test={test_season} ---")

    inner_train_raw = sorted((r for s in train_seasons for r in vl.load_fixtures(s)), key=lambda r: r["date"])
    val_raw = sorted(vl.load_fixtures(tuning_val_season), key=lambda r: r["date"])
    tuned = tuning.grid_search(_to_fixture_objs(inner_train_raw), _to_fixture_objs(val_raw), K_BASE_GRID, HALFLIFE_GRID)
    print(f"  tuned constants: k_base={tuned.k_base}, halflife={tuned.halflife_days}, rho={tuned.rho:.4f} "
          f"(inner val log_loss={tuned.inner_val_log_loss})")

    full_history_raw = sorted(
        (r for s in (train_seasons + [tuning_val_season]) for r in vl.load_fixtures(s)), key=lambda r: r["date"]
    )
    test_raw = sorted(vl.load_fixtures(test_season), key=lambda r: r["date"])
    test_events = sorted({r["event"] for r in test_raw if r["event"] is not None})

    per_match = {name: [] for name in CANDIDATE_NAMES}  # each entry: dict with event, probs, outcome, pred/actual goals

    for event in test_events:
        target_rows = [r for r in test_raw if r["event"] == event and r["home_score"] is not None]
        if not target_rows:
            continue  # blank gameweek in this season, or not yet played (shouldn't occur for a completed historical season)
        min_date = min(r["date"] for r in target_rows)
        history_raw = full_history_raw + [r for r in test_raw if r["date"] < min_date and r["home_score"] is not None]
        if not history_raw:
            continue
        history_fixtures = _to_fixture_objs(history_raw)

        models = {
            "champion_unfit": (ad.fit(history_fixtures), sl.RHO_DEFAULT),
            "challenger_tuned": (ad.fit(history_fixtures, k_base=tuned.k_base, halflife_days=tuned.halflife_days), tuned.rho),
            "baseline_constant": (baselines.fit_constant(history_fixtures), 0.0),
            "baseline_prev_season_avg": (baselines.fit_previous_season_average(history_fixtures), 0.0),
        }

        for r in target_rows:
            outcome = "H" if r["home_score"] > r["away_score"] else ("A" if r["home_score"] < r["away_score"] else "D")
            for name, (model, rho) in models.items():
                eh, ea = model.expected_goals(r["home_team"], r["away_team"], r["date"])
                wdl = tuning._wdl_from_matrix(sl.score_matrix(eh, ea, rho=rho))
                per_match[name].append({
                    "event": event, "wdl": wdl, "outcome": outcome,
                    "pred_home": eh, "pred_away": ea, "actual_home": r["home_score"], "actual_away": r["away_score"],
                })

    fold_metrics = {}
    for name, rows in per_match.items():
        if not rows:
            continue
        probs = np.array([row["wdl"] for row in rows])
        outcomes = [row["outcome"] for row in rows]
        fold_metrics[name] = em.full_metrics(
            probs, outcomes,
            pred_home=[row["pred_home"] for row in rows], pred_away=[row["pred_away"] for row in rows],
            actual_home=[row["actual_home"] for row in rows], actual_away=[row["actual_away"] for row in rows],
        )
        print(f"  {name:<26} log_loss={fold_metrics[name]['log_loss']}  rps={fold_metrics[name]['rps']}  "
              f"acc={fold_metrics[name]['accuracy']}  goals_mae={fold_metrics[name]['goals_mae']}")

    return {"test_season": test_season, "tuned_constants": tuned.__dict__, "metrics": fold_metrics, "per_match": per_match}


def block_bootstrap_log_loss_diff(fold_results: list[dict], name_a: str, name_b: str) -> tuple[float, float, float]:
    """Block bootstrap (block = one season's one gameweek) on the per-match
    log-loss difference (name_b - name_a). Negative mean = name_b (usually
    the challenger) has LOWER log loss, i.e. is better."""
    blocks: dict[tuple, list[float]] = defaultdict(list)
    for fold in fold_results:
        rows_a = fold["per_match"][name_a]
        rows_b = fold["per_match"][name_b]
        for ra, rb in zip(rows_a, rows_b):
            assert ra["outcome"] == rb["outcome"]
            ll_a = -np.log(max(ra["wdl"][em.IDX[ra["outcome"]]], 1e-15))
            ll_b = -np.log(max(rb["wdl"][em.IDX[rb["outcome"]]], 1e-15))
            blocks[(fold["test_season"], ra["event"])].append(ll_b - ll_a)

    block_keys = list(blocks.keys())
    block_means = np.array([np.mean(blocks[k]) for k in block_keys])  # one summary value per block
    rng = np.random.default_rng(SEED)
    n = len(block_means)
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot[i] = rng.choice(block_means, size=n, replace=True).mean()
    return float(block_means.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 4 team model tournament ===")
    print(f"Seasons: {SEASONS}")

    # Expanding-window outer folds: train on everything up to (but not
    # including) the val season; val season is the most recent training
    # season, held out purely for inner hyperparameter tuning; test season
    # is the true, fully-held-out outer fold.
    outer_folds = [
        (["2019-20", "2020-21"], "2021-22", "2022-23"),
        (["2019-20", "2020-21", "2021-22"], "2022-23", "2023-24"),
        (["2019-20", "2020-21", "2021-22", "2022-23"], "2023-24", "2024-25"),
    ]

    fold_results = [run_outer_fold(train, val, test) for train, val, test in outer_folds]

    print("\n=== Aggregate metrics across all 3 outer test seasons (pooled matches) ===")
    pooled = {name: [] for name in CANDIDATE_NAMES}
    for fold in fold_results:
        for name in CANDIDATE_NAMES:
            pooled[name].extend(fold["per_match"].get(name, []))

    aggregate_metrics = {}
    for name, rows in pooled.items():
        if not rows:
            continue
        probs = np.array([row["wdl"] for row in rows])
        outcomes = [row["outcome"] for row in rows]
        aggregate_metrics[name] = em.full_metrics(
            probs, outcomes,
            pred_home=[row["pred_home"] for row in rows], pred_away=[row["pred_away"] for row in rows],
            actual_home=[row["actual_home"] for row in rows], actual_away=[row["actual_away"] for row in rows],
        )
        m = aggregate_metrics[name]
        print(f"  {name:<26} n={m['n']:4d}  log_loss={m['log_loss']}  rps={m['rps']}  "
              f"brier={m['brier']}  acc={m['accuracy']}  ece={m['ece']}  goals_mae={m['goals_mae']}")

    mean_diff, ci_low, ci_high = block_bootstrap_log_loss_diff(fold_results, "champion_unfit", "challenger_tuned")
    print(f"\nBlock-bootstrapped log-loss difference (challenger - champion): {mean_diff:+.4f} "
          f"(95% CI [{ci_low:+.4f}, {ci_high:+.4f}])")
    promote = ci_high < 0  # challenger's log loss is significantly LOWER (better) than champion's
    print(f"Promotion decision: {'PROMOTE challenger_tuned to champion' if promote else 'DO NOT PROMOTE — keep champion_unfit'}")

    summary = {
        "seasons": SEASONS,
        "outer_folds": [{"train": tr, "val": v, "test": te} for tr, v, te in outer_folds],
        "per_fold_metrics": [{"test_season": f["test_season"], "tuned_constants": f["tuned_constants"], "metrics": f["metrics"]} for f in fold_results],
        "aggregate_metrics": aggregate_metrics,
        "challenger_vs_champion_log_loss_diff_mean": mean_diff,
        "challenger_vs_champion_log_loss_diff_95ci": [ci_low, ci_high],
        "promotion_decision": "promote" if promote else "do_not_promote",
    }
    (ARTIFACT_DIR / "tournament_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'tournament_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
