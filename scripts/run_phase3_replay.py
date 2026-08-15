#!/usr/bin/env python3
"""Phase 3: deadline-by-deadline historical replay across many gameweeks.

Generalizes the Phase 2 milestone (docs/phase2_milestone_report.md) into a
loop across a full season's remaining gameweeks, producing per-GW frozen
artifacts plus an aggregate, block-bootstrapped comparison against the
recent-form baseline — the statistical evidence the Phase 2 report
explicitly said a single gameweek could not provide (spec Part XXXIX:
protect against research overfitting, use a block bootstrap rather than
trusting one point comparison).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.backtesting.replay import BlankGameweekError, run_gameweek

REPO_ROOT = Path(__file__).resolve().parents[1]
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2022-23"
START_GW = 7  # first GW with enough lookback history available
LOOKBACK = 15  # matches the Phase 4b-promoted attacking model's tuned window (docs/phase4b_tournament_report.md); minutes model no longer uses this param at all (exponential decay has its own half-life)
N_BOOTSTRAP = 5000
SEED = 2026


def bootstrap_mean_ci(diffs: list[float], n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap over gameweeks: each gameweek's paired difference is
    already the natural resampling unit, since within-gameweek player
    outcomes are correlated (they share the same simulated/actual
    scorelines) but different gameweeks are treated as independent draws.
    Returns (observed_mean, ci_low, ci_high) for a 95% CI."""
    rng = np.random.default_rng(seed)
    arr = np.array(diffs, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = rng.choice(arr, size=n, replace=True).mean()
    return float(arr.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main() -> None:
    gameweeks = [g for g in vl.season_gameweeks(SEASON) if g >= START_GW]
    print(f"=== Phase 3 replay: {SEASON}, GW{gameweeks[0]}-GW{gameweeks[-1]} "
          f"({len(gameweeks)} real gameweeks) ===\n")
    artifact_dir = REPO_ROOT / "artifacts" / "phase3_replay" / SEASON

    rows, failures, blanks = [], [], []
    for gw in gameweeks:
        try:
            result = run_gameweek(SEASON, gw, lookback=LOOKBACK, artifact_dir=artifact_dir)
        except BlankGameweekError as e:
            print(f"GW{gw}: blank gameweek, skipped ({e})")
            blanks.append(gw)
            continue
        except Exception as e:
            print(f"GW{gw}: FAILED ({type(e).__name__}: {e})")
            failures.append({"gw": gw, "error": f"{type(e).__name__}: {e}"})
            continue
        ev = result.evaluation
        rows.append(ev)
        print(f"GW{gw:2d}: model={ev['model_squad_realized_points']:3d}  "
              f"baseline={ev['baseline_recent_form_realized_points']:3d}  "
              f"diff={ev['difference']:+3d}  captain={ev['captain_name']} ({ev['captain_realized_points']}pts)")

    if not rows:
        print("No gameweeks replayed successfully — aborting summary.")
        return

    diffs = [r["difference"] for r in rows]
    model_totals = [r["model_squad_realized_points"] for r in rows]
    baseline_totals = [r["baseline_recent_form_realized_points"] for r in rows]

    mean_diff, ci_low, ci_high = bootstrap_mean_ci(diffs)
    n_wins = sum(1 for d in diffs if d > 0)
    n_losses = sum(1 for d in diffs if d < 0)
    n_ties = sum(1 for d in diffs if d == 0)

    summary = {
        "season": SEASON, "start_gw": gameweeks[0], "end_gw": gameweeks[-1],
        "season_gameweek_count": len(gameweeks),
        "n_gameweeks_replayed": len(rows),
        "blank_gameweeks": blanks,
        "n_failures": len(failures), "failures": failures,
        "model_total_points": sum(model_totals), "baseline_total_points": sum(baseline_totals),
        "model_mean_points": float(np.mean(model_totals)), "baseline_mean_points": float(np.mean(baseline_totals)),
        "mean_difference": mean_diff, "bootstrap_95ci_low": ci_low, "bootstrap_95ci_high": ci_high,
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "gameweeks_model_won": n_wins, "gameweeks_baseline_won": n_losses, "gameweeks_tied": n_ties,
        "per_gameweek": rows,
    }
    summary_path = artifact_dir / "season_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== Season summary ({len(rows)}/{len(gameweeks)} gameweeks replayed, "
          f"{len(blanks)} blank, {len(failures)} failed) ===")
    print(f"Model total:    {sum(model_totals)} pts  (mean {np.mean(model_totals):.1f}/GW)")
    print(f"Baseline total: {sum(baseline_totals)} pts  (mean {np.mean(baseline_totals):.1f}/GW)")
    print(f"Mean difference per GW: {mean_diff:+.2f}  (95% bootstrap CI: [{ci_low:+.2f}, {ci_high:+.2f}])")
    print(f"Gameweeks: model won {n_wins}, baseline won {n_losses}, tied {n_ties}")
    print(f"\nWritten to {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
