#!/usr/bin/env python3
"""Block 2.1 (Phase 13 promotion schedule) — pre-registered validation
of a SURGICAL, threshold-gated bonus overlay, as opposed to the blanket
joint-simulator swap docs/phase6_joint_simulation_report.md already
tested and rejected (aggregate MAE worse, +0.071, CI excludes zero,
because ~95% of players win zero bonus in any given gameweek and the
model can only add noise there — closed, not re-litigated here).

**The idea, precisely.** Rather than replacing the baseline simulator's
estimate with the joint simulator's own (which also uses a different,
correlated goal/assist allocation mechanism — a bigger, confounded
change), this overlays ONE additional term on top of the EXISTING live
baseline estimate: `overlay_pred = baseline_pred + pred_bonus` ONLY for
players whose predicted mean bonus (from the same BPS model /
competition-rank bonus mechanism already built and validated in Phase 6)
clears a pre-registered threshold; everyone else stays exactly as today
(implicit zero bonus). This is the "exploit the signal more surgically"
path Phase 6's own report named as the honest next step, made concrete.

**Pre-registered gate (fixed BEFORE looking at results):** for each of
3 threshold candidates (pred_bonus >= 0.5, 1.0, 1.5 — chosen to span
"lightly selective" to "highly selective" without an exhaustive sweep,
keeping the multiple-comparison burden to 3 tests, not dozens), report
the aggregate MAE difference (overlay - baseline) across all 4
independent seasons' pooled player-gameweeks with a block-bootstrap 95%
CI (block = one gameweek, matching every other significance test in
this project) and a Bonferroni-corrected significance threshold
(alpha=0.05/3 ~ CI must exclude zero at the 98.33% level to count as a
promotable finding, not just nominal 95%). A threshold only clears the
gate if its CI excludes zero in the IMPROVING direction AND the
Bonferroni-adjusted check also holds.

**Ruleset-validity caveat, disclosed not hidden:** all 4 test seasons
predate the 2026/27 BPS reweighting (CBI 1-per-3 vs 1-per-2, tackled-
penalty removed — configs/seasons/2026_27.yaml's bps_2026_27_changes).
The BPS model's feature set (goals/assists/clean-sheets/saves/cards —
see apex_fpl.models.bonus.bps_model's own docstring) never included
CBI/tackle counts for ANY season tested (not present in the historical
archive), so this reweighting doesn't invalidate the model's existing
relationships so much as sharpen an already-known blind spot: any
promoted overlay will be least reliable for defensively-driven bonus
(CBI/tackle-heavy defenders), most reliable for goal/assist-driven
bonus (attackers) -- consistent with Block 1.5's finding that bonus
already concentrates more in forwards than defenders under the real
2025/26 ruleset. This is exactly the demographic where this overlay's
evidence, gathered here, is strongest.

Walk-forward discipline matches scripts/run_phase6_multi_gw_validation.py
exactly: BPS model + p_goal_assisted fit ONCE on 2021-22 only, evaluated
against the 4 seasons used throughout this project's decision-level
replays (2020-21, 2022-23, 2023-24, 2024-25), GW7/11/15/19/23/27/31/35
per season (32 gameweek observations minus blanks) — the same
established test grid, not a new one invented for this analysis.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.models.bonus import bps_model
from apex_fpl.rules import scoring
from apex_fpl.simulation import joint_simulator as js
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "bonus_surgical_promotion_validation"
BPS_TRAIN_SEASON = "2021-22"
TEST_SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
GAMEWEEKS = [7, 11, 15, 19, 23, 27, 31, 35]
N_SCENARIOS_JOINT = 3000
SEED = 2026
N_BOOTSTRAP = 5000
THRESHOLDS = [0.5, 1.0, 1.5]  # pre-registered -- see module docstring
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(THRESHOLDS)


def load_merged_gw(season: str) -> list[dict]:
    with open(REPO_ROOT / "data" / "external" / "vaastav" / season / "merged_gw.csv") as f:
        return list(csv.DictReader(f))


def run_one_gameweek(season: str, gw: int, bps_models, p_assisted: float) -> list[dict] | None:
    """Returns per-player rows: {player_id, baseline_pred, pred_bonus,
    actual_total, actual_bonus} -- enough to test ANY threshold rule
    offline afterward without re-simulating per threshold candidate."""
    try:
        data = build_gameweek_scenario_data(season, gw)
    except BlankGameweekError:
        return None

    rules = scoring.load_scoring_rules("2026_27")
    baseline_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)

    joint_fixture_inputs = [js.JointFixtureInput(f.home_team, f.away_team, f.score_matrix) for f in data.fixture_inputs]
    joint_players = [
        js.JointPlayerInput(
            player_id=p.player_id, team=p.team, position=p.position, minutes_forecast=p.minutes_forecast,
            goal_share=data.shares[p.player_id].goal_share, assist_share=data.shares[p.player_id].assist_share,
        )
        for p in data.players_for_sim
    ]
    joint_results = js.simulate_gameweek_joint(
        joint_fixture_inputs, joint_players, rules, bps_models, n_scenarios=N_SCENARIOS_JOINT, p_goal_assisted=p_assisted, seed=SEED,
    )

    actual_by_player = {r["element"]: r for r in data.target_rows}
    rows = []
    for pid in data.candidates_meta:
        if pid not in baseline_results or pid not in joint_results or pid not in actual_by_player:
            continue
        rows.append({
            "player_id": pid,
            "baseline_pred": baseline_results[pid].mean_points,
            "pred_bonus": float(joint_results[pid].bonus_samples.mean()),
            "actual_total": int(actual_by_player[pid]["total_points"]),
            "actual_bonus": int(actual_by_player[pid]["bonus"]),
        })
    return rows


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = ALPHA) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(values.mean()), float(np.percentile(boot, lo_pct)), float(np.percentile(boot, hi_pct))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Bonus surgical-promotion validation: {len(TEST_SEASONS)} seasons x {len(GAMEWEEKS)} gameweeks, thresholds={THRESHOLDS} ===\n")

    print(f"--- Fitting BPS model + p_goal_assisted on {BPS_TRAIN_SEASON} only ---")
    train_rows = load_merged_gw(BPS_TRAIN_SEASON)
    bps_models = bps_model.fit_bps_models(train_rows)
    p_assisted = js.fit_p_goal_assisted(train_rows)

    per_gw_rows: dict[str, list[dict]] = {}
    blanks = []
    for season in TEST_SEASONS:
        for gw in GAMEWEEKS:
            key = f"{season}_gw{gw}"
            rows = run_one_gameweek(season, gw, bps_models, p_assisted)
            if rows is None:
                print(f"{season} GW{gw}: blank gameweek, skipped")
                blanks.append([season, gw])
                continue
            per_gw_rows[key] = rows
            print(f"{season} GW{gw:2d}: n={len(rows)} players, "
                  f"n_actual_bonus_winners={sum(1 for r in rows if r['actual_bonus'] > 0)}")

    # ---- Pre-registered gate: per threshold, per-gameweek MAE diff, block-bootstrapped ----
    threshold_results = {}
    for threshold in THRESHOLDS:
        gw_diffs, gw_subset_sizes, gw_precisions = [], [], []
        for key, rows in per_gw_rows.items():
            baseline_err = np.array([abs(r["baseline_pred"] - r["actual_total"]) for r in rows])
            overlay_pred = np.array([
                r["baseline_pred"] + r["pred_bonus"] if r["pred_bonus"] >= threshold else r["baseline_pred"]
                for r in rows
            ])
            actual = np.array([r["actual_total"] for r in rows])
            overlay_err = np.abs(overlay_pred - actual)
            gw_diffs.append(float(overlay_err.mean() - baseline_err.mean()))

            adjusted = [r for r in rows if r["pred_bonus"] >= threshold]
            gw_subset_sizes.append(len(adjusted))
            if adjusted:
                gw_precisions.append(sum(1 for r in adjusted if r["actual_bonus"] > 0) / len(adjusted))

        diffs = np.array(gw_diffs)
        mean, lo, hi = bootstrap_ci(diffs, alpha=ALPHA)
        _, lo_corrected, hi_corrected = bootstrap_ci(diffs, alpha=BONFERRONI_ALPHA)
        excludes_zero_nominal = lo > 0 or hi < 0
        excludes_zero_corrected = lo_corrected > 0 or hi_corrected < 0
        promoted = excludes_zero_corrected and mean < 0  # improving = lower MAE = negative diff

        threshold_results[threshold] = {
            "mae_diff_mean": mean, "ci95_nominal": [lo, hi], "ci_bonferroni": [lo_corrected, hi_corrected],
            "excludes_zero_nominal": excludes_zero_nominal, "excludes_zero_bonferroni_corrected": excludes_zero_corrected,
            "promoted": promoted,
            "mean_subset_size_per_gw": float(np.mean(gw_subset_sizes)),
            "mean_precision": float(np.mean(gw_precisions)) if gw_precisions else None,
        }
        print(f"\nThreshold pred_bonus>={threshold}: MAE diff (overlay-baseline) = {mean:+.4f}, "
              f"95% CI {[round(lo,4), round(hi,4)]}, Bonferroni-corrected CI {[round(lo_corrected,4), round(hi_corrected,4)]}, "
              f"promoted={promoted}, mean subset size/GW={np.mean(gw_subset_sizes):.1f}, "
              f"mean precision={np.mean(gw_precisions) if gw_precisions else float('nan'):.3f}")

    summary = {
        "n_gameweeks": len(per_gw_rows), "n_blank": len(blanks), "blanks": blanks,
        "thresholds_tested": THRESHOLDS, "bonferroni_alpha": BONFERRONI_ALPHA,
        "threshold_results": {str(k): v for k, v in threshold_results.items()},
        "per_gw_rows": per_gw_rows,
    }
    (ARTIFACT_DIR / "validation_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'validation_results.json').relative_to(REPO_ROOT)}")

    any_promoted = [t for t, r in threshold_results.items() if r["promoted"]]
    print(f"\n=== DECISION: {'PROMOTE threshold(s) ' + str(any_promoted) if any_promoted else 'NO THRESHOLD PROMOTED'} ===")


if __name__ == "__main__":
    main()
