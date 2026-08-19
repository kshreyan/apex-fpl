#!/usr/bin/env python3
"""Block 2.2 (Phase 13 promotion schedule) — pre-registered validation
of the DefCon EP overlay (apex_fpl.models.defensive.defcon_model) on
real outcomes, walk-forward across GW7-38 of 2025/26 — the ONLY season
that can provide this evidence (DefCon began 2025/26; 2026/27's own
real gameweeks are the only future source of a second season, a
calendar constraint, not an effort one). This is necessarily a
single-season validation. It is reported as exactly that -- not
described as multi-season-confirmed the way this project's other
promoted components are, and not silently treated as if it met that
higher bar.

**Pre-registered gate (fixed before running):** ONE comparison, not
several — overlay_pred = baseline_pred + defcon_ep (added only for
DEF/MID/FWD; GK is structurally excluded, see defcon_model.py) versus
baseline_pred alone, pooled per-player-gameweek MAE against real
`total_points` (which for 2025/26 already includes real DefCon points
earned), block-bootstrapped (block = one gameweek) at alpha=0.05. No
threshold sweep this time (unlike Block 2.1's bonus overlay) — DefCon's
threshold is a fixed real game rule (10/12 qualifying actions), not a
design choice being searched over, so there is only one model to test,
not several candidates needing a multiple-comparison correction.

Walk-forward discipline: for gameweek t, each player's forecast uses
ONLY their own real action counts from gameweeks strictly before t in
2025/26 — no future information, no cross-season leakage (there is no
other season to leak from). half_life=3.0 and max_window=15 are reused
from apex_fpl.models.minutes.challengers.exponential_decay's own
defaults, not tuned on this data.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.backtesting.scenario_builder import build_gameweek_scenario_data
from apex_fpl.models.defensive import defcon_model as dc
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "defcon_validation"
SEASON = "2025-26"
GAMEWEEKS = list(range(7, 39))  # GW7-38: the same evaluation window this project uses elsewhere, extended to season's end since only 38 GWs exist and every one is real, unique DefCon evidence
N_BOOTSTRAP = 5000
SEED = 2026
ALPHA = 0.05


def run_one_gameweek(all_rows: list[dict], gw: int) -> list[dict] | None:
    try:
        data = build_gameweek_scenario_data(SEASON, gw)
    except BlankGameweekError:
        return None

    rules = scoring.load_scoring_rules("2026_27")
    baseline_results = mc.simulate_gameweek(data.fixture_inputs, data.players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)

    history_rows = [r for r in all_rows if int(r["GW"]) < gw]
    actions_by_player: dict[str, list[int]] = defaultdict(list)
    for r in sorted(history_rows, key=lambda r: int(r["GW"])):
        pos = r.get("position")
        if pos not in ("DEF", "MID", "FWD"):
            continue
        actions_by_player[r["element"]].append(dc.defcon_action_count(r, pos))

    actual_by_player = {r["element"]: r for r in data.target_rows}
    rows = []
    for pid, meta in data.candidates_meta.items():
        if pid not in baseline_results or pid not in actual_by_player:
            continue
        position = meta["position"]
        hist = actions_by_player.get(pid, [])
        defcon_ep = dc.forecast_defcon_expected_points(hist, position)
        rows.append({
            "player_id": pid, "position": position,
            "baseline_pred": baseline_results[pid].mean_points,
            "defcon_ep": defcon_ep,
            "actual_total": int(actual_by_player[pid]["total_points"]),
            "actual_defcon_hit": int(dc.defcon_action_count(actual_by_player[pid], position) >= (dc.DEFCON_THRESHOLDS.get(position) or 10**9)),
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
    print(f"=== DefCon EP overlay validation: {SEASON}, GW{GAMEWEEKS[0]}-{GAMEWEEKS[-1]} (single-season evidence ceiling, disclosed) ===\n")

    all_rows = vl.load_merged_gw(SEASON)
    all_rows = [r for r in all_rows if r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]

    per_gw_rows: dict[int, list[dict]] = {}
    blanks = []
    for gw in GAMEWEEKS:
        rows = run_one_gameweek(all_rows, gw)
        if rows is None:
            print(f"GW{gw}: blank gameweek, skipped")
            blanks.append(gw)
            continue
        per_gw_rows[gw] = rows
        n_hits = sum(1 for r in rows if r["actual_defcon_hit"])
        print(f"GW{gw:2d}: n={len(rows):3d} outfield players, real DefCon hits={n_hits}")

    gw_diffs = []
    for gw, rows in per_gw_rows.items():
        baseline_err = np.array([abs(r["baseline_pred"] - r["actual_total"]) for r in rows])
        overlay_err = np.array([abs(r["baseline_pred"] + r["defcon_ep"] - r["actual_total"]) for r in rows])
        gw_diffs.append(float(overlay_err.mean() - baseline_err.mean()))

    diffs = np.array(gw_diffs)
    mean, lo, hi = bootstrap_ci(diffs)
    excludes_zero = lo > 0 or hi < 0
    promoted = excludes_zero and mean < 0

    all_defcon_ep = np.array([r["defcon_ep"] for rows in per_gw_rows.values() for r in rows])
    all_actual_hit = np.array([r["actual_defcon_hit"] for rows in per_gw_rows.values() for r in rows])
    pooled_corr = float(np.corrcoef(all_defcon_ep, all_actual_hit)[0, 1]) if all_defcon_ep.std() > 0 else float("nan")

    summary = {
        "season": SEASON, "n_gameweeks": len(per_gw_rows), "n_blank": len(blanks), "blanks": blanks,
        "single_season_evidence_ceiling": True,
        "mae_diff_mean": mean, "ci95": [lo, hi], "excludes_zero": excludes_zero, "promoted": promoted,
        "pooled_corr_defcon_ep_vs_actual_hit": pooled_corr,
        "per_gw_rows": {str(k): v for k, v in per_gw_rows.items()},
    }
    (ARTIFACT_DIR / "validation_results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Result ({len(per_gw_rows)} gameweeks, {len(blanks)} blank) ===")
    print(f"MAE diff (overlay - baseline): {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excludes_zero={excludes_zero}")
    print(f"Pooled correlation(defcon_ep, actual_hit): {pooled_corr:.3f}")
    print(f"\n=== DECISION: {'PROMOTE (single-season evidence)' if promoted else 'NOT PROMOTED'} ===")
    print(f"Written to {(ARTIFACT_DIR / 'validation_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
