#!/usr/bin/env python3
"""Block 2.2 follow-up — the blanket DefCon EP overlay
(run_defcon_validation.py) confirmed NOT PROMOTED: aggregate MAE
diff +0.0083, 95% CI [+0.0056,+0.0110] (confidently worse), even though
the underlying signal is real and non-spurious (pooled correlation
0.394 between predicted DefCon EP and actual threshold-hit — notably
stronger than the equivalent bonus-overlay correlation of ~0.25 in
Block 2.1). Same root cause as the blanket bonus overlay: most players
never hit DefCon (~5% of player-gameweeks), so adding a nonzero EP to
the other ~95% adds more error than it removes from the real hitters.

This is a genuinely NEW, freshly pre-registered follow-up study (same
pattern as Phase 6's blanket-bonus failure motivating Block 2.1's
surgical bonus follow-up, itself pre-registered before running) — not
a second look at the same test. Pre-registered here, before computing
any of these numbers: 3 P(hit) threshold candidates (>= 0.3, 0.5, 0.7),
Bonferroni-corrected (alpha=0.05/3), reusing the ALREADY-COMPUTED
per-player-gameweek rows cached in
artifacts/defcon_validation/validation_results.json — no new
simulation needed, since predicted DefCon EP and actual outcomes for
every player-gameweek are already stored there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHED_RESULTS = REPO_ROOT / "artifacts" / "defcon_validation" / "validation_results.json"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "defcon_surgical_followup"
DEFCON_POINTS = 2
P_HIT_THRESHOLDS = [0.3, 0.5, 0.7]  # pre-registered -- see module docstring
N_BOOTSTRAP = 5000
SEED = 2026
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(P_HIT_THRESHOLDS)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED, alpha: float = ALPHA) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return float(values.mean()), float(np.percentile(boot, lo_pct)), float(np.percentile(boot, hi_pct))


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cached = json.loads(CACHED_RESULTS.read_text())
    per_gw_rows = cached["per_gw_rows"]
    print(f"=== DefCon surgical follow-up: {len(per_gw_rows)} gameweeks (reusing cached simulation), thresholds={P_HIT_THRESHOLDS} ===\n")

    threshold_results = {}
    for threshold in P_HIT_THRESHOLDS:
        ep_threshold = threshold * DEFCON_POINTS
        gw_diffs, gw_subset_sizes, gw_precisions = [], [], []
        for gw, rows in per_gw_rows.items():
            baseline_err = np.array([abs(r["baseline_pred"] - r["actual_total"]) for r in rows])
            overlay_pred = np.array([
                r["baseline_pred"] + r["defcon_ep"] if r["defcon_ep"] >= ep_threshold else r["baseline_pred"]
                for r in rows
            ])
            actual = np.array([r["actual_total"] for r in rows])
            overlay_err = np.abs(overlay_pred - actual)
            gw_diffs.append(float(overlay_err.mean() - baseline_err.mean()))

            adjusted = [r for r in rows if r["defcon_ep"] >= ep_threshold]
            gw_subset_sizes.append(len(adjusted))
            if adjusted:
                gw_precisions.append(sum(1 for r in adjusted if r["actual_defcon_hit"]) / len(adjusted))

        diffs = np.array(gw_diffs)
        mean, lo, hi = bootstrap_ci(diffs, alpha=ALPHA)
        _, lo_c, hi_c = bootstrap_ci(diffs, alpha=BONFERRONI_ALPHA)
        excludes_zero_corrected = lo_c > 0 or hi_c < 0
        promoted = excludes_zero_corrected and mean < 0

        threshold_results[threshold] = {
            "mae_diff_mean": mean, "ci95_nominal": [lo, hi], "ci_bonferroni": [lo_c, hi_c],
            "promoted": promoted, "mean_subset_size_per_gw": float(np.mean(gw_subset_sizes)),
            "mean_precision": float(np.mean(gw_precisions)) if gw_precisions else None,
        }
        print(f"P(hit)>={threshold} (defcon_ep>={ep_threshold}): MAE diff={mean:+.4f}, 95% CI {[round(lo,4),round(hi,4)]}, "
              f"Bonferroni CI {[round(lo_c,4),round(hi_c,4)]}, promoted={promoted}, "
              f"mean subset/GW={np.mean(gw_subset_sizes):.1f}, precision={np.mean(gw_precisions) if gw_precisions else float('nan'):.3f}")

    (ARTIFACT_DIR / "followup_results.json").write_text(json.dumps({str(k): v for k, v in threshold_results.items()}, indent=2))
    any_promoted = [t for t, r in threshold_results.items() if r["promoted"]]
    print(f"\n=== DECISION: {'PROMOTE threshold(s) ' + str(any_promoted) if any_promoted else 'NO THRESHOLD PROMOTED'} ===")
    print(f"Written to {(ARTIFACT_DIR / 'followup_results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
