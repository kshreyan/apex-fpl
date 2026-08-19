#!/usr/bin/env python3
"""Block 2.7 (Phase 13 promotion schedule) — re-applies Phase 9's
already-validated 1/e stopping rule (apex_fpl.optimization.chips.
apply_1e_stopping_rule, unchanged) to the REAL two-half 2026/27 chip
window structure (apex_fpl.rules.chip_windows), instead of the single
open GW2-38 window that script's own demo explicitly disclosed as a
methodological simplification ("not a claim about what was actually
legal"). Reuses the already-computed real per-gameweek Bench Boost /
Triple Captain EP values from that demo
(artifacts/phase9_chip_valuation/chip_valuation_results.json,
2022-23, GW2-37) — no new simulation, since the underlying EP values
don't depend on chip-window rules at all, only which gameweek the
stopping rule is allowed to consider.

**What this tests:** does the window-boundary correction (one 36-
gameweek window -> two independent ~18/18-gameweek windows) change the
stopping rule's recommended gameweek or captured value materially? This
is a real behavioral question, not just documentation -- the 1/e rule's
observation phase length depends directly on window size, so halving
the window changes WHEN it stops observing and starts committing.

Caveat carried over honestly, unchanged from the original demo: 2022-23
did not actually have this exact chip structure (this project has not
audited historical chip rules for that season) -- the real GW2-37 EP
VALUES are real, but applying 2026/27's window boundaries to a
different season's calendar is a structural stand-in for demonstrating
the corrected mechanism, not a claim that 2022-23 itself had two
19-gameweek halves. The correction is validated on the RULE's behavior,
not asserted as historically accurate for 2022-23.
"""
from __future__ import annotations

import json
from pathlib import Path

from apex_fpl.optimization import chips
from apex_fpl.rules import chip_windows as cw

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHED_RESULTS = REPO_ROOT / "artifacts" / "phase9_chip_valuation" / "chip_valuation_results.json"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "chip_window_redesign_analysis"
HALF1_BOUNDARY_GW = 19  # bboost/3xc half-1 stop_event, from configs/seasons/2026_27.yaml


def split_by_half(gws: list[int], values: list[float]) -> tuple[list[float], list[float]]:
    half1 = [v for gw, v in zip(gws, values) if gw <= HALF1_BOUNDARY_GW]
    half2 = [v for gw, v in zip(gws, values) if gw > HALF1_BOUNDARY_GW]
    return half1, half2


def analyze(label: str, gws: list[int], values: list[float]) -> dict:
    half1_values, half2_values = split_by_half(gws, values)
    half1_gws, half2_gws = [g for g in gws if g <= HALF1_BOUNDARY_GW], [g for g in gws if g > HALF1_BOUNDARY_GW]

    original_idx = chips.apply_1e_stopping_rule(values)
    original_gw, original_value = gws[original_idx], values[original_idx]

    h1_idx = chips.apply_1e_stopping_rule(half1_values)
    h1_gw, h1_value = half1_gws[h1_idx], half1_values[h1_idx]
    h2_idx = chips.apply_1e_stopping_rule(half2_values)
    h2_gw, h2_value = half2_gws[h2_idx], half2_values[h2_idx]

    # Two-half total value: whichever half's stopping value is used, the
    # OTHER half's chip is a DIFFERENT chip instance (2026/27 grants one
    # full set of 4 per half) -- so a real manager gets BOTH half-values,
    # not a choice between them. That's the real, structural difference
    # from the single-window model, which only ever captures ONE use of
    # the chip across the whole season.
    two_half_total = h1_value + h2_value

    result = {
        "label": label,
        "single_window_36gw": {"stopping_gw": original_gw, "value": original_value},
        "two_half_corrected": {
            "half1": {"stopping_gw": h1_gw, "value": h1_value, "window_size": len(half1_values)},
            "half2": {"stopping_gw": h2_gw, "value": h2_value, "window_size": len(half2_values)},
            "total_value_both_chips": two_half_total,
        },
        "value_delta_two_half_minus_single": two_half_total - original_value,
    }
    print(f"\n=== {label} ===")
    print(f"  Single 36-GW window (original demo):     stop at GW{original_gw}, value={original_value:.3f}")
    print(f"  Two-half corrected -- half 1 (GW2-19, n={len(half1_values)}): stop at GW{h1_gw}, value={h1_value:.3f}")
    print(f"  Two-half corrected -- half 2 (GW20-37, n={len(half2_values)}): stop at GW{h2_gw}, value={h2_value:.3f}")
    print(f"  Two-half total (BOTH chip instances used): {two_half_total:.3f}  (delta vs single-window: {two_half_total - original_value:+.3f})")
    return result


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    cached = json.loads(CACHED_RESULTS.read_text())
    gws = cached["real_gws"]

    windows = cw.load_chip_windows()
    bb_window_h1 = cw.active_window("bboost", 1, windows)
    bb_window_h2 = cw.active_window("bboost", 19 + 1, windows)  # any GW in half 2
    print(f"Real 2026/27 windows loaded: bboost half 1 = GW{bb_window_h1.start_event}-{bb_window_h1.stop_event}, "
          f"half 2 = GW{bb_window_h2.start_event}-{bb_window_h2.stop_event}\n"
          f"(Note: cached EP data starts at GW2, not GW1 -- half 1's window here is effectively GW2-19, one "
          f"gameweek short of the true GW1-19 window; a real live application would include GW1's own values.)")

    bb_result = analyze("Bench Boost", gws, cached["bench_boost_values"])
    tc_result = analyze("Triple Captain", gws, cached["triple_captain_values"])

    summary = {"season": cached["season"], "bench_boost": bb_result, "triple_captain": tc_result}
    (ARTIFACT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'results.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
