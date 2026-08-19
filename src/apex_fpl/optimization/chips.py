"""Chip valuation (Phase 9; research_plan.md's "dynamic chip valuation
with option-value framing").

Values each of the 4 chip types on real, per-gameweek simulated EP data,
reusing existing squad/simulation/transfer machinery rather than
building new forecasting logic:

- **Bench Boost**: marginal value = EP of the CURRENT bench that
  gameweek (their points already exist in the simulation; bench boost
  just makes them count toward the total).
- **Triple Captain**: marginal value = EP of the current captain that
  gameweek (the extra 1x beyond the normal 2x multiplier already applied
  by the squad optimizer).
- **Free Hit**: marginal value = EP(best UNCONSTRAINED one-gameweek
  squad under budget) − EP(current squad's best XI), for that gameweek
  only — correct because a Free Hit squad reverts completely after the
  gameweek, so its entire value genuinely IS confined to that one week.
- **Wildcard**: marginal value = the FULL multi-gameweek benefit of an
  unconstrained rebuild this week, meant to be computed by comparing
  `apex_fpl.optimization.transfers.rolling_horizon_transfers` with
  free transfers effectively unlimited for just this gameweek against
  continuing the normal transfer-constrained policy over the SAME
  forecast horizon — unlike Free Hit, a wildcard's squad persists, so
  its value must include future gameweeks too, not just the play week.
  `value_wildcard` here just takes the two already-computed horizon
  totals; assembling them is the caller's job (see
  scripts/run_phase9_chip_valuation_demo.py).

Bench Boost and Triple Captain are "memoryless" single-use options —
playing (or not playing) one doesn't change future squad state or future
chip values — so the classic optimal-stopping "when should I play my
one-shot option" question applies cleanly. `apply_1e_stopping_rule`
implements the textbook 1/e observe-then-commit rule (a form of the
"secretary problem," Ferguson 1989): observe the first ~1/e fraction of
the decision window purely to calibrate a threshold (the best value seen
so far), then commit to the first candidate afterward that beats
everything seen — a real, well-established stopping rule with a proven
theoretical success rate (~1/e ≈ 36.8% chance of finding the TRUE best
option among n candidates seen in random order), not an invented
heuristic. Chosen specifically because it's the natural way to formalize
"option value" — the value of PATIENCE — for a chip that can only be
used once within a window and whose future candidates aren't known in
advance (a real manager doesn't get to see gameweek 30's fixtures before
gameweek 10).
"""
from __future__ import annotations

import math


def value_bench_boost(bench_ep: list[float]) -> float:
    return float(sum(bench_ep))


def value_triple_captain(captain_ep: float) -> float:
    return float(captain_ep)


def value_free_hit(current_xi_ep: float, best_possible_xi_ep: float) -> float:
    return float(best_possible_xi_ep - current_xi_ep)


def value_wildcard(constrained_horizon_total_ep: float, unconstrained_horizon_total_ep: float) -> float:
    return float(unconstrained_horizon_total_ep - constrained_horizon_total_ep)


def should_play_chip_now(values_so_far: list[float], window_size: int) -> bool:
    """Online/live counterpart to `apply_1e_stopping_rule` (Phase 13
    Block 2.8 (a)) — that function only ever answers "given the WHOLE
    sequence in hindsight, which index would the rule have picked,"
    useful for backtesting but not directly usable week-by-week in
    production, where only the values observed SO FAR are known.
    `values_so_far[-1]` is this gameweek's own value; `window_size` is
    the total length of the decision window (e.g. 19 for a first-half
    chip). Returns whether the rule says to play THIS gameweek.

    Reproduces the exact same decision boundary as
    `apply_1e_stopping_rule` when evaluated incrementally: during the
    observation phase (the first round(window_size/e) candidates),
    always returns False (still calibrating the threshold, matching the
    offline rule never selecting an index inside that range). After the
    observation phase, returns True iff this gameweek's value exceeds
    the best value seen during observation, OR this is the LAST
    gameweek of the window (forced use-it-or-lose-it, matching the
    offline rule's own fallback to the last candidate) --
    `test_apply_1e_stopping_rule_online_matches_offline_on_full_sequences`
    verifies this equivalence directly, not just by construction."""
    n = len(values_so_far)
    if n == 0:
        raise ValueError("no candidates")
    if window_size < n:
        raise ValueError(f"window_size ({window_size}) shorter than values_so_far ({n})")
    r = max(1, round(window_size / math.e))
    if n <= r:
        return n == window_size  # window somehow already exhausted inside the observation phase (tiny window)
    threshold = max(values_so_far[:r])
    if values_so_far[-1] > threshold:
        return True
    return n == window_size


def apply_1e_stopping_rule(values: list[float]) -> int:
    """Returns the 0-based index into `values` chosen by the classical
    1/e observe-then-commit stopping rule, processing values IN ORDER
    (as if seeing gameweeks sequentially, with no knowledge of the
    future) — never looks ahead. Observes (skips) the first
    round(n/e) candidates purely to calibrate a threshold, then commits
    to the first candidate afterward whose value exceeds that threshold.
    If none ever does, commits to the LAST candidate — a real, forced
    decision, since a chip unused by the end of its window is simply
    wasted; "hold forever" is not an available option in real FPL."""
    n = len(values)
    if n == 0:
        raise ValueError("no candidates")
    if n == 1:
        return 0
    r = max(1, round(n / math.e))
    threshold = max(values[:r])
    for i in range(r, n):
        if values[i] > threshold:
            return i
    return n - 1
