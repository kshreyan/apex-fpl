from __future__ import annotations

import math

import numpy as np

from apex_fpl.optimization import chips


def test_value_bench_boost_sums_bench_ep():
    assert chips.value_bench_boost([2.0, 5.5, 0.0, 3.2]) == 10.7


def test_value_bench_boost_empty_bench_is_zero():
    assert chips.value_bench_boost([]) == 0.0


def test_value_triple_captain_returns_captain_ep():
    assert chips.value_triple_captain(7.3) == 7.3


def test_value_free_hit_is_the_positive_difference():
    assert chips.value_free_hit(current_xi_ep=50.0, best_possible_xi_ep=62.5) == 12.5


def test_value_free_hit_can_be_zero_or_negative_if_current_squad_is_already_optimal():
    assert chips.value_free_hit(current_xi_ep=60.0, best_possible_xi_ep=60.0) == 0.0


def test_value_wildcard_is_the_positive_difference():
    assert chips.value_wildcard(constrained_horizon_total_ep=200.0, unconstrained_horizon_total_ep=215.0) == 15.0


def test_1e_stopping_rule_never_indexes_out_of_bounds():
    for n in [1, 2, 3, 5, 10, 37]:
        values = list(range(n))
        idx = chips.apply_1e_stopping_rule(values)
        assert 0 <= idx < n


def test_1e_stopping_rule_finds_the_max_when_it_appears_after_the_observation_window():
    # observation window for n=10 is round(10/e)=4 -> indices 0-3 observed (threshold = max(1,2,3,1) = 3)
    # everything else in the window stays below 3, so the max (9, at the very end) is the
    # first candidate to exceed the threshold and gets correctly selected.
    values = [1, 2, 3, 1, 1, 1, 1, 1, 1, 9]
    idx = chips.apply_1e_stopping_rule(values)
    assert values[idx] == 9


def test_1e_stopping_rule_beats_naive_first_choice_on_average_secretary_problem():
    """The textbook guarantee: applied to n candidates in a uniformly
    random order, the 1/e rule finds the TRUE maximum with probability
    ~1/e (~36.8%), far better than always picking the first candidate
    (which only wins if the max happens to be first, probability 1/n).
    Verified empirically here rather than just trusted, matching this
    project's general practice of checking claimed statistical
    properties directly rather than assuming a well-known result applies
    exactly as expected in this specific implementation."""
    rng = np.random.default_rng(7)
    n = 20
    trials = 3000
    stopping_wins = 0
    naive_wins = 0
    for _ in range(trials):
        perm = rng.permutation(n).astype(float)
        idx = chips.apply_1e_stopping_rule(list(perm))
        if perm[idx] == perm.max():
            stopping_wins += 1
        if perm[0] == perm.max():
            naive_wins += 1

    stopping_rate = stopping_wins / trials
    naive_rate = naive_wins / trials
    assert stopping_rate > naive_rate
    # loose bounds around the theoretical 1/e ~ 0.368, generous enough not to be flaky
    assert 0.25 < stopping_rate < 0.50
    assert abs(naive_rate - 1.0 / n) < 0.03
