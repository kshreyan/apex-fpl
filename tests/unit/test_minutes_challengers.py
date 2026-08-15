from __future__ import annotations

from apex_fpl.models.minutes import challengers as mc


def test_always_90_ignores_history():
    fc = mc.always_90([0, 0, 0])
    assert fc.p_appearance == 1.0
    assert fc.p_60_plus == 1.0
    assert fc.expected_minutes_if_played == 90.0


def test_persistence_uses_only_last_match():
    fc = mc.persistence([90, 90, 0])  # most recent (last) is 0
    assert fc.p_appearance == 0.0
    assert fc.p_60_plus == 0.0

    fc2 = mc.persistence([0, 0, 75])  # most recent (last) is 75
    assert fc2.p_appearance == 1.0
    assert fc2.p_60_plus == 1.0
    assert fc2.expected_minutes_if_played == 75.0


def test_persistence_neutral_prior_when_empty():
    fc = mc.persistence([])
    assert fc.n_history_gws == 0


def test_exponential_decay_weights_recent_more_than_old():
    # A player who used to start every week but has been benched the last
    # 3 weeks should have a LOWER p_appearance under exponential decay than
    # under a flat average over the same window, because decay emphasizes
    # the recent zeros more.
    history = [90] * 6 + [0] * 3
    decayed = mc.exponential_decay(history, half_life_matches=2.0)
    flat_avg = sum(1 for m in history if m > 0) / len(history)
    assert decayed.p_appearance < flat_avg


def test_exponential_decay_neutral_prior_when_empty():
    fc = mc.exponential_decay([])
    assert fc.n_history_gws == 0


def test_exponential_decay_full_confidence_when_all_recent_starts():
    fc = mc.exponential_decay([90, 90, 90, 90], half_life_matches=3.0)
    assert fc.p_appearance == 1.0
    assert fc.p_60_plus == 1.0
    assert fc.expected_minutes_if_played == 90.0
