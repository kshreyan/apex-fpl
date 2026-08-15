from __future__ import annotations

from apex_fpl.models.attacking import challengers as ac
from apex_fpl.models.attacking import proportional as prop


def test_equal_split_gives_identical_shares():
    shares = ac.equal_split(["a", "b", "c", "d"])
    assert all(s.goal_share == 0.25 and s.assist_share == 0.25 for s in shares.values())


def test_shrinkage_shares_sum_to_one():
    history = {
        "star": [(3, 1), (2, 0), (1, 1)],
        "squad_player": [(0, 1), (0, 0), (0, 0)],
        "bench_player": [(0, 0), (0, 0), (0, 0)],
    }
    for alpha in [0.0, 1.0, 3.0, 20.0]:
        shares = ac.shrinkage_share(history, alpha=alpha)
        total_goal = sum(s.goal_share for s in shares.values())
        total_assist = sum(s.assist_share for s in shares.values())
        assert abs(total_goal - 1.0) < 1e-9, f"alpha={alpha}: goal shares sum to {total_goal}"
        assert abs(total_assist - 1.0) < 1e-9, f"alpha={alpha}: assist shares sum to {total_assist}"


def test_shrinkage_alpha_zero_matches_raw_proportional():
    history = {"a": [(2, 0), (1, 1)], "b": [(0, 1), (0, 0)]}
    shrunk = ac.shrinkage_share(history, alpha=0.0)
    raw = prop.compute_shares(history)
    for pid in history:
        assert abs(shrunk[pid].goal_share - raw[pid].goal_share) < 1e-9
        assert abs(shrunk[pid].assist_share - raw[pid].assist_share) < 1e-9


def test_shrinkage_large_alpha_approaches_equal_split():
    history = {"a": [(5, 0)], "b": [(0, 0)]}  # "a" scored everything in a tiny sample
    shrunk = ac.shrinkage_share(history, alpha=10000.0)
    assert abs(shrunk["a"].goal_share - 0.5) < 0.01
    assert abs(shrunk["b"].goal_share - 0.5) < 0.01


def test_shrinkage_pulls_small_sample_star_toward_prior():
    """The whole point of shrinkage: a player who scored the only goal in a
    tiny lookback window should NOT get a full 100% share once shrunk —
    that's exactly the small-sample noise the champion is vulnerable to."""
    history = {"lucky_scorer": [(1, 0)], "teammate": [(0, 0)], "teammate2": [(0, 0)]}
    raw = prop.compute_shares(history)
    shrunk = ac.shrinkage_share(history, alpha=3.0)
    assert raw["lucky_scorer"].goal_share == 1.0
    assert shrunk["lucky_scorer"].goal_share < 1.0
