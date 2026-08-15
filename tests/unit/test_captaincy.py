from __future__ import annotations

import numpy as np

from apex_fpl.optimization import captaincy as capt
from apex_fpl.simulation.monte_carlo import PlayerSimResult


def _result(pid, points, minutes):
    points = np.array(points, dtype=float)
    minutes = np.array(minutes, dtype=float)
    return PlayerSimResult(pid, float(points.mean()), float(points.std()), points, minutes)


def test_captain_profile_matches_empirical_frequencies():
    # 10000 sims: 20% blank (0-1 pts), rest a mix including some big hauls
    rng = np.random.default_rng(0)
    n = 10000
    pts = np.concatenate([np.zeros(2000), rng.poisson(6, n - 2000).astype(float)])
    minutes = np.where(pts == 0, rng.choice([0, 90], size=n, p=[0.3, 0.7]), 90)
    result = _result("p", pts, minutes)
    profile = capt.captain_profile(result)

    assert abs(profile.mean_points - pts.mean()) < 1e-9
    assert abs(profile.p_blank - np.mean(pts <= 1)) < 1e-9
    assert abs(profile.p_10_plus - np.mean(pts >= 10)) < 1e-9
    assert 0 <= profile.p_no_appearance <= 1


def test_captain_bonus_uses_vice_when_captain_blanks():
    n = 5
    captain = _result("cap", [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])  # never plays
    vice = _result("vc", [8, 5, 3, 7, 2], [90, 90, 90, 90, 90])  # always plays
    bonus = capt.captain_bonus_points(captain, vice)
    assert list(bonus) == [8, 5, 3, 7, 2]


def test_captain_bonus_uses_captain_when_captain_plays():
    captain = _result("cap", [10, 8, 6], [90, 90, 90])
    vice = _result("vc", [1, 1, 1], [90, 90, 90])
    bonus = capt.captain_bonus_points(captain, vice)
    assert list(bonus) == [10, 8, 6]


def test_captain_bonus_wasted_when_neither_plays():
    captain = _result("cap", [0, 0], [0, 0])
    vice = _result("vc", [0, 0], [0, 0])
    bonus = capt.captain_bonus_points(captain, vice)
    assert list(bonus) == [0, 0]


def test_captain_bonus_rejects_mismatched_sample_counts():
    captain = _result("cap", [1, 2, 3], [90, 90, 90])
    vice = _result("vc", [1, 2], [90, 90])
    try:
        capt.captain_bonus_points(captain, vice)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_select_captain_ev_picks_highest_mean():
    candidates = {
        "safe": _result("safe", [4, 5, 4, 5, 4], [90] * 5),
        "explosive": _result("explosive", [0, 0, 20, 0, 20], [90] * 5),  # mean=8, higher
    }
    assert capt.select_captain_ev(candidates) == "explosive"


def test_select_captain_risk_averse_prefers_safe_floor_over_higher_mean_boom_bust():
    rng = np.random.default_rng(1)
    n = 5000
    # "safe": consistently around 5 points
    safe_pts = rng.normal(5, 0.5, n)
    # "boom_bust": mean is higher (6) but with a much worse floor (frequently blanks)
    boom_bust_pts = np.where(rng.random(n) < 0.5, 0, 12)
    candidates = {
        "safe": _result("safe", safe_pts, np.full(n, 90)),
        "boom_bust": _result("boom_bust", boom_bust_pts, np.full(n, 90)),
    }
    assert candidates["boom_bust"].mean_points > candidates["safe"].mean_points  # confirm the setup: boom_bust wins on EV
    assert capt.select_captain_ev(candidates) == "boom_bust"
    assert capt.select_captain_risk_averse(candidates, quantile=0.25) == "safe"


def test_select_captain_ceiling_prefers_high_upside_over_safe():
    rng = np.random.default_rng(2)
    n = 5000
    safe_pts = rng.normal(5, 0.5, n)
    boom_bust_pts = np.where(rng.random(n) < 0.3, 0, 15)  # high ceiling
    candidates = {
        "safe": _result("safe", safe_pts, np.full(n, 90)),
        "boom_bust": _result("boom_bust", boom_bust_pts, np.full(n, 90)),
    }
    assert capt.select_captain_ceiling(candidates, quantile=0.90) == "boom_bust"
