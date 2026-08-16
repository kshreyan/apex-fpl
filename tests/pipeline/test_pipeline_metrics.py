"""Synthetic prediction/result pairs with hand-computed expected metric
values, per the brief this was built from -- including a deliberately
badly-calibrated set specifically to prove the metrics detect it, not
just that they run without crashing.
"""
from __future__ import annotations

import json

from pipeline import metrics


def _write_ledger(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(metrics, "RAW_DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(metrics, "CALIBRATION_PATH", tmp_path / "calibration.json")
    monkeypatch.setattr(metrics, "_git_sha", lambda: "deadbeef")


def _binary_call(call_id, predicted_probability, outcome, kind="captain", player_id="1"):
    return {"call_id": call_id, "type": "binary_probability", "subject": {"kind": kind, "player_id": player_id, "threshold": 6}, "predicted_probability": predicted_probability, "outcome": outcome}


def _points_call(call_id, predicted, actual, kind="player", player_id="1"):
    return {"call_id": call_id, "type": "points_forecast", "subject": {"kind": kind, "player_id": player_id}, "predicted": predicted, "actual": actual, "error": round(actual - predicted, 3)}


def _scored_result(gw, call_results, avg=40, template=42, top10k=None, starting_xi_points=20, captain_actual=6):
    return {
        "status": "SCORED", "gameweek": gw, "call_results": call_results,
        "squad_actual": {"starting_xi_points": starting_xi_points, "captain_actual_points": captain_actual, "total_points": starting_xi_points + captain_actual},
        "baselines": {"average_manager_score": avg, "template_team_score": template, "top_10k_average": top10k},
    }


def test_brier_score_matches_hand_computed_value(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    # (p=0.8, outcome=True) -> (0.8-1)^2=0.04 ; (p=0.3, outcome=False) -> (0.3-0)^2=0.09 ; mean=0.065
    calls = [_binary_call("c1", 0.8, True), _binary_call("c2", 0.3, False)]
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    assert cal["probability_metrics"]["brier_score"] == 0.065


def test_deliberately_overconfident_predictions_show_bad_calibration(monkeypatch, tmp_path):
    """The core check the brief asks for: a model that says 90% confident
    but is right only ~10% of the time must show up as badly calibrated,
    not just 'the code ran'."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    calls = []
    for i in range(20):
        outcome = i < 2  # only 2 of 20 actually hit, despite 0.9 predicted every time
        calls.append(_binary_call(f"c{i}", 0.9, outcome, player_id=str(i)))
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    # a well-calibrated model would have brier near 0.09 (0.9 confident, 90% hit rate);
    # this one is wildly overconfident and must show a much higher (worse) brier score.
    assert cal["probability_metrics"]["brier_score"] > 0.6
    # the 0.9 bin (n=20, above the suppression threshold) must show the real gap
    bin_90 = next(b for b in cal["probability_metrics"]["calibration_bins"] if b["bin_range"][0] == 0.9)
    assert bin_90["suppressed"] is False
    assert bin_90["predicted_mean"] == 0.9
    assert bin_90["actual_rate"] == 0.1  # 2/20 -- the real, honest gap between claimed and actual confidence


def test_well_calibrated_predictions_show_good_calibration(monkeypatch, tmp_path):
    """The contrast case: proves the metric can also correctly recognize
    GOOD calibration, not just flag bad calibration by default."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    calls = []
    for i in range(20):
        outcome = i < 18  # 18/20 = 90% hit rate, matching the 0.9 predicted confidence
        calls.append(_binary_call(f"c{i}", 0.9, outcome, player_id=str(i)))
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    assert cal["probability_metrics"]["brier_score"] < 0.1
    bin_90 = next(b for b in cal["probability_metrics"]["calibration_bins"] if b["bin_range"][0] == 0.9)
    assert bin_90["actual_rate"] == 0.9


def test_calibration_bin_below_min_n_is_suppressed(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    calls = [_binary_call(f"c{i}", 0.5, True, player_id=str(i)) for i in range(3)]  # only 3, below min_n=10
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    bin_50 = next(b for b in cal["probability_metrics"]["calibration_bins"] if b["bin_range"][0] == 0.5)
    assert bin_50["n"] == 3
    assert bin_50["suppressed"] is True
    assert bin_50["predicted_mean"] is None  # count shown, rate NOT shown -- not a misleading number from 3 observations
    assert bin_50["actual_rate"] is None


def test_points_forecast_mae_rmse_mean_error_match_hand_computed_values(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    # errors: +2, -4, +0 -> abs errors 2,4,0 -> mae=2.0; squared 4,16,0 -> rmse=sqrt(20/3)=2.5820; mean_error=(2-4+0)/3=-0.6667
    calls = [_points_call("c1", predicted=3.0, actual=5.0), _points_call("c2", predicted=8.0, actual=4.0), _points_call("c3", predicted=1.0, actual=1.0)]
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()
    m = cal["points_forecast_metrics"]

    assert m["n"] == 3
    assert m["mae"] == 2.0
    assert abs(m["rmse"] - 2.582) < 0.001
    assert abs(m["mean_error"] - (-0.6667)) < 0.001


def test_captaincy_hit_rate_matches_hand_computed_hits_and_misses(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    # GW1: captain (player "2") scores 10, teammate "1" scores 5 -> captain IS the top scorer -> HIT
    gw1_calls = [
        _points_call("gw01-player-1", 3.0, 5.0, kind="player", player_id="1"),
        _points_call("gw01-player-2", 3.0, 10.0, kind="player", player_id="2"),
        _points_call("gw01-captain", 6.0, 20.0, kind="captain", player_id="2"),
    ]
    # GW2: captain (player "2") scores 2, teammate "1" scores 9 -> captain is NOT top scorer -> MISS
    gw2_calls = [
        _points_call("gw02-player-1", 3.0, 9.0, kind="player", player_id="1"),
        _points_call("gw02-player-2", 3.0, 2.0, kind="player", player_id="2"),
        _points_call("gw02-captain", 6.0, 4.0, kind="captain", player_id="2"),
    ]
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, gw1_calls)])
    _write_ledger(tmp_path / "results" / "gw02.jsonl", [_scored_result(2, gw2_calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])
    _write_ledger(tmp_path / "predictions" / "gw02.jsonl", [{"status": "PUBLISHED", "gameweek": 2}])

    cal = metrics.build_calibration()
    cap = cal["captaincy"]

    assert cap["n"] == 2
    assert cap["hits"] == 1
    assert cap["misses"] == 1
    assert cap["hit_rate"] is None  # n=2 < MIN_N_FOR_RATE_DISPLAY=10 -- suppressed, not a misleading 50%
    assert cap["hit_rate_suppressed"] is True
    assert {"gameweek": 1, "hit": True} in cap["per_gameweek"]
    assert {"gameweek": 2, "hit": False} in cap["per_gameweek"]


def test_points_vs_baselines_cumulative_sums_and_diffs(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    r1 = _scored_result(1, [], avg=40, template=45, top10k=55, starting_xi_points=30, captain_actual=10)  # total=40
    r2 = _scored_result(2, [], avg=35, template=38, top10k=None, starting_xi_points=20, captain_actual=8)  # total=28, top10k unavailable this week
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [r1])
    _write_ledger(tmp_path / "results" / "gw02.jsonl", [r2])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])
    _write_ledger(tmp_path / "predictions" / "gw02.jsonl", [{"status": "PUBLISHED", "gameweek": 2}])

    cal = metrics.build_calibration()
    b = cal["points_vs_baselines"]

    assert b["cumulative_model_points"] == 68  # 40 + 28
    assert b["cumulative_average_manager_points"] == 75  # 40 + 35
    assert b["cumulative_template_team_points"] == 83  # 45 + 38
    assert b["cumulative_top_10k_points"] == 55  # only GW1 had a value -- not silently treated as 0 for GW2
    assert b["diff_vs_average_manager"] == 68 - 75
    assert b["diff_vs_template_team"] == 68 - 83


def test_resolves_to_latest_result_line_not_a_superseded_one(monkeypatch, tmp_path):
    """A correction (score.py --correct) must replace the original in
    every metric, not get pooled alongside it."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    original = _scored_result(1, [_points_call("c1", 3.0, 5.0)], starting_xi_points=10, captain_actual=5)
    corrected = _scored_result(1, [_points_call("c1", 3.0, 999.0)], starting_xi_points=10, captain_actual=5)  # deliberately extreme, to prove which one won
    corrected["supersedes"] = "whatever"
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [original, corrected])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    assert cal["points_forecast_metrics"]["n"] == 1  # not 2 -- only the latest line's calls are pooled
    assert cal["points_forecast_metrics"]["mae"] == abs(999.0 - 3.0)


def test_coverage_identifies_published_scored_and_blank_gameweeks(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])
    _write_ledger(tmp_path / "predictions" / "gw02.jsonl", [{"status": "BLANK_GAMEWEEK", "gameweek": 2}])
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, [])])

    cal = metrics.build_calibration()
    cov = cal["coverage"]

    assert cov["gameweeks_published"] == [1, 2]
    assert cov["gameweeks_scored"] == [1]
    assert cov["gameweeks_blank"] == [2]


def test_biggest_misses_ranks_points_calls_by_absolute_error(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    calls = [
        _points_call("small", predicted=3.0, actual=4.0, player_id="1"),   # error +1
        _points_call("huge", predicted=2.0, actual=15.0, player_id="2"),   # error +13 -- biggest
        _points_call("medium", predicted=8.0, actual=2.0, player_id="3"),  # error -6
    ]
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()
    misses = cal["biggest_misses"]["points_forecast"]

    assert [m["call_id"] for m in misses] == ["huge", "medium", "small"]
    assert misses[0]["gameweek"] == 1
    assert misses[0]["error"] == 13.0


def test_biggest_misses_ranks_probability_calls_by_brier_contribution_not_just_wrongness(monkeypatch, tmp_path):
    """A confident wrong call must outrank a hedged wrong call -- both are
    'misses' in the outcome sense, but only one is a confident miss."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    calls = [
        _binary_call("hedged", predicted_probability=0.55, outcome=False, player_id="1"),   # (0.55)^2 = 0.3025
        _binary_call("confident", predicted_probability=0.95, outcome=False, player_id="2"),  # (0.95)^2 = 0.9025 -- biggest
    ]
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()
    misses = cal["biggest_misses"]["binary_probability"]

    assert [m["call_id"] for m in misses] == ["confident", "hedged"]
    assert misses[0]["brier_contribution"] == 0.9025


def test_biggest_misses_caps_at_top_n_and_is_empty_when_nothing_scored(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    calls = [_points_call(f"c{i}", predicted=0.0, actual=float(i), player_id=str(i)) for i in range(10)]
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, calls)])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    assert len(cal["biggest_misses"]["points_forecast"]) == metrics.BIGGEST_MISSES_TOP_N


def test_biggest_misses_is_empty_dict_shape_with_no_scored_gameweeks(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    cal = metrics.build_calibration()

    assert cal["biggest_misses"] == {"points_forecast": [], "binary_probability": []}


def test_rebuild_is_deterministic_not_incremental(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_ledger(tmp_path / "results" / "gw01.jsonl", [_scored_result(1, [_points_call("c1", 3.0, 5.0)])])
    _write_ledger(tmp_path / "predictions" / "gw01.jsonl", [{"status": "PUBLISHED", "gameweek": 1}])

    first = metrics.build_calibration()
    second = metrics.build_calibration()

    assert first["points_forecast_metrics"]["n"] == second["points_forecast_metrics"]["n"] == 1  # not doubled by calling twice
