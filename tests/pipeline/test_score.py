from __future__ import annotations

import json

import pytest

from pipeline import predict, score

from api_payloads import make_bootstrap_static, make_element, make_event, make_fixture


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass


def _install_fake_network(monkeypatch, bootstrap_static: dict, fixtures: list):
    def fake_get(url, timeout, headers):
        if "bootstrap-static" in url:
            return _FakeResponse(json.dumps(bootstrap_static).encode())
        return _FakeResponse(json.dumps(fixtures).encode())

    monkeypatch.setattr(score.fpl_client.bronze.requests, "get", fake_get)


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(score, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(score, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(score, "RAW_DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(score, "STANDINGS_DIR", tmp_path / "raw" / "standings")
    monkeypatch.setattr(score.fpl_client, "RAW_DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(score.silver, "run_build", lambda bronze_root=None: {})


def _write_prediction_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _base_prediction(gw=1, record_id="pred1", supersedes=None, status="PUBLISHED"):
    return {
        "schema_version": "1.0", "record_id": record_id, "supersedes": supersedes,
        "gameweek": gw, "status": status,
        "squad": {
            "starting_xi": [{"player_id": "1"}, {"player_id": "2"}],
            "captain_player_id": "2",
        } if status == "PUBLISHED" else None,
        "calls": [
            {"id": f"gw{gw:02d}-player-1-points", "type": "points_forecast", "subject": {"kind": "player", "player_id": "1"}, "value": 3.0},
            {"id": f"gw{gw:02d}-captain", "type": "points_forecast", "subject": {"kind": "captain", "player_id": "2"}, "value": 10.0},
            {"id": f"gw{gw:02d}-squad-total", "type": "points_forecast", "subject": {"kind": "squad_total"}, "value": 8.0},
            {"id": f"gw{gw:02d}-captain-haul", "type": "binary_probability", "subject": {"kind": "captain", "player_id": "2", "threshold": 6}, "probability": 0.4},
        ] if status == "PUBLISHED" else [],
    }


SETTLED_DEADLINE = "2026-08-14T17:30:00Z"  # safely in the past


def _settled_bootstrap(gw=1, average_entry_score=45, player_points=None):
    player_points = player_points or {}
    elements = [make_element(i, team=1, element_type=2, event_points=player_points.get(str(i), 0)) for i in range(1, 5)]
    return make_bootstrap_static(
        [make_event(gw, SETTLED_DEADLINE, finished=True, data_checked=True, average_entry_score=average_entry_score)],
        elements=elements,
    )


def _settled_fixtures(gw=1):
    return [make_fixture(1, gw, team_h=1, team_a=2, kickoff_time=SETTLED_DEADLINE, finished=True, team_h_score=2, team_a_score=1)]


def test_no_prediction_means_nothing_to_score(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    result = score.score_gameweek(1)
    assert result is None
    assert not (tmp_path / "results").exists()


def test_not_yet_settled_means_nothing_to_score(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred_path = tmp_path / "predictions" / "gw01.jsonl"
    _write_prediction_line(pred_path, _base_prediction())

    bs = make_bootstrap_static([make_event(1, SETTLED_DEADLINE, finished=True, data_checked=False, average_entry_score=0)])  # finished but NOT data_checked
    _install_fake_network(monkeypatch, bs, _settled_fixtures())

    result = score.score_gameweek(1)
    assert result is None
    assert not (tmp_path / "results" / "gw01.jsonl").exists()


def test_scores_correctly_with_hand_computed_values(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred_path = tmp_path / "predictions" / "gw01.jsonl"
    _write_prediction_line(pred_path, _base_prediction())

    # player 1 actually scored 5, player 2 (captain) actually scored 8
    bs = _settled_bootstrap(average_entry_score=45, player_points={"1": 5, "2": 8})
    _install_fake_network(monkeypatch, bs, _settled_fixtures())

    result = score.score_gameweek(1)

    assert result["status"] == "SCORED"
    by_id = {c["call_id"]: c for c in result["call_results"]}
    assert by_id["gw01-player-1-points"]["actual"] == 5
    assert by_id["gw01-player-1-points"]["error"] == 5 - 3.0
    # subject is denormalized onto every call_result -- a reader must be able to tell
    # "this is the captain's result" from the result record alone, without cross-
    # referencing the prediction ledger (the same self-containment principle this
    # project already applied to the binary-probability threshold).
    assert by_id["gw01-captain"]["subject"]["kind"] == "captain"
    assert by_id["gw01-player-1-points"]["subject"]["kind"] == "player"
    assert by_id["gw01-captain"]["actual"] == 16  # 8 doubled
    assert by_id["gw01-captain"]["error"] == 16 - 10.0
    assert by_id["gw01-squad-total"]["actual"] == 5 + 8 + 8  # starters (1,2) + captain(2) doubled contribution
    haul = by_id["gw01-captain-haul"]
    assert haul["outcome"] is True  # actual undoubled captain points (8) >= threshold (6)
    assert result["squad_actual"]["starting_xi_points"] == 13  # player 1 (5) + player 2 (8)
    assert result["squad_actual"]["total_points"] == 21  # 13 + captain's extra 8
    assert result["baselines"]["average_manager_score"] == 45
    assert result["baselines"]["top_10k_average"] is None  # no extract file exists


def test_captain_haul_outcome_false_when_below_threshold(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction_line(tmp_path / "predictions" / "gw01.jsonl", _base_prediction())
    bs = _settled_bootstrap(player_points={"1": 1, "2": 2})  # captain scores only 2, below threshold 6
    _install_fake_network(monkeypatch, bs, _settled_fixtures())

    result = score.score_gameweek(1)
    haul = next(c for c in result["call_results"] if c["call_id"] == "gw01-captain-haul")
    assert haul["outcome"] is False


def test_already_scored_is_not_rescored_without_correct_flag(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction_line(tmp_path / "predictions" / "gw01.jsonl", _base_prediction())
    bs = _settled_bootstrap(player_points={"1": 5, "2": 8})
    _install_fake_network(monkeypatch, bs, _settled_fixtures())

    first = score.score_gameweek(1)
    second = score.score_gameweek(1)  # no correct_reason

    assert second is None
    lines = (tmp_path / "results" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 1  # not duplicated


def test_correct_appends_a_new_result_with_reason_and_supersede_chain(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction_line(tmp_path / "predictions" / "gw01.jsonl", _base_prediction())
    bs = _settled_bootstrap(player_points={"1": 5, "2": 8})
    _install_fake_network(monkeypatch, bs, _settled_fixtures())

    first = score.score_gameweek(1)
    second = score.score_gameweek(1, correct_reason="scoring_bug")

    assert second is not None
    assert second["supersedes"] == first["record_id"]
    assert second["supersede_reason"] == "scoring_bug"
    lines = (tmp_path / "results" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 2  # both preserved, nothing overwritten


def test_correct_without_reason_is_rejected():
    with pytest.raises(ValueError, match="invalid correct_reason"):
        score.score_gameweek(1, correct_reason="not_a_real_reason")


def test_correct_on_a_never_scored_gameweek_is_rejected(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction_line(tmp_path / "predictions" / "gw01.jsonl", _base_prediction())
    with pytest.raises(ValueError, match="never been scored"):
        score.score_gameweek(1, correct_reason="scoring_bug")


def test_blank_gameweek_prediction_produces_a_blank_result_without_fetching_settlement(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction_line(tmp_path / "predictions" / "gw01.jsonl", _base_prediction(status="BLANK_GAMEWEEK"))

    def _explode(url, timeout, headers):
        raise AssertionError("must not fetch settlement data for a blank-gameweek prediction")

    monkeypatch.setattr(score.fpl_client.bronze.requests, "get", _explode)

    result = score.score_gameweek(1)

    assert result["status"] == "BLANK_GAMEWEEK_NO_SCORING"
    assert result["call_results"] == []


def test_compute_template_team_picks_highest_ownership_per_position():
    players = {
        "g1": {"position": "GK", "team": "A", "selected_by_percent": 50.0},
        "g2": {"position": "GK", "team": "A", "selected_by_percent": 10.0},
        "g3": {"position": "GK", "team": "A", "selected_by_percent": 5.0},
    }
    # add enough DEF/MID/FWD to satisfy quotas
    for i, pos in enumerate(["DEF"] * 6 + ["MID"] * 6 + ["FWD"] * 4):
        players[f"{pos}{i}"] = {"position": pos, "team": "A", "selected_by_percent": 100.0 - i}

    template = score.compute_template_team(players)

    assert "g1" in template["starting_xi"] or "g1" == template["captain_player_id"] or True  # sanity: no crash
    assert len(template["starting_xi"]) == 11
    assert template["captain_player_id"] in template["starting_xi"]
    # the single highest-ownership GK (g1, 50%) must be picked over g2/g3 -- GK quota is exactly 2,
    # and only 1 starts, so g1 (highest) must start.
    assert "g1" in template["starting_xi"]


def test_read_only_on_predictions_directory(monkeypatch, tmp_path):
    """Structural enforcement check: no write-mode file handle is ever
    opened against PREDICTIONS_DIR anywhere score.py's code path touches."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction_line(tmp_path / "predictions" / "gw01.jsonl", _base_prediction())
    bs = _settled_bootstrap(player_points={"1": 5, "2": 8})
    _install_fake_network(monkeypatch, bs, _settled_fixtures())

    import pathlib

    real_open = pathlib.Path.open

    def guarded_open(self, mode="r", *a, **k):
        if "predictions" in self.parts and ("w" in mode or "a" in mode or "x" in mode):
            raise AssertionError(f"attempted to open {self} in mode {mode!r} -- predictions must be read-only")
        return real_open(self, mode, *a, **k)

    monkeypatch.setattr(pathlib.Path, "open", guarded_open)

    score.score_gameweek(1)  # must not raise
