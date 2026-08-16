"""Offline coverage of predict.py's branching logic -- no live API call,
no real model invocation (mocked out; the model's own correctness is
tested elsewhere in this project, this file only tests predict.py's OWN
responsibilities: phase gating, blank-gameweek handling, record shape,
supersede chaining, dry-run, and propagate-on-failure).
"""
from __future__ import annotations

import json

import pytest

from pipeline import predict

import run_production_recommendation as rpr  # noqa: E402 -- importable once pipeline.predict has run its sys.path insertion
from api_payloads import make_bootstrap_static, make_event, make_fixture


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

    monkeypatch.setattr(predict.bronze.requests, "get", fake_get)


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(predict, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(predict, "RAW_DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(predict.fpl_client, "RAW_DATA_ROOT", tmp_path / "raw")
    monkeypatch.setattr(predict.silver, "run_build", lambda bronze_root=None: {})  # Silver correctness tested elsewhere
    monkeypatch.setattr(predict, "_git_sha", lambda path=None: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


def _fake_model_result(target_gw=1, captain_id="411"):
    squad = [
        {"player_id": "4", "name": "Gabriel", "position": "DEF", "team": "Arsenal", "price": 8.0, "expected_points": 3.5},
        {"player_id": captain_id, "name": "Haaland", "position": "FWD", "team": "Man City", "price": 15.5, "expected_points": 6.9},
    ]
    return {
        "squad": squad, "starting_xi": ["4", captain_id], "bench_order": [],
        "captain": captain_id, "vice_captain_fallback": "4",
        "captain_haul_probability": 0.59, "captain_haul_threshold": 6,
        "projected_gw_points": 17.3, "caveats": ["some caveat"],
        "teams_with_double_fixture": [],
        "training_data_fingerprint": {"cold_start_minutes_seasons": [], "cold_start_minutes_hash": "x", "team_model_fallback_seasons": [], "team_model_fallback_hash": "y"},
    }


NOW_EVENT_FUTURE = "2026-08-21T17:30:00Z"  # far enough from "now" (real UTC at test time) to always be PRE_DEADLINE with room to spare


def test_season_ended_is_a_clean_noop(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    past = "2020-01-01T17:30:00Z"
    bs = make_bootstrap_static([make_event(1, past, finished=True, data_checked=True)])
    _install_fake_network(monkeypatch, bs, [])

    exit_code = predict.run()

    assert exit_code == predict.EXIT_OK
    assert not (tmp_path / "predictions").exists()


def test_too_close_to_deadline_refuses_and_writes_nothing(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    from datetime import datetime, timedelta, timezone
    near_deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bs = make_bootstrap_static([make_event(1, near_deadline, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=near_deadline)]
    _install_fake_network(monkeypatch, bs, fixtures)

    exit_code = predict.run()

    assert exit_code == predict.EXIT_TOO_CLOSE_TO_DEADLINE
    assert not (tmp_path / "predictions").exists()


def test_fully_blank_gameweek_writes_a_marker_record_without_invoking_the_model(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    _install_fake_network(monkeypatch, bs, [])  # zero fixtures -> every team missing -> fully blank

    def _explode(*a, **k):
        raise AssertionError("the model must never be invoked for a fully blank gameweek")

    monkeypatch.setattr(rpr, "generate_recommendation", _explode)

    exit_code = predict.run()

    assert exit_code == predict.EXIT_OK
    ledger = tmp_path / "predictions" / "gw01.jsonl"
    record = json.loads(ledger.read_text().splitlines()[0])
    assert record["status"] == "BLANK_GAMEWEEK"
    assert record["squad"] is None
    assert record["calls"] == []
    assert set(record["teams_without_fixture"]) == {"Arsenal", "Chelsea", "Everton", "Fulham"}


def test_published_record_has_the_expected_calls_and_squad_shape(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "generate_recommendation", lambda gw, write_artifact=True: _fake_model_result(gw))

    exit_code = predict.run()

    assert exit_code == predict.EXIT_OK
    ledger = tmp_path / "predictions" / "gw01.jsonl"
    record = json.loads(ledger.read_text().splitlines()[0])
    assert record["status"] == "PUBLISHED"
    assert record["supersedes"] is None
    assert record["squad"]["captain_player_id"] == "411"
    call_ids = {c["id"] for c in record["calls"]}
    assert "gw01-captain" in call_ids
    assert "gw01-squad-total" in call_ids
    assert "gw01-captain-haul" in call_ids
    captain_call = next(c for c in record["calls"] if c["id"] == "gw01-captain")
    assert captain_call["value"] == round(6.9 * 2, 3)  # doubled, not raw EP
    haul_call = next(c for c in record["calls"] if c["id"] == "gw01-captain-haul")
    assert haul_call["type"] == "binary_probability"
    assert haul_call["probability"] == 0.59
    assert haul_call["subject"]["threshold"] == 6  # a number score.py can check against, not just prose in "claim"
    # a real, verifiable content hash -- not a placeholder
    body_without_id = {k: v for k, v in record.items() if k != "record_id"}
    assert record["record_id"] == predict._content_hash(body_without_id)


def test_second_run_supersedes_the_first_and_both_lines_survive(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "generate_recommendation", lambda gw, write_artifact=True: _fake_model_result(gw))

    predict.run()
    predict.run()

    lines = (tmp_path / "predictions" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert second["supersedes"] == first["record_id"]
    assert first["record_id"] != second["record_id"]  # generated_at_utc differs -> different content hash


def test_dry_run_computes_everything_but_writes_no_ledger_file(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "generate_recommendation", lambda gw, write_artifact=True: _fake_model_result(gw))

    exit_code = predict.run(dry_run=True)

    assert exit_code == predict.EXIT_OK
    assert not (tmp_path / "predictions" / "gw01.jsonl").exists()
    # but the raw fetch+cache DID happen for real -- dry-run only skips the ledger write
    assert (tmp_path / "raw" / "gw01").exists()


def test_model_exception_propagates_and_writes_nothing(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)

    def _raise(*a, **k):
        raise RuntimeError("simulated model failure")

    monkeypatch.setattr(rpr, "generate_recommendation", _raise)

    with pytest.raises(RuntimeError, match="simulated model failure"):
        predict.run()

    assert not (tmp_path / "predictions").exists()
