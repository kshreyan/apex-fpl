"""Offline coverage of predict_field.py's OWN responsibilities: phase
gating, the BLANK_GAMEWEEK / PUBLISHED branches, record shape,
supersede chaining, dry-run. The EP-forecast pipeline
(run_production_recommendation.build_player_forecasts) and the field
Monte Carlo (apex_fpl.simulation.field) are mocked out -- their own
correctness is tested elsewhere; this file only tests this module's
wiring and ledger discipline."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import predict_field as pf

import run_production_recommendation as rpr  # noqa: E402 -- importable once pipeline.predict_field has run its sys.path insertion
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

    monkeypatch.setattr(pf.bronze.requests, "get", fake_get)


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(pf, "LEDGER_DIR", tmp_path / "field_simulation_predictions")
    monkeypatch.setattr(pf, "_git_sha", lambda: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


NOW_EVENT_FUTURE = "2026-08-21T17:30:00Z"


class _FakePlayerSimResult:
    def __init__(self, mean_points, samples):
        self.mean_points = mean_points
        self.samples = samples


def _fake_forecast():
    candidates_meta = {
        "1": {"name": "Player One", "team": "Arsenal", "position": "GK", "price": 5.0},
        "2": {"name": "Player Two", "team": "Chelsea", "position": "MID", "price": 8.0},
    }
    sim_results = {
        "1": _FakePlayerSimResult(4.0, np.array([4.0, 4.0])),
        "2": _FakePlayerSimResult(6.0, np.array([6.0, 6.0])),
    }
    return {"sim_results": sim_results, "candidates_meta": candidates_meta}


def test_season_ended_is_a_clean_noop(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    past = "2020-01-01T17:30:00Z"
    bs = make_bootstrap_static([make_event(1, past, finished=True, data_checked=True)])
    _install_fake_network(monkeypatch, bs, [])

    exit_code = pf.run()

    assert exit_code == pf.EXIT_OK
    assert not (tmp_path / "field_simulation_predictions").exists()


def test_too_close_to_deadline_refuses_and_writes_nothing(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    from datetime import datetime, timedelta, timezone
    near_deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bs = make_bootstrap_static([make_event(1, near_deadline, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=near_deadline)]
    _install_fake_network(monkeypatch, bs, fixtures)

    exit_code = pf.run()

    assert exit_code == pf.EXIT_TOO_CLOSE_TO_DEADLINE
    assert not (tmp_path / "field_simulation_predictions").exists()


def test_blank_target_gameweek_writes_a_status_record_without_simulating_anything(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)

    def _raise(target_gw, log=None):
        raise ValueError(f"no live fixtures found for GW{target_gw}")
    monkeypatch.setattr(rpr, "build_player_forecasts", _raise)

    def _explode(*a, **k):
        raise AssertionError("must not sample rival squads for a blank gameweek")
    monkeypatch.setattr(pf.fsim, "sample_synthetic_rival_squads", _explode)

    exit_code = pf.run()

    assert exit_code == pf.EXIT_OK
    record = json.loads((tmp_path / "field_simulation_predictions" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "BLANK_GAMEWEEK"
    assert record["prediction"] is None


def test_published_record_reflects_the_field_simulation(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "build_player_forecasts", lambda gw, log=None: _fake_forecast())
    monkeypatch.setattr(pf.ld, "load_players", lambda: {"1": {"selected_by_percent": 40.0}, "2": {"selected_by_percent": 60.0}})
    monkeypatch.setattr(pf.fsim, "sample_synthetic_rival_squads", lambda *a, **k: [["1", "2"]])
    monkeypatch.setattr(pf.fsim, "simulate_field_scores", lambda *a, **k: np.array([[10.0, 12.0]]))
    monkeypatch.setattr(pf.fsim, "naive_ownership_weighted_mean_score", lambda *a, **k: 9.5)

    exit_code = pf.run()

    assert exit_code == pf.EXIT_OK
    record = json.loads((tmp_path / "field_simulation_predictions" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "PUBLISHED"
    pred = record["prediction"]
    assert pred["predicted_field_mean_score"] == 11.0
    assert pred["naive_ownership_weighted_mean_score"] == 9.5
    assert pred["n_owned_candidates"] == 2


def test_second_run_supersedes_the_first(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "build_player_forecasts", lambda gw, log=None: _fake_forecast())
    monkeypatch.setattr(pf.ld, "load_players", lambda: {"1": {"selected_by_percent": 40.0}, "2": {"selected_by_percent": 60.0}})
    monkeypatch.setattr(pf.fsim, "sample_synthetic_rival_squads", lambda *a, **k: [["1", "2"]])
    monkeypatch.setattr(pf.fsim, "simulate_field_scores", lambda *a, **k: np.array([[10.0, 12.0]]))
    monkeypatch.setattr(pf.fsim, "naive_ownership_weighted_mean_score", lambda *a, **k: 9.5)

    pf.run()
    pf.run()

    lines = (tmp_path / "field_simulation_predictions" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert second["supersedes"] == first["record_id"]


def test_dry_run_writes_no_ledger_file(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "build_player_forecasts", lambda gw, log=None: _fake_forecast())
    monkeypatch.setattr(pf.ld, "load_players", lambda: {"1": {"selected_by_percent": 40.0}, "2": {"selected_by_percent": 60.0}})
    monkeypatch.setattr(pf.fsim, "sample_synthetic_rival_squads", lambda *a, **k: [["1", "2"]])
    monkeypatch.setattr(pf.fsim, "simulate_field_scores", lambda *a, **k: np.array([[10.0, 12.0]]))
    monkeypatch.setattr(pf.fsim, "naive_ownership_weighted_mean_score", lambda *a, **k: 9.5)

    exit_code = pf.run(dry_run=True)

    assert exit_code == pf.EXIT_OK
    assert not (tmp_path / "field_simulation_predictions" / "gw01.jsonl").exists()
