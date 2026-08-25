"""Offline coverage of predict_rank_aware.py's OWN responsibilities:
phase gating, the BLANK_GAMEWEEK / PUBLISHED branches, record shape,
supersede chaining, dry-run. The EP-forecast pipeline
(run_production_recommendation.build_player_forecasts) and the rank-
aware selector (apex_fpl.optimization.rank_aware) are mocked out --
their own correctness is tested elsewhere; this file only tests this
module's wiring and ledger discipline."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import predict_rank_aware as pra

import run_production_recommendation as rpr  # noqa: E402 -- importable once pipeline.predict_rank_aware has run its sys.path insertion
from apex_fpl.optimization import rank_aware as ra
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

    monkeypatch.setattr(pra.bronze.requests, "get", fake_get)


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(pra, "LEDGER_DIR", tmp_path / "rank_aware_predictions")
    monkeypatch.setattr(pra, "_git_sha", lambda: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")


# Computed relative to the real wall clock, not a hardcoded literal --
# a fixed calendar date eventually drifts into the past as real time
# passes it, silently turning "PRE_DEADLINE" into "season has ended" for
# every test below (this happened for real: see git history/incident).
NOW_EVENT_FUTURE = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")


class _FakePlayerSimResult:
    def __init__(self, mean_points):
        self.mean_points = mean_points


def _fake_forecast():
    candidates_meta = {
        "1": {"name": "Player One", "team": "Arsenal", "position": "GK", "price": 5.0, "availability_probability": 1.0},
        "2": {"name": "Player Two", "team": "Chelsea", "position": "MID", "price": 8.0, "availability_probability": 1.0},
    }
    sim_results = {"1": _FakePlayerSimResult(4.0), "2": _FakePlayerSimResult(6.0)}
    return {"sim_results": sim_results, "candidates_meta": candidates_meta}


def _fake_selection_result():
    ev_candidate = ra.CandidateSquadResult(
        label="max_ev", squad_ids=("1", "2"), swapped_out=None, swapped_in=None,
        mean_ev=10.0, mean_simulated_score=10.0, mean_percentile=0.6, p_top10pct=0.2, p_top25pct=0.4,
    )
    diff_candidate = ra.CandidateSquadResult(
        label="differential_0", squad_ids=("1", "3"), swapped_out="2", swapped_in="3",
        mean_ev=9.5, mean_simulated_score=9.8, mean_percentile=0.65, p_top10pct=0.3, p_top25pct=0.45,
    )
    return ra.RankAwareSelectionResult(candidates=(ev_candidate, diff_candidate), selected=diff_candidate, target_metric="p_top10pct")


def test_season_ended_is_a_clean_noop(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    past = "2020-01-01T17:30:00Z"
    bs = make_bootstrap_static([make_event(1, past, finished=True, data_checked=True)])
    _install_fake_network(monkeypatch, bs, [])

    exit_code = pra.run()

    assert exit_code == pra.EXIT_OK
    assert not (tmp_path / "rank_aware_predictions").exists()


def test_too_close_to_deadline_refuses_and_writes_nothing(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    near_deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bs = make_bootstrap_static([make_event(1, near_deadline, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=near_deadline)]
    _install_fake_network(monkeypatch, bs, fixtures)

    exit_code = pra.run()

    assert exit_code == pra.EXIT_TOO_CLOSE_TO_DEADLINE
    assert not (tmp_path / "rank_aware_predictions").exists()


def test_blank_target_gameweek_writes_a_status_record_without_selecting_anything(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)

    def _raise(target_gw, log=None):
        raise ValueError(f"no live fixtures found for GW{target_gw}")
    monkeypatch.setattr(rpr, "build_player_forecasts", _raise)

    def _explode(*a, **k):
        raise AssertionError("must not select a rank-aware squad for a blank gameweek")
    monkeypatch.setattr(pra.ra, "select_rank_aware_squad", _explode)

    exit_code = pra.run()

    assert exit_code == pra.EXIT_OK
    record = json.loads((tmp_path / "rank_aware_predictions" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "BLANK_GAMEWEEK"
    assert record["candidates"] is None
    assert record["selected"] is None


def test_published_record_reflects_the_rank_aware_selection(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "build_player_forecasts", lambda gw, log=None: _fake_forecast())
    monkeypatch.setattr(pra.ld, "load_players", lambda: {"1": {"selected_by_percent": 40.0}, "2": {"selected_by_percent": 60.0}})
    monkeypatch.setattr(pra.ra, "select_rank_aware_squad", lambda *a, **k: _fake_selection_result())

    exit_code = pra.run()

    assert exit_code == pra.EXIT_OK
    record = json.loads((tmp_path / "rank_aware_predictions" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "PUBLISHED"
    assert record["target_metric"] == "p_top10pct"
    assert len(record["candidates"]) == 2
    assert record["selected"]["label"] == "differential_0"
    assert record["selected"]["swapped_out"]["player_id"] == "2"
    assert record["selected"]["swapped_out"]["name"] == "Player Two"


def test_second_run_supersedes_the_first(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "build_player_forecasts", lambda gw, log=None: _fake_forecast())
    monkeypatch.setattr(pra.ld, "load_players", lambda: {"1": {"selected_by_percent": 40.0}, "2": {"selected_by_percent": 60.0}})
    monkeypatch.setattr(pra.ra, "select_rank_aware_squad", lambda *a, **k: _fake_selection_result())

    pra.run()
    pra.run()

    lines = (tmp_path / "rank_aware_predictions" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert second["supersedes"] == first["record_id"]


def test_dry_run_writes_no_ledger_file(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(rpr, "build_player_forecasts", lambda gw, log=None: _fake_forecast())
    monkeypatch.setattr(pra.ld, "load_players", lambda: {"1": {"selected_by_percent": 40.0}, "2": {"selected_by_percent": 60.0}})
    monkeypatch.setattr(pra.ra, "select_rank_aware_squad", lambda *a, **k: _fake_selection_result())

    exit_code = pra.run(dry_run=True)

    assert exit_code == pra.EXIT_OK
    assert not (tmp_path / "rank_aware_predictions" / "gw01.jsonl").exists()
