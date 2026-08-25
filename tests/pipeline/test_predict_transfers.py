"""Offline coverage of predict_transfers.py's OWN responsibilities:
phase gating, the NO_SETTLED_GAMEWEEK_YET / SKIPPED_INCONSISTENT_STATE /
BLANK_GAMEWEEK / PUBLISHED branches, record shape, supersede chaining,
dry-run. The real squad optimizer (apex_fpl.optimization.transfers.
plan_transfers) and the multi-gameweek EP-forecast pipeline
(run_production_recommendation.build_multi_gameweek_forecasts) are
mocked out -- their own correctness is tested elsewhere; this file only
tests this module's wiring and ledger discipline.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import predict_transfers as pt

import run_production_recommendation as rpr  # noqa: E402 -- importable once pipeline.predict_transfers has run its sys.path insertion
from apex_fpl.optimization import transfers as tr
from apex_fpl.serving import entry_state as es
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

    monkeypatch.setattr(pt.bronze.requests, "get", fake_get)


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "LEDGER_DIR", tmp_path / "transfer_recommendations")
    monkeypatch.setattr(pt, "_git_sha", lambda: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    monkeypatch.setattr(pt.ld, "load_players", lambda: {"1": {"price": 5.0}, "2": {"price": 8.0}})


# Computed relative to the real wall clock, not a hardcoded literal --
# a fixed calendar date eventually drifts into the past as real time
# passes it, silently turning "PRE_DEADLINE" into "season has ended" for
# every test below (this happened for real: see git history/incident).
NOW_EVENT_FUTURE = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")


class _FakePlayerSimResult:
    def __init__(self, mean_points):
        self.mean_points = mean_points


def _fake_multi_gw_forecasts(target_gw, horizon):
    """A minimal HORIZON-keyed forecast dict, matching build_multi_gameweek_
    forecasts' real shape -- plan_transfers itself is mocked in tests that
    need it, so the exact EP values here are only used to prove they were
    threaded through correctly (see test_published_record_reflects_the_solved_plan)."""
    candidates_meta = {
        "1": {"name": "Player One", "team": "Arsenal", "position": "DEF", "price": 5.0, "availability_probability": 1.0},
        "2": {"name": "Player Two", "team": "Chelsea", "position": "MID", "price": 8.0, "availability_probability": 1.0},
    }
    return {
        gw: {"candidates_meta": candidates_meta, "sim_results": {"1": _FakePlayerSimResult(4.0 + gw), "2": _FakePlayerSimResult(5.0 + gw)}}
        for gw in range(target_gw, target_gw + horizon)
    }


def test_season_ended_is_a_clean_noop(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    past = "2020-01-01T17:30:00Z"
    bs = make_bootstrap_static([make_event(1, past, finished=True, data_checked=True)])
    _install_fake_network(monkeypatch, bs, [])

    exit_code = pt.run()

    assert exit_code == pt.EXIT_OK
    assert not (tmp_path / "transfer_recommendations").exists()


def test_too_close_to_deadline_refuses_and_writes_nothing(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    near_deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    bs = make_bootstrap_static([make_event(1, near_deadline, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=near_deadline)]
    _install_fake_network(monkeypatch, bs, fixtures)

    exit_code = pt.run()

    assert exit_code == pt.EXIT_TOO_CLOSE_TO_DEADLINE
    assert not (tmp_path / "transfer_recommendations").exists()


def test_no_settled_gameweek_yet_writes_a_status_record_without_solving_anything(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(pt.es, "build_current_squad_state", lambda now_cost_by_element: None)

    def _explode(*a, **k):
        raise AssertionError("plan_transfers must never be called with no squad state")
    monkeypatch.setattr(pt.tr, "plan_transfers", _explode)

    exit_code = pt.run()

    assert exit_code == pt.EXIT_OK
    record = json.loads((tmp_path / "transfer_recommendations" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "NO_SETTLED_GAMEWEEK_YET"
    assert record["recommendation"] is None


def test_inconsistent_settled_gameweek_is_skipped_not_guessed(monkeypatch, tmp_path):
    """entry_history claims GW3 settled, but the pipeline's own target
    gameweek (derived independently from bootstrap-static) is GW1 --
    a real inconsistency that must be skipped, not silently trusted."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    state = es.CurrentSquadState(squad_ids=["1"], bank=0.0, free_transfers=1, sell_price_by_id={}, as_of_gw=3)
    monkeypatch.setattr(pt.es, "build_current_squad_state", lambda now_cost_by_element: state)

    exit_code = pt.run()

    assert exit_code == pt.EXIT_OK
    record = json.loads((tmp_path / "transfer_recommendations" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "SKIPPED_INCONSISTENT_STATE"
    assert "3" in record["detail"] and "1" in record["detail"]


def test_blank_target_gameweek_writes_a_status_record_without_solving_anything(monkeypatch, tmp_path):
    """target_gw itself has no fixtures at all -- build_multi_gameweek_
    forecasts maps that to None for that gw (see run_production_
    recommendation.simulate_gameweek_ep's own contract), and this module
    must record that plainly rather than crash or misreport a plan."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    state = es.CurrentSquadState(squad_ids=["1"], bank=0.5, free_transfers=1, sell_price_by_id={"1": 5.0}, as_of_gw=0)
    monkeypatch.setattr(pt.es, "build_current_squad_state", lambda now_cost_by_element: state)
    monkeypatch.setattr(rpr, "build_multi_gameweek_forecasts", lambda gw, horizon, log=None: {g: None for g in range(gw, gw + horizon)})

    def _explode(*a, **k):
        raise AssertionError("plan_transfers must never be called for a blank target gameweek")
    monkeypatch.setattr(pt.tr, "plan_transfers", _explode)

    exit_code = pt.run()

    assert exit_code == pt.EXIT_OK
    record = json.loads((tmp_path / "transfer_recommendations" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "BLANK_GAMEWEEK"
    assert record["recommendation"] is None


def test_published_record_reflects_the_solved_plan(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    state = es.CurrentSquadState(squad_ids=["1"], bank=0.5, free_transfers=1, sell_price_by_id={"1": 5.0}, as_of_gw=0)
    monkeypatch.setattr(pt.es, "build_current_squad_state", lambda now_cost_by_element: state)
    monkeypatch.setattr(rpr, "build_multi_gameweek_forecasts", lambda gw, horizon, log=None: _fake_multi_gw_forecasts(gw, horizon))

    fake_step = tr.GameweekPlan(
        gw_index=0, squad=["2"], starters=["2"], captain="2", bench_order=[],
        transfers_in=["2"], transfers_out=["1"], free_transfers_available=1,
        paid_transfers=0, hit_points=0.0, bank_after=-2.5,
    )
    fake_plan = tr.TransferPlan(horizon=1, gameweeks=[fake_step], total_net_expected_points=5.0, proven_optimal=True, mip_gap=0.0, message="ok")
    monkeypatch.setattr(pt.tr, "plan_transfers", lambda *a, **k: fake_plan)

    exit_code = pt.run()

    assert exit_code == pt.EXIT_OK
    record = json.loads((tmp_path / "transfer_recommendations" / "gw01.jsonl").read_text().splitlines()[0])
    assert record["status"] == "PUBLISHED"
    rec = record["recommendation"]
    assert rec["transfers_in"] == [{"player_id": "2", "name": "Player Two", "position": "MID", "team": "Chelsea"}]
    assert rec["transfers_out"] == [{"player_id": "1", "name": "Player One", "position": "DEF", "team": "Arsenal"}]
    assert rec["hit_points"] == 0.0
    assert rec["bank_after"] == -2.5
    assert any("horizon=4" in c for c in record["caveats"])


def test_second_run_supersedes_the_first(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(pt.es, "build_current_squad_state", lambda now_cost_by_element: None)

    pt.run()
    pt.run()

    lines = (tmp_path / "transfer_recommendations" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first, second = json.loads(lines[0]), json.loads(lines[1])
    assert second["supersedes"] == first["record_id"]


def test_dry_run_writes_no_ledger_file(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    bs = make_bootstrap_static([make_event(1, NOW_EVENT_FUTURE, is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=NOW_EVENT_FUTURE)]
    _install_fake_network(monkeypatch, bs, fixtures)
    monkeypatch.setattr(pt.es, "build_current_squad_state", lambda now_cost_by_element: None)

    exit_code = pt.run(dry_run=True)

    assert exit_code == pt.EXIT_OK
    assert not (tmp_path / "transfer_recommendations" / "gw01.jsonl").exists()
