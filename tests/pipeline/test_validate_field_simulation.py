"""Offline coverage of validate_field_simulation.py -- no live API call.
Real network access (bronze.fetch_raw) is monkeypatched throughout."""
from __future__ import annotations

import json

from pipeline import validate_field_simulation as vfs

from api_payloads import make_bootstrap_static, make_event, make_fixture


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(vfs, "PREDICTIONS_DIR", tmp_path / "field_simulation_predictions")
    monkeypatch.setattr(vfs, "LEDGER_DIR", tmp_path / "field_simulation_validation")


def _write_prediction(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def _published_prediction(gw, predicted, naive, record_id="p1"):
    return {
        "record_id": record_id, "gameweek": gw, "status": "PUBLISHED",
        "prediction": {"predicted_field_mean_score": predicted, "naive_ownership_weighted_mean_score": naive, "n_owned_candidates": 500},
    }


SETTLED_DEADLINE = "2026-08-14T17:30:00Z"  # safely in the past -- matches tests/pipeline/test_score.py's own convention
SETTLED_BOOTSTRAP = make_bootstrap_static([make_event(1, SETTLED_DEADLINE, finished=True, data_checked=True, average_entry_score=55)])
SETTLED_FIXTURES = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=SETTLED_DEADLINE, finished=True, team_h_score=1, team_a_score=0)]

from datetime import datetime, timezone
NOW = datetime.now(timezone.utc)


def test_no_prediction_means_nothing_to_check(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    record = vfs.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert record is None


def test_blank_gameweek_prediction_is_skipped(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction(tmp_path / "field_simulation_predictions" / "gw01.jsonl", {"record_id": "p1", "gameweek": 1, "status": "BLANK_GAMEWEEK", "prediction": None})

    record = vfs.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record is None


def test_not_settled_yet_is_skipped(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, predicted=52.0, naive=48.0)
    _write_prediction(tmp_path / "field_simulation_predictions" / "gw01.jsonl", pred)
    from datetime import timedelta
    future_deadline = (NOW + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    unsettled_bootstrap = make_bootstrap_static([make_event(1, future_deadline, is_next=True)])
    unsettled_fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=future_deadline)]

    record = vfs.check_gameweek(1, unsettled_bootstrap, unsettled_fixtures, NOW)

    assert record is None


def test_settled_gameweek_computes_the_prediction_error(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, predicted=52.0, naive=48.0)
    _write_prediction(tmp_path / "field_simulation_predictions" / "gw01.jsonl", pred)

    record = vfs.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record["predicted_field_mean_score"] == 52.0
    assert record["actual_average_entry_score"] == 55.0
    assert record["absolute_error"] == -3.0
    assert record["percent_error"] == round(-3.0 / 55.0 * 100.0, 2)


def test_already_checked_gameweek_is_never_rechecked(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, predicted=52.0, naive=48.0)
    _write_prediction(tmp_path / "field_simulation_predictions" / "gw01.jsonl", pred)

    first = vfs.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert first is not None

    second = vfs.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert second is None
    assert len((tmp_path / "field_simulation_validation" / "gw01.jsonl").read_text().splitlines()) == 1
