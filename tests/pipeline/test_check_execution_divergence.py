"""Offline coverage of check_execution_divergence.py -- no live API
call, no real FPL entry. Real network access (bronze.fetch_raw and
entry_state's own HTTP calls) is monkeypatched throughout."""
from __future__ import annotations

import json

import pytest

from pipeline import check_execution_divergence as ced

from api_payloads import make_bootstrap_static, make_event, make_fixture


def _install_tmp_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(ced, "PREDICTIONS_DIR", tmp_path / "predictions")
    monkeypatch.setattr(ced, "TRANSFER_RECOMMENDATIONS_DIR", tmp_path / "transfer_recommendations")
    monkeypatch.setattr(ced, "LEDGER_DIR", tmp_path / "execution_divergence")


def _write_prediction(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def _published_prediction(gw, starters, bench, captain_id, record_id="p1"):
    return {
        "record_id": record_id, "gameweek": gw, "status": "PUBLISHED",
        "squad": {
            "starting_xi": [{"player_id": pid} for pid in starters],
            "bench_order": [{"player_id": pid} for pid in bench],
            "captain_player_id": captain_id,
        },
    }


SETTLED_DEADLINE = "2026-08-14T17:30:00Z"  # safely in the past -- matches tests/pipeline/test_score.py's own convention
SETTLED_BOOTSTRAP = make_bootstrap_static([make_event(1, SETTLED_DEADLINE, finished=True, data_checked=True)])
SETTLED_FIXTURES = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=SETTLED_DEADLINE, finished=True, team_h_score=1, team_a_score=0)]

from datetime import datetime, timezone
NOW = datetime.now(timezone.utc)


def test_no_prediction_means_nothing_to_check(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    record = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert record is None


def test_blank_gameweek_prediction_is_skipped(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", {"record_id": "p1", "gameweek": 1, "status": "BLANK_GAMEWEEK", "squad": None})

    record = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record is None


def test_not_settled_yet_is_skipped(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    from datetime import timedelta
    future_deadline = (NOW + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    unsettled_bootstrap = make_bootstrap_static([make_event(1, future_deadline, is_next=True)])
    unsettled_fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=future_deadline)]

    record = ced.check_gameweek(1, unsettled_bootstrap, unsettled_fixtures, NOW)

    assert record is None


def test_real_picks_not_yet_visible_is_skipped_not_a_false_match(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: None)

    record = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record is None


def _picks_payload(elements, captain_element):
    return {"picks": [{"element": e, "position": i + 1, "multiplier": 2 if e == captain_element else 1, "is_captain": e == captain_element, "is_vice_captain": False} for i, e in enumerate(elements)]}


def test_matched_squad_and_captain(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: _picks_payload([1, 2, 3], captain_element=1))

    record = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record["status"] == "MATCHED"
    assert record["squad_diverged"] is False
    assert record["captain_diverged"] is False


def test_diverged_squad(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: _picks_payload([1, 2, 99], captain_element=1))

    record = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record["status"] == "DIVERGED"
    assert record["squad_diverged"] is True
    assert record["captain_diverged"] is False


def test_diverged_captain(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: _picks_payload([1, 2, 3], captain_element=2))

    record = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)

    assert record["status"] == "DIVERGED"
    assert record["squad_diverged"] is False
    assert record["captain_diverged"] is True


def test_already_checked_gameweek_is_never_rechecked(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: _picks_payload([1, 2, 3], captain_element=1))

    first = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert first is not None

    def _explode(*a, **k):
        raise AssertionError("must not re-fetch real picks for an already-checked gameweek")
    monkeypatch.setattr(ced.es, "fetch_entry_picks", _explode)

    second = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert second is None
    assert len((tmp_path / "execution_divergence" / "gw01.jsonl").read_text().splitlines()) == 1


# --- GW >= 2: schema 1.1's corrected comparison basis (prior real squad
# + this gameweek's PUBLISHED transfer recommendation applied) -------------

SETTLED_BOOTSTRAP_GW2 = make_bootstrap_static([
    make_event(1, SETTLED_DEADLINE, finished=True, data_checked=True),
    make_event(2, "2026-08-21T17:30:00Z", finished=True, data_checked=True),
])
SETTLED_FIXTURES_GW2 = SETTLED_FIXTURES + [make_fixture(2, 2, team_h=1, team_a=2, kickoff_time="2026-08-21T17:30:00Z", finished=True, team_h_score=1, team_a_score=0)]


def _write_transfer_recommendation(path, gw, status="PUBLISHED", transfers_in=(), transfers_out=(), record_id="t1"):
    record = {
        "record_id": record_id, "gameweek": gw, "status": status,
        "recommendation": {
            "transfers_in": [{"player_id": pid} for pid in transfers_in],
            "transfers_out": [{"player_id": pid} for pid in transfers_out],
        } if status == "PUBLISHED" else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def _install_picks_by_gw(monkeypatch, picks_by_gw: dict[int, dict | None]):
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: picks_by_gw.get(gw))


def test_gw2_matched_when_real_squad_follows_the_recommended_transfer(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(2, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw02.jsonl", pred)
    _write_transfer_recommendation(tmp_path / "transfer_recommendations" / "gw02.jsonl", 2, transfers_in=["99"], transfers_out=["3"])
    _install_picks_by_gw(monkeypatch, {
        1: _picks_payload([1, 2, 3], captain_element=1),  # prior real squad: {1,2,3}
        2: _picks_payload([1, 2, 99], captain_element=1),  # 3 -> 99, exactly as recommended
    })

    record = ced.check_gameweek(2, SETTLED_BOOTSTRAP_GW2, SETTLED_FIXTURES_GW2, NOW)

    assert record["status"] == "MATCHED"
    assert record["comparison_basis"] == "prior_squad_plus_recommended_transfer"
    assert record["squad_diverged"] is False
    assert sorted(record["expected_squad_ids"]) == ["1", "2", "99"]


def test_gw2_diverged_when_real_squad_ignores_the_recommendation(monkeypatch, tmp_path):
    """The real, live scenario this correction was built from: the
    recommendation said no transfer, but the real entry made one anyway."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(2, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw02.jsonl", pred)
    _write_transfer_recommendation(tmp_path / "transfer_recommendations" / "gw02.jsonl", 2, transfers_in=[], transfers_out=[])
    _install_picks_by_gw(monkeypatch, {
        1: _picks_payload([1, 2, 3], captain_element=1),
        2: _picks_payload([1, 2, 99], captain_element=1),  # a transfer the model never recommended
    })

    record = ced.check_gameweek(2, SETTLED_BOOTSTRAP_GW2, SETTLED_FIXTURES_GW2, NOW)

    assert record["status"] == "DIVERGED"
    assert record["squad_diverged"] is True
    assert sorted(record["expected_squad_ids"]) == ["1", "2", "3"]  # unchanged, per the "no transfer" recommendation


def test_gw2_skipped_when_no_published_transfer_recommendation_exists(monkeypatch, tmp_path):
    """Must not fall back to guessing (e.g. treating the from-scratch
    prediction as the reference) once a prior real squad exists."""
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(2, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw02.jsonl", pred)
    _install_picks_by_gw(monkeypatch, {
        1: _picks_payload([1, 2, 3], captain_element=1),
        2: _picks_payload([1, 2, 3], captain_element=1),
    })

    record = ced.check_gameweek(2, SETTLED_BOOTSTRAP_GW2, SETTLED_FIXTURES_GW2, NOW)

    assert record is None


def test_gw2_skipped_when_transfer_recommendation_not_published(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(2, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw02.jsonl", pred)
    _write_transfer_recommendation(tmp_path / "transfer_recommendations" / "gw02.jsonl", 2, status="NO_SETTLED_GAMEWEEK_YET")
    _install_picks_by_gw(monkeypatch, {
        1: _picks_payload([1, 2, 3], captain_element=1),
        2: _picks_payload([1, 2, 3], captain_element=1),
    })

    record = ced.check_gameweek(2, SETTLED_BOOTSTRAP_GW2, SETTLED_FIXTURES_GW2, NOW)

    assert record is None


def test_gw2_skipped_when_prior_real_picks_not_visible(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(2, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw02.jsonl", pred)
    _write_transfer_recommendation(tmp_path / "transfer_recommendations" / "gw02.jsonl", 2, transfers_in=["99"], transfers_out=["3"])
    _install_picks_by_gw(monkeypatch, {
        1: None,  # prior gameweek's picks not visible for some reason
        2: _picks_payload([1, 2, 99], captain_element=1),
    })

    record = ced.check_gameweek(2, SETTLED_BOOTSTRAP_GW2, SETTLED_FIXTURES_GW2, NOW)

    assert record is None


# --- --correct: the escape hatch for a bug found in this module's own
# comparison logic, mirroring score.py's own precedent --------------------

def test_correct_appends_a_new_superseding_record(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)
    monkeypatch.setattr(ced.es, "fetch_entry_picks", lambda gw, entry_id=ced.es.ENTRY_ID: _picks_payload([1, 2, 99], captain_element=1))

    first = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert first["status"] == "DIVERGED"

    # a normal run (no --correct) must not re-check
    second = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW)
    assert second is None

    corrected = ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW, correct_reason="schema_migration")
    assert corrected is not None
    assert corrected["supersedes"] == first["record_id"]
    assert corrected["supersede_reason"] == "schema_migration"
    lines = (tmp_path / "execution_divergence" / "gw01.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_correct_without_a_prior_check_raises(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    pred = _published_prediction(1, starters=["1", "2"], bench=["3"], captain_id="1")
    _write_prediction(tmp_path / "predictions" / "gw01.jsonl", pred)

    with pytest.raises(ValueError, match="never been checked"):
        ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW, correct_reason="schema_migration")


def test_invalid_correct_reason_raises(monkeypatch, tmp_path):
    _install_tmp_dirs(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="invalid correct_reason"):
        ced.check_gameweek(1, SETTLED_BOOTSTRAP, SETTLED_FIXTURES, NOW, correct_reason="not_a_real_reason")
