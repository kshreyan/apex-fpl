from __future__ import annotations

import json

import pytest

from pipeline import fpl_client as fc

from api_payloads import make_bootstrap_static, make_event, make_fixture


def _valid_bootstrap_static():
    return make_bootstrap_static([make_event(1, "2026-08-21T17:30:00Z", is_next=True)])


def _valid_fixtures():
    return [
        make_fixture(1, 1, team_h=1, team_a=2, kickoff_time="2026-08-21T14:00:00Z"),  # unplayed: null scores
        make_fixture(2, 1, team_h=3, team_a=4, kickoff_time="2026-08-14T14:00:00Z", finished=True, team_h_score=2, team_a_score=1),
    ]


def test_validate_bootstrap_static_accepts_well_formed_payload():
    fc.validate_bootstrap_static(_valid_bootstrap_static())  # must not raise


def test_validate_fixtures_accepts_null_scores_for_unplayed_fixtures():
    fc.validate_fixtures(_valid_fixtures())  # must not raise -- null is legitimate, not a schema violation


def test_validate_bootstrap_static_accepts_bare_int_zero_for_defensive_contribution_per_90():
    """Real live-API behavior (broke an actual pipeline run on 2026-08-18,
    caught within hours by the healthcheck/monitoring loop): the API
    serializes an exact-zero rate as the bare JSON int 0, not 0.0, while
    any nonzero value comes through as a float. A player with any minutes
    played but zero defensive actions -- the common case pre-season and
    for most attackers -- must not fail validation."""
    payload = _valid_bootstrap_static()
    payload["elements"][0]["defensive_contribution_per_90"] = 0
    fc.validate_bootstrap_static(payload)  # must not raise


def test_validate_bootstrap_static_raises_specific_error_on_missing_field():
    bad = _valid_bootstrap_static()
    del bad["events"][0]["data_checked"]
    with pytest.raises(fc.SchemaValidationError, match=r"events\[0\]\.data_checked: missing"):
        fc.validate_bootstrap_static(bad)


def test_validate_bootstrap_static_raises_specific_error_on_wrong_type():
    """A realistic drift scenario: selected_by_percent changes from a
    string to a number. Must name the exact field, not raise a bare
    TypeError/KeyError."""
    bad = _valid_bootstrap_static()
    bad["elements"][0]["selected_by_percent"] = 31.2  # float instead of the real API's string
    with pytest.raises(fc.SchemaValidationError, match=r"elements\[0\]\.selected_by_percent: expected str, got float"):
        fc.validate_bootstrap_static(bad)


def test_validate_bootstrap_static_raises_on_missing_top_level_list():
    bad = _valid_bootstrap_static()
    del bad["teams"]
    with pytest.raises(fc.SchemaValidationError, match=r"bootstrap_static\.teams: missing"):
        fc.validate_bootstrap_static(bad)


def test_validate_fixtures_raises_specific_error_on_missing_field():
    bad = _valid_fixtures()
    del bad[1]["team_h_score"]
    with pytest.raises(fc.SchemaValidationError, match=r"fixtures\[1\]\.team_h_score: missing"):
        fc.validate_fixtures(bad)


def test_validate_fixtures_raises_on_non_list_payload():
    with pytest.raises(fc.SchemaValidationError, match="expected a list at top level"):
        fc.validate_fixtures({"not": "a list"})


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass


def test_fetch_and_validate_writes_under_gameweek_directory_and_returns_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "RAW_DATA_ROOT", tmp_path)
    payload = _valid_bootstrap_static()
    monkeypatch.setattr(fc.bronze.requests, "get", lambda url, timeout, headers: _FakeResponse(json.dumps(payload).encode()))

    path = fc.fetch_and_validate("bootstrap_static", gameweek=1)

    assert path.parent == tmp_path / "gw01" / "bootstrap_static"
    assert json.loads(path.read_bytes()) == payload


def test_fetch_and_validate_raises_but_still_leaves_the_raw_file_on_disk(tmp_path, monkeypatch):
    """The core integrity property: a schema failure never destroys the
    evidence that was already captured."""
    monkeypatch.setattr(fc, "RAW_DATA_ROOT", tmp_path)
    bad_payload = _valid_bootstrap_static()
    del bad_payload["events"][0]["finished"]
    monkeypatch.setattr(fc.bronze.requests, "get", lambda url, timeout, headers: _FakeResponse(json.dumps(bad_payload).encode()))

    with pytest.raises(fc.SchemaValidationError):
        fc.fetch_and_validate("bootstrap_static", gameweek=1)

    written = list((tmp_path / "gw01" / "bootstrap_static").glob("*.json"))
    written = [p for p in written if not p.name.endswith(".meta.json")]
    assert len(written) == 1
    assert json.loads(written[0].read_bytes()) == bad_payload


def test_fetch_and_validate_rejects_unknown_source():
    with pytest.raises(ValueError, match="no validator registered"):
        fc.fetch_and_validate("element_summary", gameweek=1)
