"""Unit tests for apex_fpl.serving.entry_state -- the real FPL entry's
squad-state ledger (Phase 13 Block 2.5). All network access is
monkeypatched; no real HTTP calls."""
from __future__ import annotations

import json

import pytest

from apex_fpl.serving import entry_state as es


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self.content = json.dumps(payload).encode() if payload is not None else b""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code}")


def _valid_picks_payload(elements=(1, 2, 3), bank=5, value=1000, event_transfers=0, event_transfers_cost=0):
    return {
        "picks": [{"element": e, "position": i + 1, "multiplier": 1, "is_captain": i == 0, "is_vice_captain": i == 1} for i, e in enumerate(elements)],
        "entry_history": {"bank": bank, "value": value, "event_transfers": event_transfers, "event_transfers_cost": event_transfers_cost},
    }


def test_get_with_retry_returns_none_on_404(monkeypatch):
    monkeypatch.setattr(es.requests, "get", lambda url, timeout, headers: _FakeResponse(404))
    assert es._get_with_retry("https://example.invalid") is None


def test_get_with_retry_returns_response_on_200(monkeypatch):
    monkeypatch.setattr(es.requests, "get", lambda url, timeout, headers: _FakeResponse(200, {"ok": True}))
    resp = es._get_with_retry("https://example.invalid")
    assert resp is not None
    assert json.loads(resp.content) == {"ok": True}


def test_validate_entry_picks_accepts_well_formed_payload():
    es.validate_entry_picks(_valid_picks_payload())  # must not raise


def test_validate_entry_picks_raises_on_missing_picks_field():
    bad = _valid_picks_payload()
    del bad["picks"][0]["element"]
    with pytest.raises(es.EntryStateError, match=r"picks\[0\]\.element"):
        es.validate_entry_picks(bad)


def test_validate_entry_picks_raises_on_missing_entry_history_field():
    bad = _valid_picks_payload()
    del bad["entry_history"]["bank"]
    with pytest.raises(es.EntryStateError, match="entry_history.bank"):
        es.validate_entry_picks(bad)


def test_fetch_entry_picks_returns_none_when_not_yet_available(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "RAW_DATA_ROOT", tmp_path)
    monkeypatch.setattr(es.requests, "get", lambda url, timeout, headers: _FakeResponse(404))
    assert es.fetch_entry_picks(1) is None
    assert not (tmp_path / "gw01").exists(), "must not write a raw capture when nothing was actually fetched"


def test_fetch_entry_picks_writes_raw_capture_and_returns_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "RAW_DATA_ROOT", tmp_path)
    payload = _valid_picks_payload()
    monkeypatch.setattr(es.requests, "get", lambda url, timeout, headers: _FakeResponse(200, payload))

    result = es.fetch_entry_picks(1)

    assert result == payload
    written = list((tmp_path / "gw01" / "entry_picks").glob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_bytes()) == payload


def test_fetch_entry_picks_raises_on_malformed_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "RAW_DATA_ROOT", tmp_path)
    bad = {"picks": []}  # missing entry_history entirely
    monkeypatch.setattr(es.requests, "get", lambda url, timeout, headers: _FakeResponse(200, bad))
    with pytest.raises(es.EntryStateError):
        es.fetch_entry_picks(1)


def test_compute_free_transfers_raises_before_any_gameweek_settled():
    with pytest.raises(es.EntryStateError, match="before any gameweek has settled"):
        es.compute_free_transfers([], settled_gw=0)


def test_compute_free_transfers_banks_one_when_no_transfers_made():
    """GW2 settled with 0 transfers made: the free transfer earned for
    GW2 carries forward, plus the +1 for GW3 -- 2 banked."""
    history = [{"event": 2, "event_transfers": 0, "event_transfers_cost": 0}]
    assert es.compute_free_transfers(history, settled_gw=2) == 2


def test_compute_free_transfers_spends_the_free_transfer_with_no_hit():
    """GW2: exactly 1 transfer made (the free one) -- no hit, banked
    stays at the floor of 1 entering GW3."""
    history = [{"event": 2, "event_transfers": 1, "event_transfers_cost": 0}]
    assert es.compute_free_transfers(history, settled_gw=2) == 1


def test_compute_free_transfers_takes_a_hit_beyond_the_free_allowance():
    """GW2: 2 transfers made with only 1 free -- 1 paid transfer (a real
    -4 hit), free transfers still floors at 1 entering GW3, not negative."""
    history = [{"event": 2, "event_transfers": 2, "event_transfers_cost": -4}]
    assert es.compute_free_transfers(history, settled_gw=2) == 1


def test_compute_free_transfers_caps_at_the_maximum_banked():
    """No transfers made for 6 straight gameweeks (GW2-7): banking must
    cap at MAX_BANKED_FREE_TRANSFERS (5), never grow unbounded."""
    history = [{"event": gw, "event_transfers": 0, "event_transfers_cost": 0} for gw in range(2, 8)]
    assert es.compute_free_transfers(history, settled_gw=7) == es.MAX_BANKED_FREE_TRANSFERS


def test_compute_free_transfers_raises_on_a_gap_in_history():
    """GW3's row is missing entirely (a real API/capture inconsistency,
    not silently skippable) -- must fail loudly, not guess."""
    history = [{"event": 2, "event_transfers": 0, "event_transfers_cost": 0}]
    with pytest.raises(es.EntryStateError, match="no row for gameweek 3"):
        es.compute_free_transfers(history, settled_gw=3)


def test_build_current_squad_state_returns_none_when_no_gameweek_settled(monkeypatch):
    monkeypatch.setattr(es, "fetch_entry_history", lambda entry_id=es.ENTRY_ID: {"current": [], "past": [], "chips": []})
    assert es.build_current_squad_state({}) is None


def test_build_current_squad_state_builds_from_real_picks_and_history(monkeypatch, tmp_path):
    monkeypatch.setattr(es, "RAW_DATA_ROOT", tmp_path)
    history_payload = {"current": [{"event": 1, "event_transfers": 0, "event_transfers_cost": 0, "bank": 5, "value": 1000}], "past": [], "chips": []}
    picks_payload = _valid_picks_payload(elements=(101, 102, 103), bank=7, event_transfers=0)
    monkeypatch.setattr(es, "fetch_entry_history", lambda entry_id=es.ENTRY_ID: history_payload)
    monkeypatch.setattr(es.requests, "get", lambda url, timeout, headers: _FakeResponse(200, picks_payload))

    now_cost = {101: 55, 102: 60, 103: 45}
    state = es.build_current_squad_state(now_cost)

    assert state.squad_ids == ["101", "102", "103"]
    assert state.bank == 0.7
    assert state.as_of_gw == 1
    assert state.free_transfers == es.STARTING_FREE_TRANSFERS  # no gameweek 2 has happened to earn a second one
    assert state.sell_price_by_id == {"101": 5.5, "102": 6.0, "103": 4.5}


def _full_squad_picks_payload(captain_element=1):
    """15 real-shaped picks: positions 1-11 starting XI, 12-15 bench,
    in bench order."""
    picks = [
        {"element": i, "position": i, "multiplier": (2 if i == captain_element else 1) if i <= 11 else 0, "is_captain": i == captain_element, "is_vice_captain": i == captain_element + 1}
        for i in range(1, 16)
    ]
    return {"picks": picks, "entry_history": {"bank": 0, "value": 1000, "event_transfers": 0, "event_transfers_cost": 0}}


def test_parse_gameweek_lineup_extracts_squad_bench_and_captain():
    payload = _full_squad_picks_payload(captain_element=3)
    lineup = es.parse_gameweek_lineup(payload, gw=5)
    assert lineup.gw == 5
    assert len(lineup.squad_ids) == 15
    assert lineup.bench_ids == ["12", "13", "14", "15"]
    assert lineup.captain_id == "3"


def test_parse_gameweek_lineup_bench_order_follows_position_field():
    payload = _full_squad_picks_payload()
    # scramble the list order but keep position values meaningful
    import random
    shuffled = dict(payload)
    shuffled["picks"] = random.Random(3).sample(payload["picks"], len(payload["picks"]))
    lineup = es.parse_gameweek_lineup(shuffled, gw=1)
    assert lineup.bench_ids == ["12", "13", "14", "15"]


def test_already_played_chips_returns_empty_list_when_absent(monkeypatch):
    monkeypatch.setattr(es, "fetch_entry_history", lambda entry_id=es.ENTRY_ID: {"current": [], "past": []})
    assert es.already_played_chips() == []


def test_already_played_chips_returns_real_chips_field(monkeypatch):
    chips = [{"name": "wildcard", "event": 5, "time": "2026-09-01T00:00:00Z"}]
    monkeypatch.setattr(es, "fetch_entry_history", lambda entry_id=es.ENTRY_ID: {"current": [], "past": [], "chips": chips})
    assert es.already_played_chips() == chips


def test_build_current_squad_state_raises_on_inconsistent_api_state(monkeypatch):
    """entry_history claims gameweek 1 settled, but the picks endpoint
    still 404s -- a real inconsistency worth failing loudly over, not
    silently treating as "not settled yet" (which would be wrong: the
    entry_history side already disagrees with that)."""
    history_payload = {"current": [{"event": 1, "event_transfers": 0, "event_transfers_cost": 0, "bank": 5, "value": 1000}], "past": [], "chips": []}
    monkeypatch.setattr(es, "fetch_entry_history", lambda entry_id=es.ENTRY_ID: history_payload)
    monkeypatch.setattr(es, "fetch_entry_picks", lambda gw, entry_id=es.ENTRY_ID: None)

    with pytest.raises(es.EntryStateError, match="inconsistent API state"):
        es.build_current_squad_state({})
