"""Offline coverage of pipeline/should_run_now.py's gate logic -- reuses
gw_state.py's already-tested phase computation, so this only needs to
test the CLOSING_WINDOW_HOURS threshold behavior itself, not phase
logic from scratch."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pipeline import should_run_now as srn

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

    monkeypatch.setattr(srn.bronze.requests, "get", fake_get)


def test_false_when_deadline_is_far_away(monkeypatch):
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=48)  # comfortably outside the 24h closing window
    bs = make_bootstrap_static([make_event(1, deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), is_next=True)])
    _install_fake_network(monkeypatch, bs, [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=deadline.strftime("%Y-%m-%dT%H:%M:%SZ"))])

    assert srn.should_run_now() is False


def test_true_when_within_the_closing_window(monkeypatch):
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(hours=12)  # inside the 24h closing window
    bs = make_bootstrap_static([make_event(1, deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), is_next=True)])
    _install_fake_network(monkeypatch, bs, [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=deadline.strftime("%Y-%m-%dT%H:%M:%SZ"))])

    assert srn.should_run_now() is True


def test_false_when_season_has_ended(monkeypatch):
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=10)
    bs = make_bootstrap_static([make_event(1, past.strftime("%Y-%m-%dT%H:%M:%SZ"), finished=True, data_checked=True)])
    _install_fake_network(monkeypatch, bs, [])

    assert srn.should_run_now() is False


def test_false_once_settled_even_if_it_was_recently_the_deadline(monkeypatch):
    """A gameweek whose deadline just passed is IN_PROGRESS, not
    PRE_DEADLINE -- the hourly gate exists to catch pre-deadline team
    news, not to re-trigger once it's too late to act anyway."""
    now = datetime.now(timezone.utc)
    recent_past = now - timedelta(hours=2)
    bs = make_bootstrap_static([
        make_event(1, recent_past.strftime("%Y-%m-%dT%H:%M:%SZ"), finished=False, data_checked=False),
        make_event(2, (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"), is_next=True),
    ])
    _install_fake_network(monkeypatch, bs, [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time=recent_past.strftime("%Y-%m-%dT%H:%M:%SZ"))])

    assert srn.should_run_now() is False
