"""Coverage for the two Phase-13 additions to bronze.py: retry/backoff on
transient fetch failures, and the `snapshot_root` override on
capture_snapshot()/latest_snapshot(). The explicit point of
test_default_snapshot_root_behavior_is_unchanged is to make the
"byte-for-byte unaffected" claim in bronze.py's own docstring a checked
fact, not just prose — see that module's docstring for why this matters
(the research pipeline's existing callers must never be affected by
changes made for the live pipeline).
"""
from __future__ import annotations

import json

import pytest

from apex_fpl.data import bronze


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _payload_bytes() -> bytes:
    return json.dumps({"events": [{"id": 1, "is_current": False, "is_next": True}]}).encode()


def test_default_snapshot_root_behavior_is_unchanged(tmp_path, monkeypatch):
    """No snapshot_root passed -> writes under the module's own
    SNAPSHOT_ROOT, exactly as before this parameter existed."""
    monkeypatch.setattr(bronze, "SNAPSHOT_ROOT", tmp_path)
    monkeypatch.setattr(bronze.requests, "get", lambda url, timeout, headers: _FakeResponse(_payload_bytes()))

    path = bronze.capture_snapshot("bootstrap_static")

    assert path.parent == tmp_path / "bootstrap_static"
    assert bronze.latest_snapshot("bootstrap_static") == path


def test_explicit_snapshot_root_overrides_default(tmp_path, monkeypatch):
    default_root = tmp_path / "default"
    override_root = tmp_path / "override"
    monkeypatch.setattr(bronze, "SNAPSHOT_ROOT", default_root)
    monkeypatch.setattr(bronze.requests, "get", lambda url, timeout, headers: _FakeResponse(_payload_bytes()))

    path = bronze.capture_snapshot("bootstrap_static", snapshot_root=override_root)

    assert path.parent == override_root / "bootstrap_static"
    assert not default_root.exists()  # nothing written to the default location
    assert bronze.latest_snapshot("bootstrap_static", snapshot_root=override_root) == path
    assert bronze.latest_snapshot("bootstrap_static") is None  # default root sees nothing


def test_fetch_raw_retries_transient_failures_then_succeeds(monkeypatch):
    import requests

    calls = {"n": 0}

    def flaky_get(url, timeout, headers):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("simulated transient failure")
        return _FakeResponse(_payload_bytes())

    monkeypatch.setattr(bronze.requests, "get", flaky_get)
    monkeypatch.setattr(bronze.time, "sleep", lambda s: None)  # keep the test fast

    raw, status = bronze.fetch_raw("bootstrap_static")

    assert calls["n"] == 3
    assert status == 200
    assert raw == _payload_bytes()


def test_fetch_raw_raises_the_real_exception_after_exhausting_retries(monkeypatch):
    import requests

    def always_fails(url, timeout, headers):
        raise requests.exceptions.ConnectionError("simulated persistent failure")

    monkeypatch.setattr(bronze.requests, "get", always_fails)
    monkeypatch.setattr(bronze.time, "sleep", lambda s: None)

    with pytest.raises(requests.exceptions.ConnectionError):
        bronze.fetch_raw("bootstrap_static", max_attempts=3)


def test_fetch_raw_backoff_delays_are_capped_and_exponential(monkeypatch):
    import requests

    sleeps = []
    monkeypatch.setattr(bronze.time, "sleep", lambda s: sleeps.append(s))

    def always_fails(url, timeout, headers):
        raise requests.exceptions.ConnectionError("simulated")

    monkeypatch.setattr(bronze.requests, "get", always_fails)

    with pytest.raises(requests.exceptions.ConnectionError):
        bronze.fetch_raw("bootstrap_static", max_attempts=5)

    assert sleeps == [2.0, 4.0, 8.0, 16.0]  # 4 delays between 5 attempts, none yet hitting the 30s cap
    assert all(s <= bronze.BACKOFF_CAP_SECONDS for s in sleeps)
