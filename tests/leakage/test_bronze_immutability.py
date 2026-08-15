"""Leakage/integrity tests for the Bronze snapshot layer.

These are not unit tests of business logic — they exist to make a specific
falsifiable guarantee auditable: once written, a Bronze snapshot is never
silently overwritten, and its metadata's hash always matches its payload
bytes. If either guarantee breaks, point-in-time correctness for every
downstream Gold-layer feature is compromised.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from apex_fpl.data import bronze


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_capture_never_overwrites_existing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(bronze, "SNAPSHOT_ROOT", tmp_path)

    calls = {"n": 0}

    def fake_get(url, timeout, headers):
        calls["n"] += 1
        payload = {"events": [{"id": 1, "is_current": False, "is_next": True}], "call": calls["n"]}
        return _FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(bronze.requests, "get", fake_get)

    p1 = bronze.capture_snapshot("bootstrap_static")
    p2 = bronze.capture_snapshot("bootstrap_static")

    assert p1 != p2, "second capture must not silently reuse/overwrite the first snapshot's path"
    assert p1.exists() and p2.exists()
    assert p1.read_bytes() != p2.read_bytes(), "fake payloads differ by call count; both must be preserved distinctly"


def test_metadata_hash_matches_payload_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(bronze, "SNAPSHOT_ROOT", tmp_path)

    def fake_get(url, timeout, headers):
        payload = {"events": [{"id": 1, "is_current": True, "is_next": False}]}
        return _FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(bronze.requests, "get", fake_get)

    path = bronze.capture_snapshot("bootstrap_static", season="2026/27")
    meta = json.loads(path.with_suffix(".meta.json").read_text())

    assert meta["raw_payload_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["current_event_id"] == 1
    assert meta["next_event_id"] is None
    assert meta["season"] == "2026/27"
    assert meta["schema_version"] == bronze.SCHEMA_VERSION


def test_unknown_source_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bronze, "SNAPSHOT_ROOT", tmp_path)
    with pytest.raises(ValueError):
        bronze.capture_snapshot("not_a_real_source")
