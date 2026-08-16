from __future__ import annotations

import json
import subprocess

import pytest

from pipeline.site import git_commits


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    monkeypatch.setattr(git_commits, "REPO_ROOT", root)
    monkeypatch.setattr(git_commits, "CACHE_PATH", root / "data" / "site" / "commit_sha_cache.json")
    return root


def _write_ledger(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_committed_line_resolves_to_a_real_sha(repo):
    ledger = repo / "data" / "predictions" / "gw01.jsonl"
    _write_ledger(ledger, [{"record_id": "aaa"}])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")
    commit_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    resolved = git_commits.resolve_commit_shas_for_ledger(ledger, [{"record_id": "aaa"}])

    assert resolved == {"aaa": commit_sha}


def test_uncommitted_line_resolves_to_none_not_a_broken_link(repo):
    ledger = repo / "data" / "predictions" / "gw01.jsonl"
    _write_ledger(ledger, [{"record_id": "aaa"}])
    # never committed -- must not raise or fabricate a sha

    resolved = git_commits.resolve_commit_shas_for_ledger(ledger, [{"record_id": "aaa"}])

    assert resolved == {"aaa": None}


def test_mixed_committed_and_pending_lines(repo):
    ledger = repo / "data" / "predictions" / "gw01.jsonl"
    _write_ledger(ledger, [{"record_id": "aaa"}])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first line")
    committed_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    _write_ledger(ledger, [{"record_id": "aaa"}, {"record_id": "bbb"}])  # append, uncommitted

    resolved = git_commits.resolve_commit_shas_for_ledger(ledger, [{"record_id": "aaa"}, {"record_id": "bbb"}])

    assert resolved["aaa"] == committed_sha
    assert resolved["bbb"] is None


def test_resolved_sha_is_cached_and_not_recomputed_after_a_reformat(repo):
    """The core fragility fix: once a record_id's sha is resolved, a later
    reformat of the ledger file (which would make git blame reassign the
    line to a NEW commit) must not change the already-resolved value."""
    ledger = repo / "data" / "predictions" / "gw01.jsonl"
    _write_ledger(ledger, [{"record_id": "aaa"}])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first line")
    original_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    first = git_commits.resolve_commit_shas_for_ledger(ledger, [{"record_id": "aaa"}])
    assert first["aaa"] == original_sha
    assert git_commits.CACHE_PATH.exists()

    # simulate a reformat: rewrite the same content differently, commit again
    _write_ledger(ledger, [{"record_id": "aaa"}])  # identical content, but a new commit touches the line
    (repo / "unrelated.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "reformat")

    second = git_commits.resolve_commit_shas_for_ledger(ledger, [{"record_id": "aaa"}])
    assert second["aaa"] == original_sha  # unchanged, served from cache


def test_commit_url_parses_https_origin(repo):
    _git(repo, "remote", "add", "origin", "https://github.com/kshreyan/apex-fpl.git")
    assert git_commits.commit_url("deadbeef") == "https://github.com/kshreyan/apex-fpl/commit/deadbeef"


def test_commit_url_parses_ssh_origin(repo):
    _git(repo, "remote", "add", "origin", "git@github.com:kshreyan/apex-fpl.git")
    assert git_commits.commit_url("deadbeef") == "https://github.com/kshreyan/apex-fpl/commit/deadbeef"
