from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from pipeline import healthcheck
from pipeline.site import git_commits


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_ledger(path, records):
    _write(path, "".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "remote", "add", "origin", "https://github.com/kshreyan/apex-fpl.git")

    monkeypatch.setattr(healthcheck, "REPO_ROOT", root)
    monkeypatch.setattr(healthcheck, "PREDICTIONS_DIR", root / "data" / "predictions")
    monkeypatch.setattr(healthcheck, "RESULTS_DIR", root / "data" / "results")
    monkeypatch.setattr(healthcheck, "CALIBRATION_PATH", root / "data" / "calibration.json")
    monkeypatch.setattr(healthcheck, "WORKFLOWS_DIR", root / ".github" / "workflows")
    monkeypatch.setattr(git_commits, "REPO_ROOT", root)
    monkeypatch.setattr(git_commits, "CACHE_PATH", root / "data" / "site" / "commit_sha_cache.json")
    return root


def _content_hash(without_id):
    return hashlib.sha256(json.dumps(without_id, sort_keys=True).encode()).hexdigest()


def _make_record(gw=1, supersedes=None, salt="a"):
    body = {"schema_version": "1.0", "supersedes": supersedes, "gameweek": gw, "status": "PUBLISHED", "salt": salt}
    return {**body, "record_id": _content_hash(body)}


# --------------------------------------------------------- ledger integrity

def test_ledger_integrity_passes_on_a_well_formed_chain(repo):
    r1 = _make_record(salt="one")
    r2 = _make_record(supersedes=r1["record_id"], salt="two")
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [r1, r2])

    result = healthcheck.check_ledger_integrity()

    assert result.passed, result.detail


def test_ledger_integrity_catches_a_tampered_record_id(repo):
    r1 = _make_record()
    r1["record_id"] = "0" * 64  # tampered -- doesn't match recomputed hash
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [r1])

    result = healthcheck.check_ledger_integrity()

    assert not result.passed
    assert "recomputed content hash" in result.detail


def test_ledger_integrity_catches_a_broken_supersedes_chain(repo):
    r1 = _make_record(salt="one")
    r2 = _make_record(supersedes="not-the-real-prior-id", salt="two")
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [r1, r2])

    result = healthcheck.check_ledger_integrity()

    assert not result.passed
    assert "supersedes" in result.detail


def test_ledger_integrity_catches_invalid_json(repo):
    path = repo / "data" / "predictions" / "gw01.jsonl"
    _write(path, "not json at all\n")

    result = healthcheck.check_ledger_integrity()

    assert not result.passed
    assert "invalid JSON" in result.detail


def test_ledger_integrity_catches_duplicate_record_ids(repo):
    r1 = _make_record(gw=1, salt="x")
    r2 = dict(r1)  # exact duplicate record_id, different gameweek file
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [r1])
    _write_ledger(repo / "data" / "predictions" / "gw02.jsonl", [r2])

    result = healthcheck.check_ledger_integrity()

    assert not result.passed
    assert "duplicate record_id" in result.detail


# --------------------------------------------------------- blame cache -----

def test_blame_cache_integrity_passes_when_cache_agrees_with_fresh_blame(repo):
    ledger = repo / "data" / "predictions" / "gw01.jsonl"
    r1 = _make_record()
    _write_ledger(ledger, [r1])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1")

    git_commits.resolve_commit_shas_for_ledger(ledger, [r1])  # populates the cache

    result = healthcheck.check_blame_cache_integrity()

    assert result.passed, result.detail


def test_blame_cache_integrity_catches_a_stale_cache_entry(repo):
    ledger = repo / "data" / "predictions" / "gw01.jsonl"
    r1 = _make_record()
    _write_ledger(ledger, [r1])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1")
    git_commits.resolve_commit_shas_for_ledger(ledger, [r1])

    # simulate a corrupted/hand-edited cache -- a wrong SHA on record r1
    cache_path = git_commits.CACHE_PATH
    cache = json.loads(cache_path.read_text())
    cache[r1["record_id"]] = "f" * 40
    cache_path.write_text(json.dumps(cache))

    result = healthcheck.check_blame_cache_integrity()

    assert not result.passed
    assert r1["record_id"][:12] in result.detail


# --------------------------------------------------------- sha pinning -----

def test_sha_pinning_passes_when_every_uses_line_is_pinned(repo):
    _write(repo / ".github" / "workflows" / "pipeline.yml", "uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262\n")

    result = healthcheck.check_workflow_actions_are_sha_pinned()

    assert result.passed, result.detail


def test_sha_pinning_catches_a_floating_tag(repo):
    _write(repo / ".github" / "workflows" / "pipeline.yml", "uses: actions/checkout@v4\n")

    result = healthcheck.check_workflow_actions_are_sha_pinned()

    assert not result.passed
    assert "actions/checkout@v4" in result.detail


# --------------------------------------------------------- gitignore -------

def test_raw_not_gitignored_passes_when_data_raw_is_tracked(repo):
    _write(repo / ".gitignore", "data/logs/\n")

    result = healthcheck.check_raw_captures_not_gitignored()

    assert result.passed, result.detail


def test_raw_not_gitignored_catches_the_regression(repo):
    _write(repo / ".gitignore", "data/raw/\n")

    result = healthcheck.check_raw_captures_not_gitignored()

    assert not result.passed
    assert "gitignored" in result.detail


# --------------------------------------------------------- raw hashes ------

def test_raw_capture_hashes_pass_when_they_match(repo):
    raw_path = repo / "data" / "raw" / "gw01" / "bootstrap_static" / "snap.json"
    _write(raw_path, '{"x": 1}')
    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    pred = {
        "record_id": "irrelevant", "data_sources": [
            {"source": "bootstrap_static", "raw_cache_path": "data/raw/gw01/bootstrap_static/snap.json", "sha256": sha}
        ],
    }
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])

    result = healthcheck.check_raw_capture_hashes()

    assert result.passed, result.detail


def test_raw_capture_hashes_catch_a_mismatch(repo):
    raw_path = repo / "data" / "raw" / "gw01" / "bootstrap_static" / "snap.json"
    _write(raw_path, '{"x": 1}')
    pred = {
        "record_id": "irrelevant", "data_sources": [
            {"source": "bootstrap_static", "raw_cache_path": "data/raw/gw01/bootstrap_static/snap.json", "sha256": "0" * 64}
        ],
    }
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])

    result = healthcheck.check_raw_capture_hashes()

    assert not result.passed
    assert "mismatch" in result.detail


def test_raw_capture_hashes_catch_a_missing_file(repo):
    pred = {
        "record_id": "irrelevant", "data_sources": [
            {"source": "bootstrap_static", "raw_cache_path": "data/raw/gw01/bootstrap_static/gone.json", "sha256": "0" * 64}
        ],
    }
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])

    result = healthcheck.check_raw_capture_hashes()

    assert not result.passed
    assert "missing" in result.detail


# ------------------------------------------------- unavailable-player check

def _bootstrap_with_elements(elements):
    return {"elements": elements}


def _squad_player(pid, name="Player"):
    return {"player_id": pid, "name": name, "position": "MID", "team": "Arsenal", "price": 5.0}


def _published_prediction(raw_path_rel, squad_player_ids):
    return {
        "record_id": "irrelevant", "status": "PUBLISHED",
        "data_sources": [{"source": "bootstrap_static", "raw_cache_path": raw_path_rel, "sha256": "unused-by-this-check"}],
        "squad": {
            "starting_xi": [_squad_player(pid) for pid in squad_player_ids[:11]],
            "bench_order": [_squad_player(pid) for pid in squad_player_ids[11:]],
            "captain_player_id": squad_player_ids[0], "vice_captain_player_id": squad_player_ids[1],
        },
    }


def test_no_unavailable_in_squad_passes_for_a_fully_available_squad(repo):
    raw_path = repo / "data" / "raw" / "gw01" / "bootstrap_static" / "snap.json"
    elements = [{"id": i, "status": "a", "chance_of_playing_this_round": None, "chance_of_playing_next_round": None} for i in range(1, 16)]
    _write(raw_path, json.dumps(_bootstrap_with_elements(elements)))
    pred = _published_prediction("data/raw/gw01/bootstrap_static/snap.json", [str(i) for i in range(1, 16)])
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])

    result = healthcheck.check_no_unavailable_player_in_published_squad()

    assert result.passed, result.detail


def test_no_unavailable_in_squad_catches_a_zero_chance_player_in_the_squad(repo):
    raw_path = repo / "data" / "raw" / "gw01" / "bootstrap_static" / "snap.json"
    elements = [{"id": i, "status": "a", "chance_of_playing_this_round": None, "chance_of_playing_next_round": None} for i in range(1, 16)]
    elements[0] = {"id": 1, "status": "i", "chance_of_playing_this_round": None, "chance_of_playing_next_round": 0, "web_name": "Injured"}
    _write(raw_path, json.dumps(_bootstrap_with_elements(elements)))
    pred = _published_prediction("data/raw/gw01/bootstrap_static/snap.json", [str(i) for i in range(1, 16)])
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])

    result = healthcheck.check_no_unavailable_player_in_published_squad()

    assert not result.passed
    assert "status='i'" in result.detail


def test_no_unavailable_in_squad_ignores_non_published_predictions(repo):
    raw_path = repo / "data" / "raw" / "gw01" / "bootstrap_static" / "snap.json"
    elements = [{"id": 1, "status": "i", "chance_of_playing_this_round": None, "chance_of_playing_next_round": 0}]
    _write(raw_path, json.dumps(_bootstrap_with_elements(elements)))
    pred = {"record_id": "irrelevant", "status": "BLANK_GAMEWEEK", "data_sources": [], "squad": None}
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])

    result = healthcheck.check_no_unavailable_player_in_published_squad()

    assert result.passed, result.detail


# --------------------------------------------------------- calibration.json

def test_calibration_json_passes_with_a_real_source_commit(repo):
    _git(repo, "commit", "-q", "--allow-empty", "-m", "seed")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _write(repo / "data" / "calibration.json", json.dumps({"source_commit": sha}))

    result = healthcheck.check_calibration_json()

    assert result.passed, result.detail


def test_calibration_json_catches_an_unreachable_source_commit(repo):
    _write(repo / "data" / "calibration.json", json.dumps({"source_commit": "f" * 40}))

    result = healthcheck.check_calibration_json()

    assert not result.passed
    assert "not a commit reachable" in result.detail


def test_calibration_json_catches_a_missing_file(repo):
    result = healthcheck.check_calibration_json()

    assert not result.passed
    assert "does not exist" in result.detail


# --------------------------------------------------------- gh-backed checks
#
# healthcheck.subprocess and git_commits.subprocess are the SAME module
# object (both do `import subprocess`), so monkeypatching `.run` on one
# intercepts calls made through the other too -- including
# _origin_owner_repo()'s own `git remote get-url origin` call, which
# these checks call first. A fake that unconditionally returns `gh`-
# shaped JSON breaks that call silently. Real bug, caught by running
# these tests, not a hypothetical: dispatch on the command instead of
# assuming every call is the one being faked.
_real_run = subprocess.run


def _fake_gh(gh_stdout):
    def fake_run(args, **kwargs):
        if args[0] == "gh":
            return subprocess.CompletedProcess(args, 0, stdout=gh_stdout, stderr="")
        return _real_run(args, **kwargs)
    return fake_run


def test_pages_source_passes_when_build_type_is_workflow(repo, monkeypatch):
    monkeypatch.setattr(healthcheck.subprocess, "run", _fake_gh(json.dumps({"build_type": "workflow"})))

    result = healthcheck.check_pages_source_is_actions()

    assert result.passed, result.detail


def test_pages_source_catches_a_branch_based_source(repo, monkeypatch):
    monkeypatch.setattr(healthcheck.subprocess, "run", _fake_gh(json.dumps({"build_type": "legacy"})))

    result = healthcheck.check_pages_source_is_actions()

    assert not result.passed
    assert "legacy" in result.detail


def test_workflow_recency_passes_for_a_recent_run(repo, monkeypatch):
    from datetime import datetime, timezone

    recent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    monkeypatch.setattr(healthcheck.subprocess, "run", _fake_gh(json.dumps([{"createdAt": recent}])))

    result = healthcheck.check_scheduled_workflow_recency()

    assert result.passed, result.detail


def test_workflow_recency_catches_a_stale_run(repo, monkeypatch):
    stale = "2020-01-01T00:00:00Z"
    monkeypatch.setattr(healthcheck.subprocess, "run", _fake_gh(json.dumps([{"createdAt": stale}])))

    result = healthcheck.check_scheduled_workflow_recency()

    assert not result.passed
    assert "may have stopped firing" in result.detail


def test_workflow_recency_catches_zero_successful_runs(repo, monkeypatch):
    monkeypatch.setattr(healthcheck.subprocess, "run", _fake_gh("[]"))

    result = healthcheck.check_scheduled_workflow_recency()

    assert not result.passed
    assert "no successful" in result.detail


# --------------------------------------------------------- run() aggregate -

def test_run_returns_nonzero_when_any_local_check_fails(repo):
    _write(repo / "data" / "predictions" / "gw01.jsonl", "not json\n")

    code = healthcheck.run(include_gh_checks=False)

    assert code == 1


def test_run_returns_zero_when_everything_passes(repo, monkeypatch):
    _git(repo, "commit", "-q", "--allow-empty", "-m", "seed")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _write(repo / "data" / "calibration.json", json.dumps({"source_commit": sha}))
    _write(repo / ".gitignore", "data/logs/\n")
    _write(repo / ".github" / "workflows" / "pipeline.yml", "uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262\n")

    code = healthcheck.run(include_gh_checks=False)

    assert code == 0
