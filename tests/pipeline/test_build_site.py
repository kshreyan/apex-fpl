from __future__ import annotations

import json
import subprocess

import pytest

from pipeline import build_site
from pipeline.site import git_commits


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _write_ledger(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "remote", "add", "origin", "https://github.com/kshreyan/apex-fpl.git")

    monkeypatch.setattr(build_site, "REPO_ROOT", root)
    monkeypatch.setattr(build_site, "PREDICTIONS_DIR", root / "data" / "predictions")
    monkeypatch.setattr(build_site, "RESULTS_DIR", root / "data" / "results")
    monkeypatch.setattr(build_site, "CALIBRATION_PATH", root / "data" / "calibration.json")
    monkeypatch.setattr(build_site, "DOCS_ROOT", root / "docs")
    monkeypatch.setattr(git_commits, "REPO_ROOT", root)
    monkeypatch.setattr(git_commits, "CACHE_PATH", root / "data" / "site" / "commit_sha_cache.json")
    return root


def _squad(captain="2"):
    return {
        "starting_xi": [
            {"player_id": "1", "name": "Player One", "position": "MID", "team": "Arsenal", "price": 6.5},
            {"player_id": "2", "name": "Player Two", "position": "FWD", "team": "Chelsea", "price": 9.0},
        ],
        "bench_order": [{"player_id": "3", "name": "Sub One", "position": "GK", "team": "Fulham", "price": 4.5}],
        "captain_player_id": captain, "vice_captain_player_id": "1",
    }


def _prediction(gw, record_id="p1", status="PUBLISHED", captain="2"):
    return {
        "record_id": record_id, "supersedes": None, "gameweek": gw, "status": status,
        "deadline_time_utc": "2026-08-21T17:30:00Z",
        "squad": _squad(captain) if status == "PUBLISHED" else None,
        "calls": [
            {"id": f"gw{gw:02d}-player-1-points", "type": "points_forecast", "subject": {"kind": "player", "player_id": "1"}, "claim": "Player One projected 4", "value": 4.0},
            {"id": f"gw{gw:02d}-captain-haul", "type": "binary_probability", "subject": {"kind": "captain", "player_id": captain, "threshold": 6}, "claim": "Captain scores 6+", "probability": 0.4},
        ] if status == "PUBLISHED" else [],
    }


def _minimal_calibration(missing=None):
    return {
        "schema_version": "1.0", "rebuilt_at_utc": "2026-08-16T12:00:00Z", "source_commit": "abc",
        "coverage": {"gameweeks_published": [], "gameweeks_scored": [], "gameweeks_blank": [], "gameweeks_missing_prediction": missing or [], "total_gameweeks_season": 38},
        "points_forecast_metrics": {"n": 0, "mae": None, "rmse": None, "mean_error": None, "by_call_kind": {}},
        "probability_metrics": {"n": 0, "brier_score": None, "log_loss": None, "calibration_bins": [
            {"bin_range": [i / 10, (i + 1) / 10], "n": 0, "suppressed": True, "predicted_mean": None, "actual_rate": None} for i in range(10)
        ]},
        "points_vs_baselines": {"by_gameweek": [], "cumulative_average_manager_points": 0.0, "cumulative_model_points": 0.0, "cumulative_template_team_points": 0.0, "cumulative_top_10k_points": None, "diff_vs_average_manager": 0.0, "diff_vs_template_team": 0.0, "diff_vs_top_10k": None},
        "captaincy": {"definition": "d", "hit_rate": None, "hit_rate_suppressed": True, "hits": 0, "misses": 0, "n": 0, "per_gameweek": []},
        "biggest_misses": {"points_forecast": [], "binary_probability": []},
        "metric_definitions": {}, "small_sample_policy": {"min_n_for_rate_display": 10, "behavior_below_min": "x"},
    }


def test_run_fails_loudly_when_calibration_missing(repo):
    with pytest.raises(RuntimeError, match="calibration"):
        build_site.run()


def test_run_builds_index_current_methodology_and_gameweek_pages(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [_prediction(1)])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")

    build_site.run()

    docs = repo / "docs"
    assert (docs / "index.html").exists()
    assert (docs / "current" / "index.html").exists()
    assert (docs / "methodology" / "index.html").exists()
    assert (docs / "gameweek" / "gw01" / "index.html").exists()
    assert (docs / "assets" / "site.css").exists()
    assert (docs / "assets" / "staleness.js").exists()


def test_does_not_touch_preexisting_unrelated_docs_files(repo):
    (repo / "docs").mkdir()
    (repo / "docs" / "phase6_joint_simulation_report.md").write_text("# untouched research report")
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")

    build_site.run()

    assert (repo / "docs" / "phase6_joint_simulation_report.md").read_text() == "# untouched research report"


def test_current_page_never_shows_picks_without_the_record_section(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [_prediction(1)])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")

    build_site.run()

    html = (repo / "docs" / "current" / "index.html").read_text()
    record_idx = html.index("Season record")
    picks_idx = html.index("Selected squad")
    assert record_idx < picks_idx  # record section renders before picks, unconditionally


def test_current_page_states_no_live_record_at_gw1(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [_prediction(1)])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")

    build_site.run()

    html = (repo / "docs" / "current" / "index.html").read_text()
    assert "No gameweeks have been scored yet" in html


def test_missing_prediction_gap_is_permanent_on_gameweek_page_and_homepage(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration(missing=[7])))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")

    build_site.run()

    gw_page = (repo / "docs" / "gameweek" / "gw07" / "index.html").read_text()
    assert "pipeline failure" in gw_page.lower()
    home = (repo / "docs" / "index.html").read_text()
    assert "GW7" in home
    assert "pipeline gap" in home.lower() or "pipeline failure" in home.lower()


def test_gameweek_page_links_to_the_commit_that_recorded_the_prediction(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [_prediction(1)])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")
    sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    build_site.run()

    gw_page = (repo / "docs" / "gameweek" / "gw01" / "index.html").read_text()
    assert f"https://github.com/kshreyan/apex-fpl/commit/{sha}" in gw_page


def test_uncommitted_prediction_shows_pending_not_a_broken_link(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed calibration only")
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [_prediction(1)])  # never committed

    build_site.run()

    gw_page = (repo / "docs" / "gameweek" / "gw01" / "index.html").read_text()
    assert "Commit pending" in gw_page


def test_hostile_player_name_is_escaped_not_rendered_as_markup(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    pred = _prediction(1)
    pred["squad"]["starting_xi"][0]["name"] = '<img src=x onerror=alert(1)>Evil'
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [pred])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")

    build_site.run()

    gw_page = (repo / "docs" / "gameweek" / "gw01" / "index.html").read_text()
    assert "<img src=x" not in gw_page
    assert "&lt;img" in gw_page


def test_rebuild_is_byte_identical_for_unchanged_data_and_git_state(repo):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(_minimal_calibration()))
    _write_ledger(repo / "data" / "predictions" / "gw01.jsonl", [_prediction(1)])
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "gw1 prediction")

    build_site.run()
    first = (repo / "docs" / "index.html").read_text()
    first_gw = (repo / "docs" / "gameweek" / "gw01" / "index.html").read_text()

    build_site.run()
    second = (repo / "docs" / "index.html").read_text()
    second_gw = (repo / "docs" / "gameweek" / "gw01" / "index.html").read_text()

    assert first == second
    assert first_gw == second_gw


def test_biggest_misses_section_renders_on_homepage(repo):
    cal = _minimal_calibration()
    cal["biggest_misses"] = {
        "points_forecast": [{"gameweek": 3, "call_id": "gw03-player-9-points", "subject": {"kind": "player", "player_id": "9"}, "predicted": 2.0, "actual": 15.0, "error": 13.0}],
        "binary_probability": [{"gameweek": 3, "call_id": "gw03-captain-haul", "subject": {"kind": "captain", "player_id": "9", "threshold": 6}, "predicted_probability": 0.9, "outcome": False, "brier_contribution": 0.81}],
    }
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "calibration.json").write_text(json.dumps(cal))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "seed")

    build_site.run()

    home = (repo / "docs" / "index.html").read_text()
    assert "Biggest misses this season" in home
    assert "GW3" in home
