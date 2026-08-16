#!/usr/bin/env python3
"""Calibration aggregate builder (Phase 13, Stage 4) — rebuilds
data/calibration.json from the COMPLETE history of results on every run.
Never incremental: this file is a derived projection over the real
source-of-truth ledgers (data/predictions/, data/results/), not a
ledger itself, and can always be safely thrown away and regenerated —
see the source-of-truth-vs-derived-projection rule this project settled
on (CLAUDE.md).

Resolves to the LATEST (non-superseded) line per gameweek in both the
predictions and results ledgers before computing anything — a corrected
result (score.py --correct) must never get pooled together with the
result it superseded, or a correction would silently double-count rather
than replace.

Stays purely local/offline: no live API fetch. `coverage.gameweeks_
missing_prediction` (which needs to know the season's real deadline
schedule to detect a gap) is computed from whatever raw bootstrap-static
snapshot the pipeline has ALREADY captured under data/raw/ during its
own normal operation — not a new live dependency added just for this
one diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
RESULTS_DIR = REPO_ROOT / "data" / "results"
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"
CALIBRATION_PATH = REPO_ROOT / "data" / "calibration.json"

SCHEMA_VERSION = "1.0"
MIN_N_FOR_RATE_DISPLAY = 10
CALIBRATION_BIN_COUNT = 10


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _read_ledger_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gather_latest_predictions() -> dict[int, dict]:
    out = {}
    if not PREDICTIONS_DIR.exists():
        return out
    for path in sorted(PREDICTIONS_DIR.glob("gw*.jsonl")):
        gw = int(path.stem[2:])
        lines = _read_ledger_lines(path)
        if lines:
            out[gw] = lines[-1]
    return out


def gather_latest_results() -> dict[int, dict]:
    out = {}
    if not RESULTS_DIR.exists():
        return out
    for path in sorted(RESULTS_DIR.glob("gw*.jsonl")):
        gw = int(path.stem[2:])
        lines = _read_ledger_lines(path)
        if lines:
            out[gw] = lines[-1]
    return out


def compute_brier_score(binary_call_results: list[dict]) -> float | None:
    if not binary_call_results:
        return None
    total = sum((c["predicted_probability"] - (1.0 if c["outcome"] else 0.0)) ** 2 for c in binary_call_results)
    return round(total / len(binary_call_results), 4)


def compute_log_loss(binary_call_results: list[dict]) -> float | None:
    if not binary_call_results:
        return None
    eps = 1e-6
    total = 0.0
    for c in binary_call_results:
        p = min(max(c["predicted_probability"], eps), 1 - eps)
        y = 1.0 if c["outcome"] else 0.0
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return round(total / len(binary_call_results), 4)


def compute_calibration_bins(binary_call_results: list[dict], n_bins: int = CALIBRATION_BIN_COUNT, min_n: int = MIN_N_FOR_RATE_DISPLAY) -> list[dict]:
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for c in binary_call_results:
        p = min(max(c["predicted_probability"], 0.0), 1.0)
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append(c)

    out = []
    for i, items in enumerate(bins):
        n = len(items)
        entry = {"bin_range": [round(i / n_bins, 2), round((i + 1) / n_bins, 2)], "n": n, "suppressed": n < min_n}
        if n < min_n:
            entry["predicted_mean"] = None
            entry["actual_rate"] = None
        else:
            entry["predicted_mean"] = round(sum(c["predicted_probability"] for c in items) / n, 4)
            entry["actual_rate"] = round(sum(1.0 if c["outcome"] else 0.0 for c in items) / n, 4)
        out.append(entry)
    return out


def compute_points_forecast_metrics(points_call_results: list[dict]) -> dict:
    if not points_call_results:
        return {"n": 0, "mae": None, "rmse": None, "mean_error": None}
    n = len(points_call_results)
    errors = [c["error"] for c in points_call_results]
    return {
        "n": n,
        "mae": round(sum(abs(e) for e in errors) / n, 3),
        "rmse": round(math.sqrt(sum(e * e for e in errors) / n), 3),
        "mean_error": round(sum(errors) / n, 3),
    }


def compute_captaincy_hit_rate(results_by_gw: dict[int, dict]) -> dict:
    """'Hit' = captain was the actual highest scorer among the selected
    starting XI that gameweek — the exact definition locked in at
    Stage 1, restated here (metric_definitions below) so a reader never
    has to go find where it was decided."""
    hits = misses = 0
    per_gw = []
    for gw in sorted(results_by_gw):
        result = results_by_gw[gw]
        if result["status"] != "SCORED":
            continue
        player_calls = [c for c in result["call_results"] if c["type"] == "points_forecast" and c["subject"]["kind"] == "player"]
        captain_calls = [c for c in result["call_results"] if c["type"] == "points_forecast" and c["subject"]["kind"] == "captain"]
        if not player_calls or not captain_calls:
            continue
        captain_id = captain_calls[0]["subject"]["player_id"]
        captain_player_call = next((c for c in player_calls if c["subject"]["player_id"] == captain_id), None)
        if captain_player_call is None:
            continue
        is_hit = all(captain_player_call["actual"] >= c["actual"] for c in player_calls)
        per_gw.append({"gameweek": gw, "hit": is_hit})
        hits += int(is_hit)
        misses += int(not is_hit)

    n = hits + misses
    return {
        "definition": "captain was the actual highest scorer among the selected starting XI that gameweek",
        "n": n, "hits": hits, "misses": misses,
        "hit_rate": round(hits / n, 4) if n >= MIN_N_FOR_RATE_DISPLAY else None,
        "hit_rate_suppressed": n < MIN_N_FOR_RATE_DISPLAY,
        "per_gameweek": per_gw,
    }


def compute_points_vs_baselines(results_by_gw: dict[int, dict]) -> dict:
    by_gw = []
    cum_model = cum_avg = cum_template = cum_top10k = 0.0
    top10k_n = 0
    for gw in sorted(results_by_gw):
        result = results_by_gw[gw]
        if result["status"] != "SCORED":
            continue
        model_pts = result["squad_actual"]["total_points"]
        avg = result["baselines"]["average_manager_score"]
        template = result["baselines"]["template_team_score"]
        top10k = result["baselines"]["top_10k_average"]
        cum_model += model_pts
        cum_avg += avg
        cum_template += template
        if top10k is not None:
            cum_top10k += top10k
            top10k_n += 1
        by_gw.append({"gameweek": gw, "model": model_pts, "average_manager": avg, "template_team": template, "top_10k": top10k})

    return {
        "cumulative_model_points": round(cum_model, 2),
        "cumulative_average_manager_points": round(cum_avg, 2),
        "cumulative_template_team_points": round(cum_template, 2),
        "cumulative_top_10k_points": round(cum_top10k, 2) if top10k_n > 0 else None,
        "diff_vs_average_manager": round(cum_model - cum_avg, 2),
        "diff_vs_template_team": round(cum_model - cum_template, 2),
        "diff_vs_top_10k": round(cum_model - cum_top10k, 2) if top10k_n > 0 else None,
        "by_gameweek": by_gw,
    }


def _latest_captured_bootstrap_static() -> dict | None:
    """Reuses whatever raw snapshot the pipeline has already captured
    during normal operation (data/raw/gw*/bootstrap_static/*.json) --
    deliberately not a new live fetch, see module docstring."""
    if not RAW_DATA_ROOT.exists():
        return None
    candidates = sorted(RAW_DATA_ROOT.glob("gw*/bootstrap_static/*.json"))
    candidates = [p for p in candidates if not p.name.endswith(".meta.json")]
    if not candidates:
        return None
    return json.loads(candidates[-1].read_bytes())


BIGGEST_MISSES_TOP_N = 5


def compute_biggest_misses(results_by_gw: dict[int, dict], top_n: int = BIGGEST_MISSES_TOP_N) -> dict:
    """The strongest trust signal on the homepage, by explicit design
    decision: aggregate metrics alone let a reader suspect favourable
    framing, so this surfaces the model's worst individual calls by
    name, automatically, every rebuild -- not curated, not removable
    without removing the whole section.

    Two separate rankings, since "biggest miss" means something different
    per call type: points_forecast has no notion of confidence, so it's
    ranked by raw |error|; binary_probability misses are ranked by their
    per-call Brier contribution (predicted_probability - outcome)^2 --
    a confident, wrong call (p=0.9, outcome False) ranks above a
    hedged, wrong one (p=0.55, outcome False), which is the point: this
    section is about confident misses, not just any miss."""
    points_misses = []
    probability_misses = []
    for gw, result in results_by_gw.items():
        if result["status"] != "SCORED":
            continue
        for c in result["call_results"]:
            if c["type"] == "points_forecast":
                points_misses.append({
                    "gameweek": gw, "call_id": c["call_id"], "subject": c["subject"],
                    "predicted": c["predicted"], "actual": c["actual"], "error": c["error"],
                })
            elif c["type"] == "binary_probability":
                brier = (c["predicted_probability"] - (1.0 if c["outcome"] else 0.0)) ** 2
                probability_misses.append({
                    "gameweek": gw, "call_id": c["call_id"], "subject": c["subject"],
                    "predicted_probability": c["predicted_probability"], "outcome": c["outcome"],
                    "brier_contribution": round(brier, 4),
                })

    points_misses.sort(key=lambda m: -abs(m["error"]))
    probability_misses.sort(key=lambda m: -m["brier_contribution"])
    return {
        "points_forecast": points_misses[:top_n],
        "binary_probability": probability_misses[:top_n],
    }


def compute_coverage(predictions_by_gw: dict[int, dict], results_by_gw: dict[int, dict]) -> dict:
    gameweeks_published = sorted(predictions_by_gw)
    gameweeks_scored = sorted(gw for gw, r in results_by_gw.items() if r["status"] == "SCORED")
    gameweeks_blank = sorted(gw for gw, r in predictions_by_gw.items() if r["status"] == "BLANK_GAMEWEEK")

    bootstrap_static = _latest_captured_bootstrap_static()
    gameweeks_missing_prediction: list[int] | None = None
    total_gameweeks_season: int | None = None
    if bootstrap_static is not None:
        now = datetime.now(timezone.utc)
        total_gameweeks_season = len(bootstrap_static["events"])
        missing = []
        for event in bootstrap_static["events"]:
            gw = event["id"]
            deadline = datetime.strptime(event["deadline_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            if deadline < now and gw not in predictions_by_gw:
                missing.append(gw)
        gameweeks_missing_prediction = sorted(missing)

    return {
        "gameweeks_published": gameweeks_published,
        "gameweeks_scored": gameweeks_scored,
        "gameweeks_blank": gameweeks_blank,
        "gameweeks_missing_prediction": gameweeks_missing_prediction,
        "total_gameweeks_season": total_gameweeks_season,
    }


def build_calibration() -> dict:
    predictions_by_gw = gather_latest_predictions()
    results_by_gw = gather_latest_results()

    points_calls, binary_calls = [], []
    by_call_kind: dict[str, list[dict]] = {}
    for result in results_by_gw.values():
        if result["status"] != "SCORED":
            continue
        for c in result["call_results"]:
            if c["type"] == "points_forecast":
                points_calls.append(c)
                by_call_kind.setdefault(c["subject"]["kind"], []).append(c)
            elif c["type"] == "binary_probability":
                binary_calls.append(c)

    points_metrics = compute_points_forecast_metrics(points_calls)
    points_metrics["by_call_kind"] = {kind: compute_points_forecast_metrics(calls) for kind, calls in by_call_kind.items()}

    return {
        "schema_version": SCHEMA_VERSION,
        "rebuilt_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": _git_sha(),
        "coverage": compute_coverage(predictions_by_gw, results_by_gw),
        "points_forecast_metrics": points_metrics,
        "probability_metrics": {
            "n": len(binary_calls),
            "brier_score": compute_brier_score(binary_calls),
            "log_loss": compute_log_loss(binary_calls),
            "calibration_bins": compute_calibration_bins(binary_calls),
        },
        "points_vs_baselines": compute_points_vs_baselines(results_by_gw),
        "captaincy": compute_captaincy_hit_rate(results_by_gw),
        "biggest_misses": compute_biggest_misses(results_by_gw),
        "metric_definitions": {
            "brier_score": "mean((predicted_probability - outcome)^2) over all scored binary_probability calls",
            "log_loss": "-mean(outcome*ln(p) + (1-outcome)*ln(1-p)), p clipped to [1e-6, 1-1e-6]",
            "mae": "mean(abs(predicted - actual)) over all scored points_forecast calls",
            "captaincy_hit": "captain was the actual highest scorer among the selected starting XI that gameweek",
        },
        "small_sample_policy": {
            "min_n_for_rate_display": MIN_N_FOR_RATE_DISPLAY,
            "behavior_below_min": "the raw count is always shown; any rate/percentage computed from fewer than min_n observations is reported as null (suppressed), not a misleadingly precise number",
        },
    }


def run() -> dict:
    calibration = build_calibration()
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2, sort_keys=True))
    print(f"Rebuilt {_display_path(CALIBRATION_PATH)}")
    print(f"  coverage: {calibration['coverage']}")
    print(f"  points_forecast_metrics: n={calibration['points_forecast_metrics']['n']} mae={calibration['points_forecast_metrics']['mae']}")
    print(f"  probability_metrics: n={calibration['probability_metrics']['n']} brier={calibration['probability_metrics']['brier_score']}")
    print(f"  captaincy: {calibration['captaincy']['hits']}/{calibration['captaincy']['n']}")
    return calibration


if __name__ == "__main__":
    run()
