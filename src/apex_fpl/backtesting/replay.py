"""Deadline-by-deadline historical replay (spec Part XXXIII).

For a single historical gameweek, reconstructs the world strictly before
that gameweek's deadline, generates a squad recommendation, freezes it to
disk, then reveals and scores the actual outcome. This is the same
pipeline the Phase 2 milestone proved end-to-end
(docs/phase2_milestone_report.md) — extracted here into a reusable
function so Phase 3 can call it in a loop across many gameweeks without
duplicating the logic, and so later phases can reuse it too.

The model "lives through history sequentially": each call to
run_gameweek() independently refits the team model on exactly the
fixtures available before that gameweek and rebuilds player history from
exactly the rows available before that gameweek. Nothing is cached or
carried over between calls that could leak a later gameweek's information
into an earlier one.

Scope note: this does NOT yet evolve a single squad across gameweeks with
transfers/hits/chips (that sequential decision process is Phase 7). Each
gameweek here is an independent "best XV from scratch" selection — the
correct scope for Phase 3, which exists to answer "does the underlying
forecasting pipeline work across many deadlines," not "what is the optimal
season-long transfer strategy."
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.models.attacking import challengers as attacking_challengers
from apex_fpl.models.attacking import proportional as prop
from apex_fpl.calibration import production_calibrators as prod_cal
from apex_fpl.models.minutes import challengers as minutes_challengers
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[3]


class BlankGameweekError(ValueError):
    """Raised when a target gameweek has zero fixtures (a genuine blank
    gameweek), as distinct from any other failure mode — callers looping
    across a season should treat this as an expected, informative skip,
    not a bug to be alarmed by."""


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    """Path relative to REPO_ROOT when possible (the normal case), else the
    absolute path — source files can legitimately live outside the repo
    (e.g. a leakage test pointing at a tmp directory)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class GameweekReplayResult:
    season: str
    target_gw: int
    recommendation: dict
    evaluation: dict
    recommendation_path: Path
    evaluation_path: Path


def run_gameweek(
    season: str,
    target_gw: int,
    lookback: int = 15,
    minutes_halflife: float = 3.0,
    attacking_alpha: float = 10.0,
    apply_minutes_calibration: bool = True,
    artifact_dir: Path | None = None,
    sim_batch: int = 3000,
    sim_max: int = 60000,
    sim_tol: float = 0.05,
    verbose: bool = False,
) -> GameweekReplayResult:
    """Defaults as of 2026-08-15 are the PROMOTED Phase 4b champions, not
    the original Phase 2/3 baseline: minutes uses exponential-recency-decay
    (half_life_matches=3.0) rather than a flat 6-gameweek window, and
    attacking allocation uses Dirichlet/shrinkage-smoothed shares
    (alpha=10.0) rather than raw proportional shares. Both were
    block-bootstrap validated to significantly beat their Phase 2/3
    predecessors — see docs/phase4b_tournament_report.md. `lookback=15` now
    governs the underlying history window feeding the attacking model's
    shares (the tuned value from that tournament); the minutes model no
    longer uses `lookback` at all (exponential decay has its own internal
    window). As of 2026-08-15, minutes P(60+) is ALSO calibrated by default
    (`apply_minutes_calibration=True`) using the isotonic calibrator fit
    and validated in Phase 5 (docs/phase5_calibration_report.md) — a
    large, block-bootstrap-significant log-loss improvement (raw 0.7277 ->
    calibrated 0.3072). P(appearance) is left uncalibrated (never
    validated). Pass apply_minutes_calibration=False to reproduce the
    pre-calibration Phase 4b behavior exactly. The original artifacts in
    artifacts/phase2_milestone/ and artifacts/phase3_replay/ remain as the
    frozen historical record of what earlier model versions actually
    produced at the time and are not rewritten by re-running with new
    defaults.
    """
    artifact_dir = artifact_dir or (REPO_ROOT / "artifacts" / "phase3_replay" / season)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # ---- 1. Team model: fit strictly on fixtures before the target GW ----
    training_fixtures = vl.fixtures_before_gw(season, target_gw)
    if not training_fixtures:
        raise ValueError(f"no training fixtures before GW{target_gw} in {season} — cannot fit a team model this early")
    team_model = ad.fit(training_fixtures)
    target_fixtures = vl.fixtures_at_gw(season, target_gw)
    log(f"GW{target_gw}: {len(training_fixtures)} training fixtures, {len(target_fixtures)} target fixtures")

    if not target_fixtures:
        # A genuine blank gameweek (e.g. 2022-23 GW7, postponed following
        # Queen Elizabeth II's death and redistributed to later
        # gameweeks) — not a bug. Fail clearly and specifically here
        # rather than obscurely inside the simulator once it discovers it
        # has no players to simulate, per spec Part XXXIII's requirement
        # to explicitly handle blank gameweeks rather than silently
        # mishandle them.
        raise BlankGameweekError(f"{season} GW{target_gw} has zero fixtures — blank gameweek, no recommendation possible")

    # Always-on leakage guard: every training fixture must genuinely precede
    # every target-GW fixture in kickoff time. This is cheap and catches a
    # whole class of off-by-one/data-ordering bugs immediately rather than
    # relying solely on the separate leakage test suite to notice later.
    if target_fixtures:
        max_training_date = max(f.date for f in training_fixtures)
        min_target_date = min(fx["date"] for fx in target_fixtures)
        if max_training_date >= min_target_date:
            raise AssertionError(
                f"leakage guard tripped: a training fixture ({max_training_date}) is not "
                f"strictly before the earliest target GW{target_gw} fixture ({min_target_date})"
            )

    fixture_inputs, fixture_meta = [], []
    for fx in target_fixtures:
        eh, ea = team_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
        m = sl.score_matrix(eh, ea)
        fixture_inputs.append(mc.FixtureInput(home_team=fx["home_team"], away_team=fx["away_team"], score_matrix=m))
        fixture_meta.append({
            "home_team": fx["home_team"], "away_team": fx["away_team"],
            "expected_home_goals": round(eh, 3), "expected_away_goals": round(ea, 3),
        })

    # ---- 2. Player-level history strictly before the target GW ----
    all_rows = vl.load_merged_gw(season)
    # From 2024-25 GW23 onward, FPL added a selectable "Manager" pseudo-
    # player mechanic (real managers like "Fabian Hurzeler" with
    # position="AM", scored via separate mng_* fields — see the mng_*
    # fields already noted in configs/seasons/2026_27.yaml's live API
    # capture). These are not part of the classic 15-player squad
    # (2 GK/5 DEF/5 MID/3 FWD) this optimizer builds and must be excluded
    # from the roster entirely, not crash on an unrecognized position.
    all_rows = [r for r in all_rows if r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]
    history_rows = [r for r in all_rows if int(r["GW"]) < target_gw]
    target_rows = [r for r in all_rows if int(r["GW"]) == target_gw]
    assert all(int(r["GW"]) < target_gw for r in history_rows), "leakage guard: history_rows contained a row at/after target_gw"

    minutes_by_player: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in history_rows:
        minutes_by_player[r["element"]].append((int(r["GW"]), int(r["minutes"])))
    for pid in minutes_by_player:
        minutes_by_player[pid].sort(key=lambda t: t[0])

    lookback_floor = target_gw - lookback
    team_player_events: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for r in history_rows:
        gw = int(r["GW"])
        if gw < lookback_floor:
            continue
        team_player_events[r["team"]][r["element"]].append((int(r["goals_scored"]), int(r["assists"])))

    # ---- 3. Build PlayerInput for every player in the target GW's roster universe ----
    roster: dict[str, dict] = {r["element"]: r for r in target_rows}
    shares_by_team = {
        team: attacking_challengers.shrinkage_share(dict(hist), alpha=attacking_alpha)
        for team, hist in team_player_events.items()
    }
    fixture_expected_goals = {(f["home_team"], "home"): f["expected_home_goals"] for f in fixture_meta} \
        | {(f["away_team"], "away"): f["expected_away_goals"] for f in fixture_meta}
    team_side = {}
    for f in fixture_meta:
        team_side[f["home_team"]] = "home"
        team_side[f["away_team"]] = "away"

    players_for_sim, candidates_meta = [], {}
    for pid, row in roster.items():
        team = row["team"]
        if team not in team_side:
            continue
        side = team_side[team]
        team_exp_goals = fixture_expected_goals[(team, side)]

        hist_minutes = [m for _, m in minutes_by_player.get(pid, [])]
        mfc = minutes_challengers.exponential_decay(hist_minutes, half_life_matches=minutes_halflife)
        if apply_minutes_calibration:
            mfc = prod_cal.apply_minutes_calibration(mfc)

        shares = shares_by_team.get(team, {})
        player_share = shares.get(pid, prop.AttackingShare(0.0, 0.0))
        alloc = prop.allocate(team_exp_goals, {pid: player_share})[pid]

        players_for_sim.append(mc.PlayerInput(
            player_id=pid, team=team, position=row["position"],
            minutes_forecast=mfc, expected_goals=alloc[0], expected_assists=alloc[1],
        ))
        candidates_meta[pid] = {"name": row["name"], "team": team, "position": row["position"],
                                 "price": int(row["value"]) / 10.0}

    log(f"  players entering simulation: {len(players_for_sim)}")

    # ---- 4. Monte Carlo simulation ----
    rules = scoring.load_scoring_rules("2026_27")
    sim_results = mc.simulate_gameweek(fixture_inputs, players_for_sim, rules,
                                        batch=sim_batch, max_sims=sim_max, tol=sim_tol)
    total_sims = len(next(iter(sim_results.values())).samples) if sim_results else 0

    # ---- 5. Optimizer ----
    candidates = [
        sq.PlayerCandidate(pid, meta["position"], meta["team"], meta["price"], sim_results[pid].mean_points)
        for pid, meta in candidates_meta.items()
    ]
    squad = sq.select_squad(candidates, budget=sq.BUDGET)
    xi = sq.select_starting_xi(squad)
    log(f"  captain: {candidates_meta[xi.captain.player_id]['name']} (EP={xi.captain.expected_points:.2f})")

    # ---- 6. Freeze the recommendation BEFORE looking at actual results ----
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_files = [
        vl._season_dir(season) / "fixtures.csv",
        vl._season_dir(season) / "teams.csv",
        vl._season_dir(season) / "merged_gw.csv",
    ]
    recommendation = {
        "milestone": "phase3_replay",
        "season": season,
        "target_gameweek": target_gw,
        "generated_at": generated_at,
        "model_config": {
            "team_model": "attack_defense (K_BASE=0.045, HALFLIFE_DAYS=380, not tuned)",
            "scoreline_model": f"dixon_coles (rho={sl.RHO_DEFAULT}, not refit)",
            "minutes_model": (
                f"exponential_decay (half_life_matches={minutes_halflife}, promoted Phase 4b champion)"
                + (f", isotonic-calibrated (Phase 5, fit on {prod_cal.CALIBRATION_FIT_SEASON})" if apply_minutes_calibration else ", UNCALIBRATED")
            ),
            "attacking_model": f"shrinkage_allocation (alpha={attacking_alpha}, lookback={lookback}, promoted Phase 4b champion)",
            "simulation_seed": 2026,
            "simulations_run": total_sims,
        },
        "source_data_hashes": {_display_path(p): _hash_file(p) for p in source_files},
        "fixture_projections": fixture_meta,
        "squad": [
            {"player_id": p.player_id, "name": candidates_meta[p.player_id]["name"], "position": p.position,
             "team": p.team, "price": p.price, "expected_points": round(p.expected_points, 3)}
            for p in squad
        ],
        "starting_xi": [p.player_id for p in xi.starters],
        "bench_order": [p.player_id for p in xi.bench],
        "captain": xi.captain.player_id,
        "vice_captain_fallback": sorted(xi.starters, key=lambda p: -p.expected_points)[1].player_id,
        "projected_gw_points": round(sum(p.expected_points for p in xi.starters) + xi.captain.expected_points, 3),
    }
    rec_path = artifact_dir / f"gw{target_gw:02d}_recommendation.json"
    rec_path.write_text(json.dumps(recommendation, indent=2))
    artifact_hash = hashlib.sha256(rec_path.read_bytes()).hexdigest()

    # ======================================================
    # Evaluation only below this line — recommendation already frozen.
    # ======================================================
    actual_by_player = {r["element"]: r for r in target_rows}

    def realized_points(pid: str) -> int:
        return int(actual_by_player[pid]["total_points"])

    xi_ids = {p.player_id for p in xi.starters}
    realized_total = sum(realized_points(pid) for pid in xi_ids) + realized_points(xi.captain.player_id)

    recent_points_by_player: dict[str, int] = defaultdict(int)
    for r in history_rows:
        if int(r["GW"]) >= lookback_floor:
            recent_points_by_player[r["element"]] += int(r["total_points"])
    baseline_candidates = [
        sq.PlayerCandidate(pid, meta["position"], meta["team"], meta["price"],
                            float(recent_points_by_player.get(pid, 0)))
        for pid, meta in candidates_meta.items()
    ]
    baseline_squad = sq.select_squad(baseline_candidates, budget=sq.BUDGET)
    baseline_xi = sq.select_starting_xi(baseline_squad)
    baseline_ids = {p.player_id for p in baseline_xi.starters}
    baseline_realized = sum(realized_points(pid) for pid in baseline_ids) + realized_points(baseline_xi.captain.player_id)

    evaluation = {
        "artifact_hash": artifact_hash,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_gameweek": target_gw,
        "projected_gw_points": recommendation["projected_gw_points"],
        "model_squad_realized_points": realized_total,
        "baseline_recent_form_realized_points": baseline_realized,
        "difference": realized_total - baseline_realized,
        "captain_name": candidates_meta[xi.captain.player_id]["name"],
        "captain_realized_points": realized_points(xi.captain.player_id),
    }
    eval_path = artifact_dir / f"gw{target_gw:02d}_evaluation.json"
    eval_path.write_text(json.dumps(evaluation, indent=2))
    log(f"  realized {realized_total} vs baseline {baseline_realized} (diff {evaluation['difference']:+d})")

    return GameweekReplayResult(season, target_gw, recommendation, evaluation, rec_path, eval_path)
