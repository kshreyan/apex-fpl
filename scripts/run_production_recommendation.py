#!/usr/bin/env python3
"""Phase 12 production orchestration: generates one gameweek's squad
recommendation from LIVE 2026/27 Bronze/Silver data (never the
historical Vaastav archive, which every research-phase script uses
instead), using ONLY the CONFIRMED champions in artifacts/model_registry.json
— unfit team model, exponential-decay+calibrated minutes, shrinkage
attacking allocation where real history exists, the EV squad optimizer.
None of the NOT-PROMOTED challengers (CVaR, MAD, decision-focused
shrinkage) are used here; that promotion discipline is exactly why they
weren't promoted.

**Cold-start handling.** GW1 of a new season has ZERO gameweeks of
current-season history for every player and zero completed fixtures for
the team model — every prior replay script in this project could assume
that history exists (they all target GW2+ of an already-underway
historical season). This script instead:

- **Team model**: fits on real, already-finished 2026/27 fixtures (none
  exist yet before GW1) PLUS 2024-25's real historical fixtures
  (`apex_fpl.serving.live_data.build_team_model_fixtures`) — the
  existing 380-day half-life decay already down-weights older matches
  appropriately; promoted or newly-formed clubs with no fixture history
  at all fall back to the model's own built-in league-average rating
  (`AttackDefenseModel._decayed`'s `.get(team, 0.0)`), not a crash.
  Known limitation, stated plainly: 2025/26 is NOT YET in this
  project's historical archive, so 2024-25 (already ~15 months stale by
  August 2026) is the freshest available prior — meaningfully worse
  than the fresher prior this mechanism was designed to use once a more
  complete archive exists.
- **Minutes**: uses the Phase 12 cold-start model
  (`apex_fpl.models.minutes.cold_start`, real-data-validated: beats a
  flat baseline on held-out log loss in all 4 leave-one-season-out
  folds checked) rather than the champion `exponential_decay` model,
  which needs history this gameweek structurally cannot have.
- **Attacking allocation**: NO validated cold-start equivalent exists
  yet (would need reliable player-ID reconciliation between the live
  API and the historical archive, not attempted this phase — an
  explicit, stated gap, not a silent guess). Falls back to a simple,
  UNVALIDATED price-weighted split within each team's outfield
  positions — real signal (price correlates with attacking reputation)
  but not tested against real data the way the minutes cold-start model
  was. Every recommendation this script produces is tagged with this
  caveat so it's never silently presented as equally trustworthy.

**Double gameweeks: a stated, partial fix, not a full one.** Properly
simulating two independent fixtures for one player needs the model's own
simulator (apex_fpl.simulation.monte_carlo) to represent multiple
fixtures per player — it currently doesn't, and extending it is a model
change, out of scope for this orchestration layer. What WAS fixable here:
this script used to silently drop a double-gameweek team's second
fixture entirely (a plain dict overwrite, `fixture_expected_goals[(team,
side)] = eh`, clobbered by whichever fixture was processed last — found
while building the automated pipeline, not previously caught). Fixed to
SUM a team's expected goals across all its fixtures that gameweek before
allocating player shares — still an approximation (two independent
matches collapsed into one combined-goals match, losing their separate
correlation structure), but a stated one instead of a silent under-count.
Any gameweek with a double fixture gets an explicit `model_caveats` entry
naming the affected team(s), rather than looking identical to a normal
week.

`write_artifact` defaults to True (this script's own original,
standalone behavior — writes the timestamped, content-hashed artifact
under artifacts/production_recommendations/, matching
`apex_fpl.backtesting.replay.run_gameweek`'s freeze-before-reveal
discipline). The automated pipeline (pipeline/predict.py) calls this
with `write_artifact=False` — it has its own, schema-versioned ledger to
write to (data/predictions/gw{n}.jsonl) and doesn't want a second,
differently-shaped file written alongside it as a side effect.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.models.minutes import cold_start as cs
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.serving import live_data as ld
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "production_recommendations"
COLD_START_MINUTES_TRAIN_SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
TEAM_MODEL_FALLBACK_SEASONS = ("2024-25",)
CAPTAIN_HAUL_THRESHOLD = 6  # "haul" = captain (undoubled) points >= this


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_rows(rows: list[dict]) -> str:
    canonical = json.dumps(rows, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _hash_fixtures(fixtures: list) -> str:
    canonical = json.dumps(
        [[f.date.isoformat(), f.home_team, f.away_team, f.home_score, f.away_score] for f in fixtures],
        sort_keys=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_cold_start_minutes_rows() -> list[dict]:
    rows = []
    for season in COLD_START_MINUTES_TRAIN_SEASONS:
        all_rows = vl.load_merged_gw(season)
        rows += [r for r in all_rows if r["GW"] == "1" and r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]
    return rows


def fit_cold_start_minutes() -> cs.ColdStartMinutesModel:
    return cs.fit_cold_start_minutes_model(load_cold_start_minutes_rows())


def build_fixture_inputs_and_team_goals(target_fixtures: list[dict], team_model) -> tuple[list, list[dict], dict[str, list[float]]]:
    """Extracted from generate_recommendation() specifically so the
    double-gameweek summing behavior (a real bug fix -- this used to be
    a plain dict overwrite, `fixture_expected_goals[(team, side)] = eh`,
    silently dropping a double-gameweek team's second fixture entirely)
    can be tested directly against a lightweight fake team_model, without
    needing to run the full live pipeline against real Silver data just
    to exercise one aggregation path. `team_model` only needs an
    `expected_goals(home_team, away_team, at_date) -> (eh, ea)` method —
    see tests/unit/test_run_production_recommendation.py."""
    fixture_inputs, fixture_meta = [], []
    team_expected_goals: dict[str, list[float]] = {}
    for fx in target_fixtures:
        eh, ea = team_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
        m = sl.score_matrix(eh, ea)
        fixture_inputs.append(mc.FixtureInput(home_team=fx["home_team"], away_team=fx["away_team"], score_matrix=m))
        fixture_meta.append({"home_team": fx["home_team"], "away_team": fx["away_team"], "expected_home_goals": round(eh, 3), "expected_away_goals": round(ea, 3)})
        team_expected_goals.setdefault(fx["home_team"], []).append(eh)
        team_expected_goals.setdefault(fx["away_team"], []).append(ea)
    return fixture_inputs, fixture_meta, team_expected_goals


def generate_recommendation(target_gw: int, verbose: bool = True, write_artifact: bool = True) -> dict:
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"=== Production recommendation: 2026/27 GW{target_gw} ===\n")

    log("--- Fitting team model (live + 2024-25 fallback fixtures) ---")
    team_fixtures = ld.build_team_model_fixtures(fallback_seasons=TEAM_MODEL_FALLBACK_SEASONS)
    team_model = ad.fit(team_fixtures)
    log(f"Team model fit on {len(team_fixtures)} fixtures (earliest {team_fixtures[0].date.date()}, latest {team_fixtures[-1].date.date()})")

    target_fixtures = ld.load_target_gw_fixtures(target_gw)
    if not target_fixtures:
        raise ValueError(f"no live fixtures found for GW{target_gw} — check the Bronze snapshot is current")
    log(f"GW{target_gw}: {len(target_fixtures)} fixtures\n")

    log("--- Fitting cold-start minutes model (real historical GW1 data, 4 seasons) ---")
    cold_start_rows = load_cold_start_minutes_rows()
    minutes_model = cs.fit_cold_start_minutes_model(cold_start_rows)

    log("--- Loading live player roster ---")
    players = ld.load_players()
    log(f"{len(players)} players loaded\n")

    # team_expected_goals sums ALL of a team's fixtures this gameweek (a double-gameweek
    # team plays twice) -- see module docstring's "Double gameweeks" section for why this
    # is a stated approximation (one combined match), not full independent-fixture modeling.
    fixture_inputs, fixture_meta, team_expected_goals = build_fixture_inputs_and_team_goals(target_fixtures, team_model)

    teams_with_double_fixture = sorted(team for team, goals in team_expected_goals.items() if len(goals) > 1)
    if teams_with_double_fixture:
        log(f"Double gameweek detected for: {', '.join(teams_with_double_fixture)} — combining into one summed-goals match (see module docstring)\n")

    # price-weighted attacking share within each (team, outfield-position) group — see module
    # docstring's "Attacking allocation" section for why this is an explicit, unvalidated fallback.
    by_team_pos: dict[tuple[str, str], list[str]] = {}
    for pid, meta in players.items():
        if meta["team"] not in team_expected_goals or meta["position"] == "GK":
            continue
        by_team_pos.setdefault((meta["team"], meta["position"]), []).append(pid)
    price_weight: dict[str, float] = {}
    for (team, pos), ids in by_team_pos.items():
        total_price = sum(players[pid]["price"] for pid in ids)
        for pid in ids:
            price_weight[pid] = players[pid]["price"] / total_price if total_price > 0 else 1.0 / len(ids)

    players_for_sim, candidates_meta = [], {}
    for pid, meta in players.items():
        team = meta["team"]
        if team not in team_expected_goals:
            continue
        mfc = minutes_model.predict(meta["price"])
        team_exp_goals_total = sum(team_expected_goals[team])

        if meta["position"] == "GK":
            exp_goals, exp_assists = 0.0, 0.0
        else:
            share = price_weight.get(pid, 0.0) * 0.7  # ~70% of a team's expected goals distributed among outfielders proportional to price; the rest (own goals, unallocated) isn't assigned to any single player, matching the conservative spirit of the validated shrinkage model's own smoothing
            exp_goals = team_exp_goals_total * share
            exp_assists = team_exp_goals_total * share * 0.8  # assists are slightly less concentrated than goals in real data (spec Part XI); a rough, stated proxy, not fit from data

        players_for_sim.append(mc.PlayerInput(player_id=pid, team=team, position=meta["position"], minutes_forecast=mfc, expected_goals=exp_goals, expected_assists=exp_assists))
        candidates_meta[pid] = {"name": meta["name"], "team": team, "position": meta["position"], "price": meta["price"], "availability_probability": meta["availability_probability"]}

    log(f"Players entering simulation: {len(players_for_sim)}\n")

    log("--- Running Monte Carlo simulation ---")
    rules = scoring.load_scoring_rules("2026_27")
    sim_results = mc.simulate_gameweek(fixture_inputs, players_for_sim, rules, batch=3000, max_sims=60000, tol=0.05)
    total_sims = len(next(iter(sim_results.values())).samples) if sim_results else 0
    log(f"Simulations run: {total_sims}\n")

    log("--- Selecting squad (EV optimizer, the confirmed champion) ---")
    candidates = [sq.PlayerCandidate(pid, m["position"], m["team"], m["price"], sim_results[pid].mean_points, m["availability_probability"]) for pid, m in candidates_meta.items() if pid in sim_results]
    squad = sq.select_squad(candidates, budget=sq.BUDGET)
    xi = sq.select_starting_xi(squad)
    captain_haul_probability = float((sim_results[xi.captain.player_id].samples >= CAPTAIN_HAUL_THRESHOLD).mean())
    log(f"Captain: {candidates_meta[xi.captain.player_id]['name']} (EP={xi.captain.expected_points:.2f}, P({CAPTAIN_HAUL_THRESHOLD}+)={captain_haul_probability:.2f})\n")

    caveats = [
        "Minutes model is a validated cold-start fallback (beats a flat baseline in real leave-one-season-out testing), not the champion model.",
        "Attacking allocation is an UNVALIDATED price-weighted heuristic, not the champion shrinkage model -- treat with real skepticism.",
        "Team model's historical fallback uses 2024-25 (the archive's most recent season) -- 2025/26 is not yet in this project's historical archive, so this prior is staler than the mechanism was designed for.",
    ]
    if teams_with_double_fixture:
        caveats.append(
            f"Double gameweek this week for: {', '.join(teams_with_double_fixture)}. "
            "Modeled as one combined-goals match per team, not two independent fixtures with their own "
            "correlation structure -- the model's simulator does not yet support multiple fixtures per "
            "player per gameweek (a model-layer limitation, not fixed here)."
        )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recommendation = {
        "system": "apex_fpl_production_v1",
        "target_gameweek": target_gw, "season": "2026/27",
        "generated_at": generated_at,
        "model_config": {
            "team_model": f"attack_defense, fit on {len(team_fixtures)} fixtures (live + {TEAM_MODEL_FALLBACK_SEASONS} fallback)",
            "minutes_model": "COLD-START price-based isotonic model (apex_fpl.models.minutes.cold_start), NOT the champion exponential_decay model -- no current-season history exists yet",
            "attacking_model": "COLD-START price-weighted split within team/position -- UNVALIDATED fallback, NOT the champion shrinkage_share model",
            "squad_optimizer": "select_squad (EV MILP) -- the confirmed champion per artifacts/model_registry.json",
            "simulations_run": total_sims,
        },
        "caveats": caveats,
        "teams_with_double_fixture": teams_with_double_fixture,
        "training_data_fingerprint": {
            "cold_start_minutes_seasons": COLD_START_MINUTES_TRAIN_SEASONS,
            "cold_start_minutes_hash": _hash_rows(cold_start_rows),
            "team_model_fallback_seasons": list(TEAM_MODEL_FALLBACK_SEASONS),
            "team_model_fallback_hash": _hash_fixtures(team_fixtures),
        },
        "fixture_projections": fixture_meta,
        "squad": [{"player_id": p.player_id, "name": candidates_meta[p.player_id]["name"], "position": p.position, "team": p.team, "price": p.price, "expected_points": round(p.expected_points, 3)} for p in squad],
        "starting_xi": [p.player_id for p in xi.starters],
        "bench_order": [p.player_id for p in xi.bench],
        "captain": xi.captain.player_id,
        "vice_captain_fallback": sorted(xi.starters, key=lambda p: -p.expected_points)[1].player_id,
        "captain_haul_probability": round(captain_haul_probability, 4),
        "captain_haul_threshold": CAPTAIN_HAUL_THRESHOLD,
        "projected_gw_points": round(sum(p.expected_points for p in xi.starters) + xi.captain.expected_points, 3),
    }

    if write_artifact:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        rec_path = ARTIFACT_DIR / f"gw{target_gw:02d}_recommendation_{generated_at.replace(':', '').replace('+00:00', 'Z')}.json"
        rec_path.write_text(json.dumps(recommendation, indent=2))
        recommendation["artifact_hash"] = _hash_file(rec_path)
        recommendation["artifact_path"] = str(rec_path.relative_to(REPO_ROOT))
        log(f"Frozen to {rec_path.relative_to(REPO_ROOT)} (sha256={recommendation['artifact_hash'][:12]}...)")

    log(f"Projected points: {recommendation['projected_gw_points']}")
    return recommendation


if __name__ == "__main__":
    import sys

    target_gw = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    generate_recommendation(target_gw)
