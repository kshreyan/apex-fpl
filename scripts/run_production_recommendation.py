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
- **Minutes and attacking allocation, GW1-6**: use the cold-start models
  (`apex_fpl.models.minutes.cold_start`, real-data-validated: beats a
  flat baseline on held-out log loss in all 4 leave-one-season-out
  folds checked; and an UNVALIDATED price-weighted split for attacking
  allocation, since no validated cold-start equivalent for it exists —
  real signal, price correlates with attacking reputation, but not
  tested against real data the way the minutes cold-start model was).
  Both champion models (`exponential_decay`, `shrinkage_share`)
  structurally need real per-gameweek current-season history neither
  can have this early.
- **Minutes and attacking allocation, GW7+**: switches automatically to
  the Phase 4b/5 champion models
  (`apex_fpl.models.minutes.challengers.exponential_decay`, isotonic-
  calibrated per Phase 5; `apex_fpl.models.attacking.challengers.
  shrinkage_share`), fed by real per-gameweek history reconstructed from
  this pipeline's OWN captured raw snapshots
  (`apex_fpl.serving.gameweek_history` — no `element-summary` API calls,
  no historical-archive cross-reconciliation; diffs cumulative totals at
  consecutive settled-gameweek boundaries instead). The GW7 threshold
  (`IN_SEASON_TRANSITION_MIN_SETTLED_GWS = 6` prior settled gameweeks)
  matches Phase 4b's own tournament evidence boundary — the earliest
  gameweek that tournament had ANY real evidence for either champion —
  not an independently re-validated optimal switchover point for this
  specific handoff; that remains real, undone future work. If a
  pipeline outage leaves a real gap in the reconstructed history, the
  affected gameweek(s) are skipped rather than guessed at (see
  `gameweek_history`'s own docstring), so the transition can be delayed
  past GW7 in practice, or an individual player can fall back to a
  model's own uninformative-prior behavior even once `in_season_mode`
  is otherwise active. Every recommendation is tagged with which mode
  actually ran (`in_season_mode`, `settled_gameweeks_reconstructed` in
  the output) rather than presenting either path as the silent default.

**Double gameweeks (Phase 13 Block 2.6).** `apex_fpl.simulation.
monte_carlo.simulate_gameweek` now genuinely simulates a double-
gameweek player's TWO independent fixtures, not one: appearance points,
clean sheet, and the goals-conceded penalty are each drawn and scored
PER fixture and summed (matching real FPL double-gameweek scoring
exactly), using that same fixture's own simulated scoreline for each.
Found and fixed alongside this script's own earlier team-goals bug (a
plain dict overwrite that silently dropped a double-gameweek team's
SECOND fixture from the goal-allocation total) — the simulator had the
identical bug one layer deeper: `team_fixture[team] = (fx, side)`
silently overwrote to the last-processed fixture, so a double-gameweek
player's minutes/clean-sheet/conceded were previously computed against
only ONE of their two matches, not an approximation but an outright
miss (never triggered live yet, since no real double gameweek has hit
this simulator until now). The one remaining, honestly-scoped
simplification: goals/assists are POOLED into a single Poisson draw
across BOTH fixtures (using the combined expected_goals/expected_assists
this script already sums here, `team_exp_goals_total`), credited once,
rather than split per fixture — there is no principled way to attribute
a combined goal-involvement rate to one specific match without a fuller
joint model than this baseline simulator implements (Phase 6 territory,
not this).
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
from apex_fpl.calibration import production_calibrators as prod_cal
from apex_fpl.models.attacking import challengers as attacking_challengers
from apex_fpl.models.attacking import proportional as prop
from apex_fpl.models.minutes import challengers as minutes_challengers
from apex_fpl.models.minutes import cold_start as cs
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.serving import gameweek_history as gwh
from apex_fpl.serving import live_data as ld
from apex_fpl.simulation import monte_carlo as mc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "production_recommendations"
COLD_START_MINUTES_TRAIN_SEASONS = ["2020-21", "2022-23", "2023-24", "2024-25"]
TEAM_MODEL_FALLBACK_SEASONS = ("2024-25",)
CAPTAIN_HAUL_THRESHOLD = 6  # "haul" = captain (undoubled) points >= this

# In-season champion config -- must match artifacts/model_registry.json's
# minutes_model/attacking_allocation_model champion entries exactly; not
# re-derived here, just referenced (Phase 4b/5's own validated values).
MINUTES_HALFLIFE = 3.0
ATTACKING_ALPHA = 10.0
ATTACKING_LOOKBACK = 15

# The earliest gameweek Phase 4b's own tournament had ANY real evidence
# for these champions: GW7, the first gameweek where even the simplest
# candidate (flat_lookback6) has a full 6-gameweek window
# (docs/phase4b_tournament_report.md's "Setup"). Using 6 PRIOR settled
# gameweeks as the transition threshold matches that evidence boundary
# exactly -- it is NOT an independently re-validated optimal switchover
# point for this specific cold-start-to-in-season transition (that is
# real, undone future work; see the promotion schedule's minutes item).
# Counted as "gameweeks this pipeline actually reconstructed a usable
# settlement snapshot for" (apex_fpl.serving.gameweek_history), not just
# target_gw - 1, since a pipeline outage can leave real gaps.
IN_SEASON_TRANSITION_MIN_SETTLED_GWS = 6


def usable_settled_gameweek_count(target_gw: int) -> int:
    # raw_root passed explicitly (not left to gwh.find_settlement_snapshots's
    # own default parameter) so a test monkeypatching gwh.RAW_DATA_ROOT is
    # actually picked up -- a default arg is bound at gwh's import time, before
    # any monkeypatch runs.
    return len(gwh.find_settlement_snapshots(max_gw=target_gw - 1, raw_root=gwh.RAW_DATA_ROOT))


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

    settled_gw_count = usable_settled_gameweek_count(target_gw)
    in_season_mode = settled_gw_count >= IN_SEASON_TRANSITION_MIN_SETTLED_GWS
    minutes_history: dict[str, list[int]] = {}
    goals_assists_history: dict[str, list[tuple[int, int]]] = {}
    if in_season_mode:
        log(f"--- {settled_gw_count} settled gameweek(s) reconstructed -- using in-season champion models (exponential_decay minutes, shrinkage attacking) ---\n")
        deltas = gwh.reconstruct_player_gameweek_deltas(max_gw=target_gw - 1, raw_root=gwh.RAW_DATA_ROOT)
        minutes_history = gwh.minutes_history_by_code(deltas)
        goals_assists_history = gwh.goals_assists_history_by_code(deltas)
    else:
        log(f"--- Only {settled_gw_count} settled gameweek(s) reconstructed (< {IN_SEASON_TRANSITION_MIN_SETTLED_GWS}) -- staying on cold-start models ---\n")

    # team_expected_goals sums ALL of a team's fixtures this gameweek -- used
    # to compute each player's POOLED expected_goals/expected_assists (see
    # module docstring's "Double gameweeks" section); the simulator itself
    # (apex_fpl.simulation.monte_carlo) now scores appearance/clean-sheet/
    # conceded per fixture correctly, using fixture_inputs below, which
    # already carries one entry per real fixture, not a combined one.
    fixture_inputs, fixture_meta, team_expected_goals = build_fixture_inputs_and_team_goals(target_fixtures, team_model)

    teams_with_double_fixture = sorted(team for team, goals in team_expected_goals.items() if len(goals) > 1)
    if teams_with_double_fixture:
        log(f"Double gameweek detected for: {', '.join(teams_with_double_fixture)} — both fixtures simulated independently, goals/assists pooled across them (see module docstring)\n")

    if in_season_mode:
        # Grouped by team only, ALL positions included (matches the validated
        # apex_fpl.backtesting.replay.run_gameweek pattern exactly -- goalkeepers
        # are not specially zeroed, shrinkage's own alpha naturally gives them a
        # near-zero share from their real near-zero observed goal involvement).
        team_player_history: dict[str, dict[str, list[tuple[int, int]]]] = {}
        for pid, meta in players.items():
            if meta["team"] not in team_expected_goals:
                continue
            hist = goals_assists_history.get(meta["code"], [])[-ATTACKING_LOOKBACK:]
            team_player_history.setdefault(meta["team"], {})[pid] = hist
        shares_by_team = {team: attacking_challengers.shrinkage_share(hist, alpha=ATTACKING_ALPHA) for team, hist in team_player_history.items()}
    else:
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
        team_exp_goals_total = sum(team_expected_goals[team])

        if in_season_mode:
            mfc = minutes_challengers.exponential_decay(minutes_history.get(meta["code"], []), half_life_matches=MINUTES_HALFLIFE)
            mfc = prod_cal.apply_minutes_calibration(mfc)
            share = shares_by_team.get(team, {}).get(pid, prop.AttackingShare(0.0, 0.0))
            exp_goals, exp_assists = prop.allocate(team_exp_goals_total, {pid: share})[pid]
        else:
            mfc = minutes_model.predict(meta["price"])
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

    if in_season_mode:
        caveats = [
            f"Minutes and attacking allocation use the Phase 4b/5 champion models (exponential_decay, calibrated; shrinkage_share), fed by {settled_gw_count} settled gameweek(s) of history reconstructed from this pipeline's own raw snapshots (apex_fpl.serving.gameweek_history) -- not the historical Vaastav archive.",
            f"The GW{IN_SEASON_TRANSITION_MIN_SETTLED_GWS + 1} switchover point matches Phase 4b's own tournament evidence boundary, not an independently re-validated optimal transition for this specific cold-start handoff -- a dedicated replay study of the exact best switchover point has not been done.",
            "Reconstructed history can have real gaps (a pipeline outage skips the affected gameweek(s) rather than guessing) -- individual players with little or no reconstructed history fall back to each model's own uninformative-prior behavior, not a crash.",
        ]
    else:
        caveats = [
            "Minutes model is a validated cold-start fallback (beats a flat baseline in real leave-one-season-out testing), not the champion model.",
            "Attacking allocation is an UNVALIDATED price-weighted heuristic, not the champion shrinkage model -- treat with real skepticism.",
        ]
    caveats.append(
        "Team model's historical fallback uses 2024-25 (the archive's most recent season) -- 2025/26 is not yet in this project's historical archive, so this prior is staler than the mechanism was designed for."
    )
    if teams_with_double_fixture:
        caveats.append(
            f"Double gameweek this week for: {', '.join(teams_with_double_fixture)}. "
            "Appearance points, clean sheet, and the goals-conceded penalty are simulated independently "
            "per fixture and summed, matching real FPL double-gameweek scoring. Goals/assists are the "
            "one remaining simplification: pooled into a single combined-rate draw across both fixtures "
            "rather than split per match, since there's no principled way to attribute a combined "
            "goal-involvement rate to one specific fixture without a fuller joint model than this "
            "baseline simulator implements."
        )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recommendation = {
        "system": "apex_fpl_production_v1",
        "target_gameweek": target_gw, "season": "2026/27",
        "generated_at": generated_at,
        "model_config": {
            "team_model": f"attack_defense, fit on {len(team_fixtures)} fixtures (live + {TEAM_MODEL_FALLBACK_SEASONS} fallback)",
            "minutes_model": (
                f"IN-SEASON champion: exponential_decay (half_life={MINUTES_HALFLIFE}, isotonic-calibrated per Phase 5), fed by {settled_gw_count} reconstructed settled gameweek(s)"
                if in_season_mode else
                "COLD-START price-based isotonic model (apex_fpl.models.minutes.cold_start), NOT the champion exponential_decay model -- no current-season history exists yet"
            ),
            "attacking_model": (
                f"IN-SEASON champion: shrinkage_share (alpha={ATTACKING_ALPHA}, lookback={ATTACKING_LOOKBACK}), fed by {settled_gw_count} reconstructed settled gameweek(s)"
                if in_season_mode else
                "COLD-START price-weighted split within team/position -- UNVALIDATED fallback, NOT the champion shrinkage_share model"
            ),
            "squad_optimizer": "select_squad (EV MILP) -- the confirmed champion per artifacts/model_registry.json",
            "simulations_run": total_sims,
        },
        "in_season_mode": in_season_mode,
        "settled_gameweeks_reconstructed": settled_gw_count,
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
