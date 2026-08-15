"""Reusable assembly of simulation-ready player/fixture data for one real
historical gameweek — the shared core behind both `replay.run_gameweek()`
(which only exposes the final squad recommendation) and any script that
needs the underlying `PlayerSimResult` objects directly, e.g. for CVaR
scenario optimization or captaincy risk analysis
(scripts/run_robust_captaincy_demo.py, scripts/run_cvar_multi_gw_replay.py).

Consolidated here (rather than duplicated per-script) specifically because
duplicating it already produced a real bug once: an earlier standalone
copy of this logic in run_robust_captaincy_demo.py omitted the
Manager-pseudo-player position filter and blank-gameweek guard that
replay.py had already needed fixing (2024-25's "AM" rows, 2022-23 GW7).
One shared implementation means one fix benefits every caller.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.backtesting.replay import BlankGameweekError
from apex_fpl.calibration import production_calibrators as prod_cal
from apex_fpl.models.attacking import challengers as attacking_challengers
from apex_fpl.models.attacking import proportional as prop
from apex_fpl.models.minutes import challengers as minutes_challengers
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl
from apex_fpl.simulation import monte_carlo as mc

CLASSIC_POSITIONS = ("GK", "GKP", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class GameweekScenarioData:
    fixture_inputs: list[mc.FixtureInput]
    players_for_sim: list[mc.PlayerInput]
    candidates_meta: dict[str, dict]  # player_id -> {name, team, position, price}
    target_rows: list[dict]  # raw merged_gw rows for the target GW (for revealing actual outcomes)
    shares: dict[str, prop.AttackingShare]  # player_id -> RAW (goal_share, assist_share), pre-multiplication — needed by the joint simulator's multinomial allocation, which players_for_sim's pre-multiplied expected_goals/expected_assists don't preserve


def build_gameweek_scenario_data(
    season: str,
    target_gw: int,
    lookback: int = 15,
    minutes_halflife: float = 3.0,
    attacking_alpha: float = 10.0,
    apply_minutes_calibration: bool = True,
) -> GameweekScenarioData:
    training_fixtures = vl.fixtures_before_gw(season, target_gw)
    if not training_fixtures:
        raise ValueError(f"no training fixtures before GW{target_gw} in {season}")
    team_model = ad.fit(training_fixtures)
    target_fixtures = vl.fixtures_at_gw(season, target_gw)
    if not target_fixtures:
        raise BlankGameweekError(f"{season} GW{target_gw} has zero fixtures — blank gameweek")

    fixture_inputs, fixture_meta = [], []
    for fx in target_fixtures:
        eh, ea = team_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
        m = sl.score_matrix(eh, ea)
        fixture_inputs.append(mc.FixtureInput(home_team=fx["home_team"], away_team=fx["away_team"], score_matrix=m))
        fixture_meta.append({"home_team": fx["home_team"], "away_team": fx["away_team"], "eh": eh, "ea": ea})

    all_rows = vl.load_merged_gw(season)
    # Exclude the FPL "Manager" pseudo-player mechanic (2024-25 GW23+,
    # position="AM") — not part of the classic 15-player squad. See
    # docs/phase3_extended_replay_report.md for how this bug first surfaced.
    all_rows = [r for r in all_rows if r.get("position") in CLASSIC_POSITIONS]
    history_rows = [r for r in all_rows if int(r["GW"]) < target_gw]
    target_rows = [r for r in all_rows if int(r["GW"]) == target_gw]

    minutes_by_player: dict[str, list[int]] = defaultdict(list)
    for r in sorted(history_rows, key=lambda r: int(r["GW"])):
        minutes_by_player[r["element"]].append(int(r["minutes"]))

    lookback_floor = target_gw - lookback
    team_player_events: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for r in history_rows:
        if int(r["GW"]) < lookback_floor:
            continue
        team_player_events[r["team"]][r["element"]].append((int(r["goals_scored"]), int(r["assists"])))

    team_side, fixture_expected_goals = {}, {}
    for f in fixture_meta:
        team_side[f["home_team"]] = "home"
        team_side[f["away_team"]] = "away"
        fixture_expected_goals[(f["home_team"], "home")] = f["eh"]
        fixture_expected_goals[(f["away_team"], "away")] = f["ea"]

    shares_by_team = {
        team: attacking_challengers.shrinkage_share(dict(hist), alpha=attacking_alpha)
        for team, hist in team_player_events.items()
    }

    roster = {r["element"]: r for r in target_rows}
    players_for_sim, candidates_meta, player_shares = [], {}, {}
    for pid, row in roster.items():
        team = row["team"]
        if team not in team_side:
            continue
        side = team_side[team]
        hist_minutes = minutes_by_player.get(pid, [])
        mfc = minutes_challengers.exponential_decay(hist_minutes, half_life_matches=minutes_halflife)
        if apply_minutes_calibration:
            mfc = prod_cal.apply_minutes_calibration(mfc)
        shares = shares_by_team.get(team, {})
        share = shares.get(pid, prop.AttackingShare(0.0, 0.0))
        player_shares[pid] = share
        team_exp_goals = fixture_expected_goals[(team, side)]
        players_for_sim.append(mc.PlayerInput(
            player_id=pid, team=team, position=row["position"], minutes_forecast=mfc,
            expected_goals=team_exp_goals * share.goal_share, expected_assists=team_exp_goals * share.assist_share,
        ))
        candidates_meta[pid] = {"name": row["name"], "team": team, "position": row["position"], "price": int(row["value"]) / 10.0}

    return GameweekScenarioData(fixture_inputs, players_for_sim, candidates_meta, target_rows, player_shares)
