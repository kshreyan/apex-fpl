#!/usr/bin/env python3
"""Phase 4b model tournament: minutes model + attacking-allocation model
(spec Parts VIII, X, XXXVI, XXXVII).

Motivated directly by docs/phase4_tournament_report.md's conclusion: the
team model is validated and not the bottleneck behind Phase 3's flat
FPL-points result, so the minutes model and the attacking-allocation
model (explicitly named in spec Part X as something to move past) are the
next things to actually test against challengers.

Nested split: 2021-22 is used ONLY for inner hyperparameter tuning
(minutes lookback/half-life, attacking lookback/shrinkage alpha) —
2022-23 and 2023-24 are the true held-out outer test seasons, walked
forward gameweek-by-gameweek (GW7-38, same lookback-buffer convention as
Phase 3) exactly like the replay framework: history strictly before each
gameweek, never that gameweek's own results.

Team-model expected goals (needed to turn attacking SHARES into actual
allocated expected-goals) use the Phase 4-validated champion team model
(default, unfit-but-validated constants) fit on the full 6-season fixture
history available (2019-20..2024-25), refit cumulatively per date exactly
as in Phase 3/4.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.evaluation import metrics as em
from apex_fpl.models.attacking import challengers as ac
from apex_fpl.models.attacking import proportional as prop
from apex_fpl.models.minutes import challengers as mcx
from apex_fpl.models.minutes.baseline import forecast_minutes
from apex_fpl.models.teams import attack_defense as ad

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase4b_tournament"

ALL_FIXTURE_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
INNER_TUNE_SEASON = "2021-22"
OUTER_TEST_SEASONS = ["2022-23", "2023-24"]
START_GW = 7
END_GW = 38
MAX_LOOKBACK_BUFFER = 15  # must be >= the largest lookback in LOOKBACK_GRID
LOOKBACK_GRID = [3, 4, 6, 8, 10, 15]
HALFLIFE_GRID = [1.0, 2.0, 3.0, 5.0, 8.0]
ALPHA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
CHAMPION_LOOKBACK = 6


def load_all_fixture_rows():
    return sorted((r for s in ALL_FIXTURE_SEASONS for r in vl.load_fixtures(s)), key=lambda r: r["date"])


def team_expected_goals(all_fixture_rows, home_team: str, away_team: str, before_date) -> tuple[float, float]:
    history = [r for r in all_fixture_rows if r["date"] < before_date and r["home_score"] is not None]
    fixtures = [ad.Fixture(r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"]) for r in history]
    model = ad.fit(fixtures)
    return model.expected_goals(home_team, away_team, before_date)


def collect_gw_data(season: str, target_gw: int, all_fixture_rows):
    """Returns (records, team_player_events) for one gameweek, or None if
    it's a blank gameweek. team_player_events[team][player_id] is a list
    of (gw, goals, assists) tuples within MAX_LOOKBACK_BUFFER gameweeks —
    kept at per-gameweek granularity so different candidates can apply
    different lookback windows to the SAME underlying data."""
    all_rows = vl.load_merged_gw(season)
    history_rows = [r for r in all_rows if int(r["GW"]) < target_gw]
    target_rows = [r for r in all_rows if int(r["GW"]) == target_gw]
    if not target_rows:
        return None

    fixtures_this_gw = vl.fixtures_at_gw(season, target_gw)
    if not fixtures_this_gw:
        return None  # blank gameweek

    team_exp_goals: dict[str, float] = {}
    for fx in fixtures_this_gw:
        eh, ea = team_expected_goals(all_fixture_rows, fx["home_team"], fx["away_team"], fx["date"])
        team_exp_goals[fx["home_team"]] = eh
        team_exp_goals[fx["away_team"]] = ea

    minutes_by_player: dict[str, list[int]] = defaultdict(list)
    for r in sorted(history_rows, key=lambda r: int(r["GW"])):
        minutes_by_player[r["element"]].append(int(r["minutes"]))

    lookback_floor = target_gw - MAX_LOOKBACK_BUFFER
    team_player_events: dict[str, dict[str, list[tuple[int, int, int]]]] = defaultdict(lambda: defaultdict(list))
    for r in history_rows:
        gw = int(r["GW"])
        if gw < lookback_floor:
            continue
        team_player_events[r["team"]][r["element"]].append((gw, int(r["goals_scored"]), int(r["assists"])))

    records = []
    for r in target_rows:
        team = r["team"]
        if team not in team_exp_goals:
            continue
        pid = r["element"]
        records.append({
            "player_id": pid, "team": team, "position": r["position"], "target_gw": target_gw, "season": season,
            "historical_minutes": list(minutes_by_player.get(pid, [])),
            "team_expected_goals": team_exp_goals[team],
            "actual_minutes": int(r["minutes"]),
            "actual_goals": int(r["goals_scored"]),
            "actual_assists": int(r["assists"]),
        })
    return records, dict(team_player_events)


def evaluate_minutes_candidates(all_records, lookback_tuned: int, halflife_tuned: float):
    candidates = {
        "champion_flat_lookback6": lambda h: forecast_minutes(h, lookback=CHAMPION_LOOKBACK),
        "challenger_flat_tuned": lambda h: forecast_minutes(h, lookback=lookback_tuned),
        "challenger_exp_decay": lambda h: mcx.exponential_decay(h, half_life_matches=halflife_tuned),
        "baseline_always_90": mcx.always_90,
        "baseline_persistence": mcx.persistence,
    }
    results = {}
    for name, fn in candidates.items():
        p60, y60, papp, yapp = [], [], [], []
        for rec in all_records:
            fc = fn(rec["historical_minutes"])
            p60.append(fc.p_60_plus)
            y60.append(1 if rec["actual_minutes"] >= 60 else 0)
            papp.append(fc.p_appearance)
            yapp.append(1 if rec["actual_minutes"] > 0 else 0)
        results[name] = {
            "p60_metrics": em.full_binary_metrics(np.array(p60), y60),
            "appearance_metrics": em.full_binary_metrics(np.array(papp), yapp),
        }
    return results


def _windowed_history(team_events: dict[str, list[tuple[int, int, int]]], target_gw: int, lookback: int) -> dict[str, list[tuple[int, int]]]:
    floor = target_gw - lookback
    return {
        pid: [(g, a) for gw, g, a in events if gw >= floor]
        for pid, events in team_events.items()
    }


def evaluate_attacking_candidates(records_by_gw, lookback_tuned: int, alpha_tuned: float):
    """records_by_gw: list of (season, gw, records, team_player_events).
    Also returns per_record: a list of {season, gw, actual_goals,
    pred_champion, pred_shrinkage} for the block-bootstrap significance
    test in main() — avoids recomputing the share logic a second time."""
    candidate_names = ["champion_proportional_lookback6", "challenger_proportional_tuned",
                        "challenger_shrinkage", "baseline_equal_split"]
    goal_preds = {n: [] for n in candidate_names}
    goal_actuals = {n: [] for n in candidate_names}
    assist_preds = {n: [] for n in candidate_names}
    assist_actuals = {n: [] for n in candidate_names}
    per_record = []

    for season, gw, records, team_player_events in records_by_gw:
        by_team: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            by_team[rec["team"]].append(rec)

        for team, team_records in by_team.items():
            team_exp_goals = team_records[0]["team_expected_goals"]
            team_events = team_player_events.get(team, {})
            # Ensure every player who's ON the roster this GW (even with zero
            # recorded events) appears in the history dict, so shares are
            # computed over the true squad, not just scorers.
            for rec in team_records:
                team_events.setdefault(rec["player_id"], [])

            hist_champion = _windowed_history(team_events, gw, CHAMPION_LOOKBACK)
            hist_tuned = _windowed_history(team_events, gw, lookback_tuned)

            shares_champion = prop.compute_shares(hist_champion)
            shares_tuned = prop.compute_shares(hist_tuned)
            shares_shrinkage = ac.shrinkage_share(hist_tuned, alpha=alpha_tuned)
            shares_equal = ac.equal_split(list(hist_champion.keys()))

            for rec in team_records:
                pid = rec["player_id"]
                for name, shares in [
                    ("champion_proportional_lookback6", shares_champion),
                    ("challenger_proportional_tuned", shares_tuned),
                    ("challenger_shrinkage", shares_shrinkage),
                    ("baseline_equal_split", shares_equal),
                ]:
                    s = shares.get(pid, prop.AttackingShare(0.0, 0.0))
                    goal_preds[name].append(team_exp_goals * s.goal_share)
                    goal_actuals[name].append(rec["actual_goals"])
                    assist_preds[name].append(team_exp_goals * s.assist_share)
                    assist_actuals[name].append(rec["actual_assists"])

                per_record.append({
                    "season": season, "gw": gw, "actual_goals": rec["actual_goals"],
                    "pred_champion": team_exp_goals * shares_champion.get(pid, prop.AttackingShare(0.0, 0.0)).goal_share,
                    "pred_shrinkage": team_exp_goals * shares_shrinkage.get(pid, prop.AttackingShare(0.0, 0.0)).goal_share,
                })

    results = {}
    for name in candidate_names:
        results[name] = {
            "goal_nll": round(em.poisson_nll(np.array(goal_preds[name]), goal_actuals[name]), 4),
            "goal_mae": round(float(np.mean(np.abs(np.array(goal_preds[name]) - np.array(goal_actuals[name])))), 4),
            "assist_nll": round(em.poisson_nll(np.array(assist_preds[name]), assist_actuals[name]), 4),
            "assist_mae": round(float(np.mean(np.abs(np.array(assist_preds[name]) - np.array(assist_actuals[name])))), 4),
            "n": len(goal_preds[name]),
        }
    return results, per_record


N_BOOTSTRAP = 5000
SEED = 2026


def block_bootstrap_minutes(outer_records, lookback_champion: int, halflife_challenger: float):
    """Block bootstrap (block = one season's one gameweek) on the per-record
    binary log-loss difference (challenger_exp_decay - champion), per spec
    Part XXXIX."""
    blocks: dict[tuple, list[float]] = defaultdict(list)
    for r in outer_records:
        y = 1.0 if r["actual_minutes"] >= 60 else 0.0
        p_champ = forecast_minutes(r["historical_minutes"], lookback=lookback_champion).p_60_plus
        p_chal = mcx.exponential_decay(r["historical_minutes"], half_life_matches=halflife_challenger).p_60_plus
        ll_champ = em.log_loss_binary(np.array([p_champ]), [y])
        ll_chal = em.log_loss_binary(np.array([p_chal]), [y])
        blocks[(r["season"], r["target_gw"])].append(ll_chal - ll_champ)
    return _bootstrap_ci(blocks)


def block_bootstrap_attacking(per_record):
    """Block bootstrap on the per-record Poisson-NLL difference
    (challenger_shrinkage - champion), block = one season's one gameweek."""
    blocks: dict[tuple, list[float]] = defaultdict(list)
    for r in per_record:
        from scipy.stats import poisson
        nll_champ = -poisson.logpmf(r["actual_goals"], max(r["pred_champion"], 1e-9))
        nll_chal = -poisson.logpmf(r["actual_goals"], max(r["pred_shrinkage"], 1e-9))
        blocks[(r["season"], r["gw"])].append(float(nll_chal - nll_champ))
    return _bootstrap_ci(blocks)


def _bootstrap_ci(blocks: dict[tuple, list[float]]) -> tuple[float, float, float]:
    block_means = np.array([np.mean(v) for v in blocks.values()])
    rng = np.random.default_rng(SEED)
    n = len(block_means)
    boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        boot[i] = rng.choice(block_means, size=n, replace=True).mean()
    return float(block_means.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _collect_season(season: str, all_fixture_rows) -> list[tuple[str, int, list, dict]]:
    out = []
    for gw in range(START_GW, END_GW + 1):
        result = collect_gw_data(season, gw, all_fixture_rows)
        if result is None:
            continue
        records, team_player_events = result
        out.append((season, gw, records, team_player_events))
    return out


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 4b tournament: minutes + attacking-allocation models ===")
    all_fixture_rows = load_all_fixture_rows()

    print(f"\n--- Inner tuning on {INNER_TUNE_SEASON} (never used for the outer test) ---")
    tune_by_gw = _collect_season(INNER_TUNE_SEASON, all_fixture_rows)
    tune_records = [rec for _, _, records, _ in tune_by_gw for rec in records]
    print(f"  {len(tune_records)} player-gameweek records collected for tuning")

    best_lookback, best_ll = CHAMPION_LOOKBACK, None
    for L in LOOKBACK_GRID:
        p60 = [forecast_minutes(r["historical_minutes"], lookback=L).p_60_plus for r in tune_records]
        y60 = [1 if r["actual_minutes"] >= 60 else 0 for r in tune_records]
        ll = em.log_loss_binary(np.array(p60), y60)
        if best_ll is None or ll < best_ll:
            best_ll, best_lookback = ll, L
    print(f"  minutes lookback tuned: {best_lookback} (log_loss={best_ll:.4f})")

    best_halflife, best_hl_ll = 3.0, None
    for H in HALFLIFE_GRID:
        p60 = [mcx.exponential_decay(r["historical_minutes"], half_life_matches=H).p_60_plus for r in tune_records]
        y60 = [1 if r["actual_minutes"] >= 60 else 0 for r in tune_records]
        ll = em.log_loss_binary(np.array(p60), y60)
        if best_hl_ll is None or ll < best_hl_ll:
            best_hl_ll, best_halflife = ll, H
    print(f"  minutes exp-decay half-life tuned: {best_halflife} (log_loss={best_hl_ll:.4f})")

    best_alpha, best_alpha_nll = 3.0, None
    for A in ALPHA_GRID:
        res, _ = evaluate_attacking_candidates(tune_by_gw, lookback_tuned=best_lookback, alpha_tuned=A)
        nll = res["challenger_shrinkage"]["goal_nll"]
        if best_alpha_nll is None or nll < best_alpha_nll:
            best_alpha_nll, best_alpha = nll, A
    print(f"  shrinkage alpha tuned: {best_alpha} (goal_nll={best_alpha_nll:.4f})")

    print(f"\n--- Outer test on {OUTER_TEST_SEASONS} ---")
    outer_by_gw = []
    for season in OUTER_TEST_SEASONS:
        outer_by_gw.extend(_collect_season(season, all_fixture_rows))
    outer_records = [rec for _, _, records, _ in outer_by_gw for rec in records]
    print(f"  {len(outer_records)} player-gameweek records collected for outer evaluation")

    minutes_results = evaluate_minutes_candidates(outer_records, best_lookback, best_halflife)
    print("\nMinutes model results (P(60+) binary metrics):")
    for name, r in minutes_results.items():
        m = r["p60_metrics"]
        print(f"  {name:<28} log_loss={m['log_loss']}  brier={m['brier']}  ece={m['ece']}  "
              f"pred_mean={m['mean_predicted']}  obs_mean={m['mean_observed']}")

    attacking_results, attacking_per_record = evaluate_attacking_candidates(outer_by_gw, best_lookback, best_alpha)
    print("\nAttacking-allocation model results (goals Poisson NLL):")
    for name, r in attacking_results.items():
        print(f"  {name:<28} goal_nll={r['goal_nll']}  goal_mae={r['goal_mae']}  "
              f"assist_nll={r['assist_nll']}  assist_mae={r['assist_mae']}  n={r['n']}")

    print("\n--- Significance testing (block bootstrap, block=season+gameweek) ---")
    m_mean, m_lo, m_hi = block_bootstrap_minutes(outer_records, CHAMPION_LOOKBACK, best_halflife)
    print(f"Minutes: exp_decay - champion log-loss diff = {m_mean:+.4f}  95% CI [{m_lo:+.4f}, {m_hi:+.4f}]")
    minutes_promote = m_hi < 0

    a_mean, a_lo, a_hi = block_bootstrap_attacking(attacking_per_record)
    print(f"Attacking: shrinkage - champion goal-NLL diff = {a_mean:+.4f}  95% CI [{a_lo:+.4f}, {a_hi:+.4f}]")
    attacking_promote = a_hi < 0

    print(f"\nMinutes promotion decision: {'PROMOTE challenger_exp_decay' if minutes_promote else 'do not promote'}")
    print(f"Attacking promotion decision: {'PROMOTE challenger_shrinkage' if attacking_promote else 'do not promote'}")

    summary = {
        "inner_tune_season": INNER_TUNE_SEASON,
        "outer_test_seasons": OUTER_TEST_SEASONS,
        "tuned_lookback": best_lookback, "tuned_halflife": best_halflife, "tuned_alpha": best_alpha,
        "minutes_results": minutes_results,
        "attacking_results": attacking_results,
        "minutes_bootstrap": {"mean_diff": m_mean, "ci95": [m_lo, m_hi], "promote": minutes_promote},
        "attacking_bootstrap": {"mean_diff": a_mean, "ci95": [a_lo, a_hi], "promote": attacking_promote},
    }
    (ARTIFACT_DIR / "tournament_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWritten to {(ARTIFACT_DIR / 'tournament_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
