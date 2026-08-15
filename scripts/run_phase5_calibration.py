#!/usr/bin/env python3
"""Phase 5: probability calibration + uncertainty decomposition (spec
Parts XVIII, XX).

Part A — Calibration: fits isotonic/Platt calibrators for two real
champion-model probability outputs — the promoted minutes model's
P(60+ minutes), and the champion team model's clean-sheet probability —
on a DEDICATED calibration-fitting season (2020-21, which spec Part XX
requires to be separate from the evaluation sample), then measures
whether calibration significantly improves log loss/Brier/ECE on 3
genuinely held-out test seasons (2022-23, 2023-24, 2024-25 — the same
independent pool used in docs/phase3_extended_replay_report.md, reused
here for a different, non-circular purpose: evaluating calibration, not
tuning a model).

Part B — Uncertainty decomposition: runs the full promoted-champion
pipeline for one real gameweek (2022-23 GW20, matching the original
Phase 2 milestone for continuity), decomposes each player's simulated
point variance into selection/minutes vs aleatoric components (law of
total variance — spec Part XVIII), and reports team-model disagreement
(champion_unfit vs challenger_tuned from Phase 4a) per fixture as the
model-uncertainty signal.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from apex_fpl.backtesting import vaastav_loader as vl
from apex_fpl.calibration import calibrator as cal
from apex_fpl.evaluation import metrics as em
from apex_fpl.models.attacking import challengers as attacking_challengers
from apex_fpl.models.attacking import proportional as prop
from apex_fpl.models.minutes import challengers as minutes_challengers
from apex_fpl.models.minutes.baseline import forecast_minutes
from apex_fpl.models.teams import attack_defense as ad
from apex_fpl.models.teams import scoreline as sl
from apex_fpl.optimization import squad as sq
from apex_fpl.rules import scoring
from apex_fpl.simulation import monte_carlo as mc
from apex_fpl.simulation import uncertainty as unc

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "phase5_calibration"
ALL_FIXTURE_SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
CALIBRATION_FIT_SEASON = "2020-21"
TEST_SEASONS = ["2022-23", "2023-24", "2024-25"]
START_GW = 7
MINUTES_HALFLIFE = 3.0  # promoted Phase 4b champion
N_BOOTSTRAP = 5000
SEED = 2026


def load_all_fixture_rows():
    return sorted((r for s in ALL_FIXTURE_SEASONS for r in vl.load_fixtures(s)), key=lambda r: r["date"])


def collect_season_calibration_data(season: str, all_fixture_rows):
    """Returns minutes_records: [(season, gw, p_raw, y)] and
    clean_sheet_records: [(season, gw, p_raw, y)]."""
    minutes_records, cs_records = [], []
    gameweeks = [g for g in vl.season_gameweeks(season) if g >= START_GW]
    all_rows = vl.load_merged_gw(season)
    all_rows = [r for r in all_rows if r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]

    for gw in gameweeks:
        fixtures_this_gw = vl.fixtures_at_gw(season, gw)
        if not fixtures_this_gw:
            continue
        history_rows = [r for r in all_rows if int(r["GW"]) < gw]
        target_rows = [r for r in all_rows if int(r["GW"]) == gw]
        if not target_rows:
            continue

        minutes_by_player: dict[str, list[int]] = defaultdict(list)
        for r in sorted(history_rows, key=lambda r: int(r["GW"])):
            minutes_by_player[r["element"]].append(int(r["minutes"]))

        for r in target_rows:
            hist = minutes_by_player.get(r["element"], [])
            fc = minutes_challengers.exponential_decay(hist, half_life_matches=MINUTES_HALFLIFE)
            y = 1 if int(r["minutes"]) >= 60 else 0
            minutes_records.append((season, gw, fc.p_60_plus, y))

        training_fixtures = [
            ad.Fixture(f["date"], f["home_team"], f["away_team"], f["home_score"], f["away_score"])
            for f in all_fixture_rows if f["date"] < fixtures_this_gw[0]["date"] and f["home_score"] is not None
        ]
        if not training_fixtures:
            continue
        team_model = ad.fit(training_fixtures)
        for fx in fixtures_this_gw:
            if fx["home_score"] is None:
                continue
            eh, ea = team_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
            m = sl.score_matrix(eh, ea)
            home_cs_p = sl.clean_sheet_prob(m, "home")
            away_cs_p = sl.clean_sheet_prob(m, "away")
            cs_records.append((season, gw, home_cs_p, 1 if fx["away_score"] == 0 else 0))
            cs_records.append((season, gw, away_cs_p, 1 if fx["home_score"] == 0 else 0))
    return minutes_records, cs_records


def block_bootstrap(records, transform_fn) -> tuple[float, float, float]:
    """records: [(season, gw, p_raw, y)]. transform_fn(p_raw) -> p_calibrated.
    Block-bootstraps the (calibrated - raw) log-loss difference, block=(season,gw)."""
    blocks: dict[tuple, list[float]] = defaultdict(list)
    for season, gw, p_raw, y in records:
        p_cal = float(transform_fn(np.array([p_raw]))[0])
        ll_raw = em.log_loss_binary(np.array([p_raw]), [y])
        ll_cal = em.log_loss_binary(np.array([p_cal]), [y])
        blocks[(season, gw)].append(ll_cal - ll_raw)
    block_means = np.array([np.mean(v) for v in blocks.values()])
    rng = np.random.default_rng(SEED)
    n = len(block_means)
    boot = np.array([rng.choice(block_means, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
    return float(block_means.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def part_a_calibration(all_fixture_rows) -> dict:
    print(f"=== Part A: Calibration (fit on {CALIBRATION_FIT_SEASON}, test on {TEST_SEASONS}) ===\n")

    fit_minutes, fit_cs = collect_season_calibration_data(CALIBRATION_FIT_SEASON, all_fixture_rows)
    print(f"Calibration-fitting data: {len(fit_minutes)} minutes obs, {len(fit_cs)} clean-sheet obs")

    minutes_p = np.array([r[2] for r in fit_minutes])
    minutes_y = np.array([r[3] for r in fit_minutes])
    minutes_calibrator = cal.fit_calibrator(minutes_p, minutes_y)
    print(f"Minutes calibrator selected: {minutes_calibrator.method}")

    cs_p = np.array([r[2] for r in fit_cs])
    cs_y = np.array([r[3] for r in fit_cs])
    cs_calibrator = cal.fit_calibrator(cs_p, cs_y)
    print(f"Clean-sheet calibrator selected: {cs_calibrator.method}")

    test_minutes, test_cs = [], []
    for season in TEST_SEASONS:
        m, c = collect_season_calibration_data(season, all_fixture_rows)
        test_minutes.extend(m)
        test_cs.extend(c)
    print(f"\nTest data: {len(test_minutes)} minutes obs, {len(test_cs)} clean-sheet obs")

    results = {}
    for name, records, calibrator in [("minutes_p60", test_minutes, minutes_calibrator), ("clean_sheet_prob", test_cs, cs_calibrator)]:
        p_raw = np.array([r[2] for r in records])
        y = np.array([r[3] for r in records])
        p_cal = calibrator.transform(p_raw)

        raw_metrics = em.full_binary_metrics(p_raw, y)
        cal_metrics = em.full_binary_metrics(p_cal, y)
        raw_slope, raw_intercept = cal.calibration_slope_intercept(p_raw, y)
        cal_slope, cal_intercept = cal.calibration_slope_intercept(p_cal, y)
        mean_diff, ci_lo, ci_hi = block_bootstrap(records, calibrator.transform)

        print(f"\n{name}: method={calibrator.method}")
        print(f"  raw:        log_loss={raw_metrics['log_loss']}  brier={raw_metrics['brier']}  ece={raw_metrics['ece']}  slope={raw_slope:.3f}  intercept={raw_intercept:.3f}")
        print(f"  calibrated: log_loss={cal_metrics['log_loss']}  brier={cal_metrics['brier']}  ece={cal_metrics['ece']}  slope={cal_slope:.3f}  intercept={cal_intercept:.3f}")
        print(f"  block-bootstrap (calibrated-raw) log-loss diff: {mean_diff:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        promote = ci_hi < 0
        print(f"  decision: {'PROMOTE calibrated probabilities' if promote else 'do not promote (raw probabilities retained)'}")

        results[name] = {
            "calibrator_method": calibrator.method,
            "raw_metrics": raw_metrics, "calibrated_metrics": cal_metrics,
            "raw_slope_intercept": [raw_slope, raw_intercept], "calibrated_slope_intercept": [cal_slope, cal_intercept],
            "bootstrap_diff": {"mean": mean_diff, "ci95": [ci_lo, ci_hi], "promote": promote},
            "reliability_raw": cal.reliability_table(p_raw, y),
            "reliability_calibrated": cal.reliability_table(p_cal, y),
        }
    return results


def part_b_uncertainty(all_fixture_rows) -> dict:
    season, target_gw = "2022-23", 20
    print(f"\n=== Part B: Uncertainty decomposition ({season} GW{target_gw}) ===\n")

    training_fixtures = vl.fixtures_before_gw(season, target_gw)
    champion_model = ad.fit(training_fixtures)
    challenger_model = ad.fit(training_fixtures, k_base=0.08, halflife_days=730.0)  # Phase 4a's fold-1-winning tuned combo, for a concrete disagreement signal

    target_fixtures = vl.fixtures_at_gw(season, target_gw)
    fixture_inputs, disagreements = [], []
    for fx in target_fixtures:
        eh_champ, ea_champ = champion_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
        eh_chal, ea_chal = challenger_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
        m = sl.score_matrix(eh_champ, ea_champ)
        fixture_inputs.append(mc.FixtureInput(home_team=fx["home_team"], away_team=fx["away_team"], score_matrix=m))
        disagreements.append({
            "home_team": fx["home_team"], "away_team": fx["away_team"],
            "home": unc.model_disagreement(eh_champ, eh_chal).__dict__,
            "away": unc.model_disagreement(ea_champ, ea_chal).__dict__,
        })

    all_rows = vl.load_merged_gw(season)
    all_rows = [r for r in all_rows if r.get("position") in ("GK", "GKP", "DEF", "MID", "FWD")]
    history_rows = [r for r in all_rows if int(r["GW"]) < target_gw]
    target_rows = [r for r in all_rows if int(r["GW"]) == target_gw]
    minutes_by_player: dict[str, list[int]] = defaultdict(list)
    for r in sorted(history_rows, key=lambda r: int(r["GW"])):
        minutes_by_player[r["element"]].append(int(r["minutes"]))
    lookback_floor = target_gw - 15
    team_player_events: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for r in history_rows:
        if int(r["GW"]) < lookback_floor:
            continue
        team_player_events[r["team"]][r["element"]].append((int(r["goals_scored"]), int(r["assists"])))

    team_side = {}
    fixture_expected_goals = {}
    for fx in target_fixtures:
        team_side[fx["home_team"]] = "home"
        team_side[fx["away_team"]] = "away"
        eh, ea = champion_model.expected_goals(fx["home_team"], fx["away_team"], fx["date"])
        fixture_expected_goals[(fx["home_team"], "home")] = eh
        fixture_expected_goals[(fx["away_team"], "away")] = ea

    shares_by_team = {team: attacking_challengers.shrinkage_share(dict(hist), alpha=10.0) for team, hist in team_player_events.items()}

    roster = {r["element"]: r for r in target_rows}
    players_for_sim = []
    example_names = {}
    for pid, row in roster.items():
        team = row["team"]
        if team not in team_side:
            continue
        side = team_side[team]
        hist_minutes = [m for _, m in sorted([(int(x["GW"]), int(x["minutes"])) for x in history_rows if x["element"] == pid])]
        mfc = minutes_challengers.exponential_decay(hist_minutes, half_life_matches=MINUTES_HALFLIFE)
        shares = shares_by_team.get(team, {})
        share = shares.get(pid, prop.AttackingShare(0.0, 0.0))
        team_exp_goals = fixture_expected_goals[(team, side)]
        players_for_sim.append(mc.PlayerInput(
            player_id=pid, team=team, position=row["position"], minutes_forecast=mfc,
            expected_goals=team_exp_goals * share.goal_share, expected_assists=team_exp_goals * share.assist_share,
        ))
        example_names[pid] = row["name"]

    rules = scoring.load_scoring_rules("2026_27")
    sim_results = mc.simulate_gameweek(fixture_inputs, players_for_sim, rules, batch=3000, max_sims=30000, tol=0.05)

    decompositions = []
    for pid, result in sim_results.items():
        d = unc.decompose_variance(result)
        decompositions.append({
            "player_id": pid, "name": example_names[pid], "mean_points": round(result.mean_points, 3),
            "total_variance": round(d.total_variance, 3),
            "selection_minutes_share": d.selection_minutes_share, "aleatoric_share": d.aleatoric_share,
            "state_probs": d.state_probs, "state_means": {k: round(v, 3) for k, v in d.state_means.items()},
        })
    decompositions.sort(key=lambda r: -r["mean_points"])

    print("Top 10 players by projected points, with uncertainty decomposition:")
    for r in decompositions[:10]:
        print(f"  {r['name']:<22} EP={r['mean_points']:5.2f}  var={r['total_variance']:5.2f}  "
              f"selection/minutes_share={r['selection_minutes_share']:.2f}  aleatoric_share={r['aleatoric_share']:.2f}  "
              f"state_probs={r['state_probs']}")

    print("\nTeam model disagreement (champion vs a Phase-4a-tuned challenger) per fixture:")
    for d in disagreements:
        print(f"  {d['home_team']:<18} vs {d['away_team']:<18} "
              f"home_disagreement={d['home']['absolute_disagreement']:.3f}  away_disagreement={d['away']['absolute_disagreement']:.3f}")

    return {"season": season, "target_gw": target_gw, "player_decompositions": decompositions, "fixture_model_disagreement": disagreements}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    all_fixture_rows = load_all_fixture_rows()

    calibration_results = part_a_calibration(all_fixture_rows)
    uncertainty_results = part_b_uncertainty(all_fixture_rows)

    summary = {"calibration": calibration_results, "uncertainty": uncertainty_results}
    out_path = ARTIFACT_DIR / "phase5_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWritten to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
