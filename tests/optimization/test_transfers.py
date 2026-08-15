from __future__ import annotations

from collections import Counter

from apex_fpl.optimization import transfers as tf

POSITIONS = ["GK", "GK", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"]
CAPTAIN_ANCHOR_ID = "cur0"  # always the highest-EP player, so it always keeps the captaincy — isolates
                            # "does this candidate get bought as a normal starter" from the captain-doubling
                            # effect (a higher-EP transfer-in is worth its raw gain PLUS an extra captain-swap
                            # bonus if it would outrank the current captain — a real, correct MILP behavior,
                            # not something these threshold tests want to be testing).


def _pl(pid, pos, team, price, ep):
    return tf.HorizonPlayer(pid, pos, team, price, tuple(ep))


def _base_squad(horizon, ep_overrides: dict | None = None, price=5.0):
    ep_overrides = ep_overrides or {}
    players = []
    for i, pos in enumerate(POSITIONS):
        pid = f"cur{i}"
        default_ep = 100.0 if pid == CAPTAIN_ANCHOR_ID else 4.0
        ep = ep_overrides.get(pid, tuple(default_ep for _ in range(horizon)))
        players.append(_pl(pid, pos, f"club{i}", price, ep))
    return players


def _current_ids():
    return [f"cur{i}" for i in range(15)]


def _sell_prices(price=5.0):
    return {pid: price for pid in _current_ids()}


def _assert_legal_squad(squad_ids, players_by_id):
    assert len(squad_ids) == 15
    pos_counts = Counter(players_by_id[pid].position for pid in squad_ids)
    assert pos_counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, pos_counts
    club_counts = Counter(players_by_id[pid].team for pid in squad_ids)
    assert all(v <= 3 for v in club_counts.values()), club_counts


def test_no_beneficial_transfer_never_starts_the_worse_candidate():
    """A strictly worse candidate should never make the starting XI or
    change the score — whether the solver also happens to swap it onto
    the BENCH for a currently-benched player is a genuine, harmless tie
    (bench players don't count this gameweek either way), not something
    this test should assert against."""
    players = _base_squad(1) + [_pl("worse_mid", "MID", "clubX", 5.0, (1.0,))]
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=1, players=players, horizon=1, shortlist_per_position=None)
    gw = plan.gameweeks[0]
    assert "worse_mid" not in gw.starters
    assert gw.captain == CAPTAIN_ANCHOR_ID
    assert plan.total_net_expected_points == 240.0  # 1*100 (GK, doubled) + 5*4 (DEF) + 2*4 (MID, only 2 of 5 start) + 3*4 (FWD) + 100 (captain bonus)


def test_beneficial_swap_uses_free_transfer_no_hit():
    players = _base_squad(1) + [_pl("better_mid", "MID", "clubX", 5.0, (14.0,))]  # still well below the captain anchor (100)
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=1, players=players, horizon=1, shortlist_per_position=None)
    gw = plan.gameweeks[0]
    assert gw.transfers_in == ["better_mid"]
    assert len(gw.transfers_out) == 1
    out_player = next(p for p in players if p.player_id == gw.transfers_out[0])
    assert out_player.position == "MID"
    assert gw.captain == CAPTAIN_ANCHOR_ID  # unaffected — the anchor still dominates
    assert gw.paid_transfers == 0
    assert gw.hit_points == 0.0


def test_no_hit_taken_when_gain_is_smaller_than_hit_cost():
    players = _base_squad(1) + [_pl("marginal_mid", "MID", "clubX", 5.0, (6.5,))]  # gain 2.5 < hit cost 4, no captaincy change
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=0, players=players, horizon=1, shortlist_per_position=None)
    gw = plan.gameweeks[0]
    assert gw.transfers_in == [] and gw.transfers_out == []


def test_hit_taken_when_gain_exceeds_hit_cost():
    players = _base_squad(1) + [_pl("great_mid", "MID", "clubX", 5.0, (10.0,))]  # gain 6 > hit cost 4, no captaincy change (10 < 100)
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=0, players=players, horizon=1, shortlist_per_position=None)
    gw = plan.gameweeks[0]
    assert gw.transfers_in == ["great_mid"]
    assert gw.paid_transfers == 1
    assert gw.hit_points == -4.0


def test_free_transfer_banking_caps_at_five():
    h = 6
    players = _base_squad(h) + [_pl("worse_mid", "MID", "clubX", 5.0, tuple(1.0 for _ in range(h)))]
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=1, players=players, horizon=h, shortlist_per_position=None)
    ft_seq = [gw.free_transfers_available for gw in plan.gameweeks]
    assert ft_seq == [1, 2, 3, 4, 5, 5]
    for gw in plan.gameweeks:
        assert gw.transfers_in == [] and gw.transfers_out == []


def test_budget_constraint_blocks_unaffordable_transfer():
    players = _base_squad(1) + [_pl("dream_signing", "MID", "clubX", 50.0, (99.0,))]  # below anchor's 100, so captaincy unaffected — isolates the budget block
    plan = tf.plan_transfers(_current_ids(), _sell_prices(price=5.0), bank=0.0, free_transfers=1, players=players, horizon=1, shortlist_per_position=None)
    gw = plan.gameweeks[0]
    assert gw.transfers_in == [] and gw.transfers_out == []
    assert gw.bank_after == 0.0


def test_squad_continuity_and_legality_across_periods_with_a_mid_horizon_transfer():
    h = 3
    ep_spike = _pl("spike_fwd", "FWD", "clubX", 5.0, (1.0, 20.0, 20.0))  # worth transferring in only from gw index 1 onward; stays below the anchor's 100 throughout
    players = _base_squad(h) + [ep_spike]
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=1, players=players, horizon=h, shortlist_per_position=None)
    players_by_id = {p.player_id: p for p in players}

    prev_squad = set(_current_ids())
    for gw in plan.gameweeks:
        squad_ids = set(gw.squad)
        _assert_legal_squad(squad_ids, players_by_id)
        assert squad_ids - prev_squad == set(gw.transfers_in)
        assert prev_squad - squad_ids == set(gw.transfers_out)
        prev_squad = squad_ids

    # spike_fwd MUST be owned by gw index 1 (necessary — that's when its EP matters).
    # Whether it's acquired exactly at gw0 or gw1 is a genuine, harmless tie (its
    # gw0 EP of 1.0 is bench-caliber either way, and the transfer is free), so
    # this test doesn't assert on that timing.
    assert "spike_fwd" in plan.gameweeks[1].squad
    assert "spike_fwd" in plan.gameweeks[2].squad
    assert plan.gameweeks[2].transfers_in == []  # no reason to sell back and forth
    assert "spike_fwd" in plan.gameweeks[1].starters
    assert "spike_fwd" in plan.gameweeks[2].starters


def _split_into_gw_universes(players, horizon):
    return [[tf.HorizonPlayer(p.player_id, p.position, p.team, p.price, (p.ep_by_gw[t],)) for p in players] for t in range(horizon)]


def test_lookahead_beats_myopic_by_banking_a_free_transfer_for_a_double_opportunity():
    """The flagship test: a real FPL strategic pattern — bank this week's
    free transfer instead of spending it on a small upgrade, so that NEXT
    week you have 2 free transfers and can capture a bigger two-player
    opportunity (e.g. a double gameweek) without taking a hit.

    Candidate A: a modest gw0-only upgrade (+2 EP over the baseline MID it
    would replace). Candidates B1/B2: both spike hugely at gw1 only
    (+11 EP each over their baseline DEF/FWD), and capturing BOTH needs 2
    simultaneous transfers.

    - Take A at gw0 (uses the only free transfer): +2 now, but only 1 FT
      left at gw1, so getting both B1+B2 needs 1 paid hit: net gw1 =
      +22-4=+18. Total = +20.
    - Skip A at gw0 (0 now, banks the FT to 2 available at gw1): both
      B1+B2 free at gw1 (+22, no hit). Total = +22.

    Skipping A is the GLOBALLY better 2-week plan (+22 > +20) — but a
    myopic, one-week-at-a-time policy has no way to know that skipping a
    currently-positive gain is worth it, and will always greedily take A.
    Both policies are run through the exact same `rolling_horizon_transfers`
    driver (varying only `horizon`), so this isn't comparing two different
    code paths — it's the same machinery seeing more or less of the future.
    """
    players_full = _base_squad(2) + [
        _pl("A", "MID", "clubA_", 5.0, (6.0, 4.0)),
        _pl("B1", "DEF", "clubB1", 5.0, (1.0, 15.0)),
        _pl("B2", "FWD", "clubB2", 5.0, (1.0, 15.0)),
    ]

    lookahead_plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=1, players=players_full, horizon=2, shortlist_per_position=None)
    # "A" is never bought at all — buying it would cost a free transfer that
    # dominant play needs to keep both B1 and B2 hit-free. WHICH week each of
    # B1/B2 gets bought in is itself a harmless tie (both are bench-caliber
    # before gw1, so it doesn't matter whether they arrive early or exactly on
    # time) — the solver may split them one-per-week or both at once, both
    # reach the same total. What's robustly true: no hit is ever taken, and
    # both are owned by the time they matter (gw index 1).
    assert "A" not in lookahead_plan.gameweeks[0].squad and "A" not in lookahead_plan.gameweeks[1].squad
    assert {"B1", "B2"} <= set(lookahead_plan.gameweeks[1].squad)
    assert sum(gw.paid_transfers for gw in lookahead_plan.gameweeks) == 0
    assert lookahead_plan.total_net_expected_points == 502.0

    gw_universes = _split_into_gw_universes(players_full, 2)
    myopic_steps = tf.rolling_horizon_transfers(_current_ids(), _sell_prices(), 0.0, 1, gw_universes, horizon=1, shortlist_per_position=None)
    assert myopic_steps[0].transfers_in == ["A"]  # greedily takes the immediate gain, with no way to know better
    assert set(myopic_steps[1].transfers_in) == {"B1", "B2"}
    assert myopic_steps[1].paid_transfers == 1  # forced to pay a hit for the second transfer

    def _step_points(step, ep_by_id):
        return sum(ep_by_id[pid] for pid in step.starters) + ep_by_id[step.captain] + step.hit_points

    myopic_total = sum(_step_points(s, {p.player_id: p.ep_by_gw[0] for p in gw_universes[t]}) for t, s in enumerate(myopic_steps))
    assert myopic_total == 500.0

    assert lookahead_plan.total_net_expected_points - myopic_total == 2.0


def test_shortlist_candidates_keeps_current_squad_and_top_n_per_position():
    current = {"cur0"}
    players = [
        tf.HorizonPlayer("cur0", "MID", "clubA", 5.0, (1.0,)),
        tf.HorizonPlayer("m1", "MID", "clubB", 5.0, (9.0,)),
        tf.HorizonPlayer("m2", "MID", "clubC", 5.0, (8.0,)),
        tf.HorizonPlayer("m3", "MID", "clubD", 5.0, (1.0,)),
        tf.HorizonPlayer("f1", "FWD", "clubE", 5.0, (5.0,)),
    ]
    kept = tf.shortlist_candidates(current, players, per_position=2)
    kept_ids = {p.player_id for p in kept}
    assert kept_ids == {"cur0", "m1", "m2", "f1"}  # m3 dropped (worst MID beyond top-2)


def test_rolling_horizon_keeps_currently_owned_player_absent_from_a_later_universe():
    """A real bug found running the Phase 7 replay on real data: a
    currently-owned player can legitimately be absent from a later
    gameweek's universe (their team had no fixture that gameweek — a
    postponement, not a full blank gameweek). They are still a real squad
    member, not a player who ceased to exist, and must not make
    rolling_horizon_transfers crash or silently vanish from the squad."""
    full_squad = _base_squad(1)
    gw0_universe = [tf.HorizonPlayer(p.player_id, p.position, p.team, p.price, (p.ep_by_gw[0],)) for p in full_squad]
    gw1_universe = [tf.HorizonPlayer(p.player_id, p.position, p.team, p.price, (p.ep_by_gw[0],)) for p in full_squad if p.player_id != "cur7"]
    steps = tf.rolling_horizon_transfers(_current_ids(), _sell_prices(), 0.0, 1, [gw0_universe, gw1_universe], horizon=1, shortlist_per_position=None)
    assert "cur7" in steps[1].squad  # still owned
    assert "cur7" not in steps[1].starters  # correctly treated as scoring 0 (didn't play) rather than its stale prior EP


def test_total_net_expected_points_matches_manual_arithmetic():
    players = _base_squad(1)
    plan = tf.plan_transfers(_current_ids(), _sell_prices(), bank=0.0, free_transfers=1, players=players, horizon=1, shortlist_per_position=None)
    gw = plan.gameweeks[0]
    assert gw.transfers_in == []
    ep_by_id = {p.player_id: p.ep_by_gw[0] for p in players}
    expected = sum(ep_by_id[pid] for pid in gw.starters) + ep_by_id[gw.captain] + gw.hit_points
    assert abs(plan.total_net_expected_points - expected) < 1e-6
