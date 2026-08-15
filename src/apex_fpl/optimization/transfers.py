"""Multi-gameweek transfer-and-squad optimizer (Phase 7; spec Parts
XXIV-XXX subset: transfers, hits, free-transfer banking, bench order,
captaincy, multi-gameweek horizon).

An exact MILP over a rolling horizon of H forecasted gameweeks: for each
gameweek it jointly chooses squad membership, starting XI, captain, AND
which transfers to make to get there from the previous gameweek's squad,
correctly accounting for free-transfer banking (1/gameweek, up to 5
banked, per configs/seasons/2026_27.yaml's confirmed 2026/27 rules) and
the -4-point hit cost for transfers beyond the free allowance. This is
the real structural difference from `apex_fpl.backtesting.replay`, which
deliberately picks "best XV from scratch" independently each gameweek
(see that module's own docstring) — this module evolves ONE persistent
squad through transfer decisions, the actual question spec Part
XXIV-XXX and this project's research_plan.md Phase 7 entry ask for.

Two explicit, honestly-flagged simplifications, both driven by real gaps
already tracked in docs/fpl_gap_analysis.md, not new omissions:

1. **Prices are held fixed across the optimization horizon.** No
   price-change forecasting model exists yet (Part XXII, tracked NONE).
   A real mid-horizon price rise/fall is not modeled. This is far less
   costly than it sounds given the intended usage pattern below.
2. **The transfer-IN candidate universe is shortlisted** (current squad
   plus the top `shortlist_per_position` non-owned players per position,
   ranked by their own best single-gameweek EP anywhere in the horizon)
   purely for MILP tractability — global optimality over the full
   ~600-700 player pool is not claimed. See `shortlist_candidates()`.

**Intended usage: receding horizon.** Call with `horizon=1` for a myopic
(single-gameweek-lookahead) transfer policy, or `horizon>1` for genuine
multi-gameweek lookahead — in BOTH cases, re-solve every real gameweek
with fresh forecasts and only commit that gameweek's transfers/squad,
then discard the rest of the plan and re-solve next week. This is
standard receding-horizon control (the same principle spec Part XXVI
applies to scenario robustness, applied here to a transfer plan instead)
and is exactly what makes simplification (1) low-cost: the committed
decision is always the FIRST gameweek's, decided using this week's real
prices, not a stale future price. `scripts/run_phase7_rolling_horizon_replay.py`
implements this rolling pattern and is the only supported way to use a
plan's transfers beyond gameweek 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from apex_fpl.optimization.squad import (
    MAX_PER_CLUB,
    SQUAD_QUOTAS,
    STARTING_XI_MAX,
    STARTING_XI_MIN,
    STARTING_XI_SIZE,
)

SQUAD_SIZE = sum(SQUAD_QUOTAS.values())


@dataclass(frozen=True)
class HorizonPlayer:
    player_id: str
    position: str  # GK/DEF/MID/FWD
    team: str
    price: float  # £m, held fixed across the horizon — see module docstring
    ep_by_gw: tuple[float, ...]  # expected points per horizon gameweek, len == horizon (0.0 for a blank gameweek)


@dataclass(frozen=True)
class GameweekPlan:
    gw_index: int  # 0-based offset into the horizon (0 = the gameweek being decided now)
    squad: list[str]
    starters: list[str]
    captain: str
    bench_order: list[str]  # non-starters, ranked by descending EP this GW — see note below
    transfers_in: list[str]
    transfers_out: list[str]
    free_transfers_available: int
    paid_transfers: int
    hit_points: float  # <= 0
    bank_after: float


@dataclass(frozen=True)
class TransferPlan:
    horizon: int
    gameweeks: list[GameweekPlan]
    total_net_expected_points: float  # sum over the horizon of (starters EP + captain EP bonus) + hit_points
    proven_optimal: bool
    mip_gap: float | None
    message: str


def shortlist_candidates(current_squad_ids: set[str], players: list[HorizonPlayer], per_position: int = 15) -> list[HorizonPlayer]:
    """Restrict the transfer-in candidate universe for MILP tractability
    (see module docstring) — the current squad is always kept in full;
    non-owned players are ranked by their own best single-gameweek EP
    anywhere in the horizon and the top `per_position` per position are
    kept. Not a claim that excluded players could never be worth buying."""
    kept = [p for p in players if p.player_id in current_squad_ids]
    by_pos: dict[str, list[HorizonPlayer]] = {}
    for p in players:
        if p.player_id in current_squad_ids:
            continue
        by_pos.setdefault(p.position, []).append(p)
    for pool in by_pos.values():
        pool.sort(key=lambda p: -max(p.ep_by_gw))
        kept.extend(pool[:per_position])
    return kept


def plan_transfers(
    current_squad_ids: list[str],
    sell_price_by_id: dict[str, float],
    bank: float,
    free_transfers: int,
    players: list[HorizonPlayer],
    horizon: int,
    hit_cost: float = -4.0,
    max_free_transfers: int = 5,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    shortlist_per_position: int | None = 15,
) -> TransferPlan:
    if len(current_squad_ids) != SQUAD_SIZE:
        raise ValueError(f"current squad must have exactly {SQUAD_SIZE} players, got {len(current_squad_ids)}")
    if shortlist_per_position is not None:
        players = shortlist_candidates(set(current_squad_ids), players, per_position=shortlist_per_position)

    ids = [p.player_id for p in players]
    idx = {pid: i for i, pid in enumerate(ids)}
    missing = [pid for pid in current_squad_ids if pid not in idx]
    if missing:
        raise ValueError(f"current squad player(s) not present in `players` (after shortlisting): {missing}")
    for p in players:
        if len(p.ep_by_gw) != horizon:
            raise ValueError(f"player {p.player_id} has ep_by_gw of length {len(p.ep_by_gw)}, expected horizon={horizon}")

    n, h = len(ids), horizon
    current_set = set(current_squad_ids)
    prev_in_squad = np.array([1.0 if pid in current_set else 0.0 for pid in ids])
    price = np.array([p.price for p in players])
    effective_sell_price = np.array([sell_price_by_id.get(pid, players[i].price) for i, pid in enumerate(ids)])
    ep = np.array([[players[i].ep_by_gw[t] for i in range(n)] for t in range(h)])  # (h, n)

    block = n * h
    n_ft = max(h - 1, 0)
    n_vars = 5 * block + n_ft + h + h
    off_s, off_y, off_cap, off_buy, off_sell = 0, block, 2 * block, 3 * block, 4 * block
    off_ft = 5 * block
    off_paid = off_ft + n_ft
    off_bank = off_paid + h

    def S(i, t): return off_s + t * n + i
    def Y(i, t): return off_y + t * n + i
    def CAP(i, t): return off_cap + t * n + i
    def BUY(i, t): return off_buy + t * n + i
    def SELL(i, t): return off_sell + t * n + i
    def FT(t): return off_ft + (t - 1)  # only defined for t=1..h-1
    def PAID(t): return off_paid + t
    def BANK(t): return off_bank + t

    pos_hit = -hit_cost  # hit_cost is <=0 (e.g. -4.0); minimizing c@x should PENALIZE paid transfers by +4 each

    c = np.zeros(n_vars)
    for t in range(h):
        c[[Y(i, t) for i in range(n)]] = -ep[t]
        c[[CAP(i, t) for i in range(n)]] = -ep[t]
        c[PAID(t)] = pos_hit

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0

    def add_row(pairs: list[tuple[int, float]], lo: float, hi: float) -> None:
        nonlocal r
        for col, val in pairs:
            rows.append(r); cols.append(col); vals.append(val)
        lb.append(lo); ub.append(hi)
        r += 1

    positions = sorted(SQUAD_QUOTAS)
    clubs = sorted({p.team for p in players})

    for t in range(h):
        add_row([(S(i, t), 1.0) for i in range(n)], SQUAD_SIZE, SQUAD_SIZE)
        for pos in positions:
            add_row([(S(i, t), 1.0) for i in range(n) if players[i].position == pos], SQUAD_QUOTAS[pos], SQUAD_QUOTAS[pos])
        for club in clubs:
            add_row([(S(i, t), 1.0) for i in range(n) if players[i].team == club], -np.inf, MAX_PER_CLUB)

        add_row([(Y(i, t), 1.0) for i in range(n)], STARTING_XI_SIZE, STARTING_XI_SIZE)
        for i in range(n):
            add_row([(Y(i, t), 1.0), (S(i, t), -1.0)], -np.inf, 0)  # y <= s
        for pos in positions:
            add_row([(Y(i, t), 1.0) for i in range(n) if players[i].position == pos], STARTING_XI_MIN[pos], STARTING_XI_MAX[pos])

        add_row([(CAP(i, t), 1.0) for i in range(n)], 1, 1)
        for i in range(n):
            add_row([(CAP(i, t), 1.0), (Y(i, t), -1.0)], -np.inf, 0)  # cap <= y

        # squad continuity: s(t) - s(t-1) - buy(t) + sell(t) = 0 (s(-1) = prev_in_squad, a constant moved to RHS)
        for i in range(n):
            if t == 0:
                add_row([(S(i, 0), 1.0), (BUY(i, 0), -1.0), (SELL(i, 0), 1.0)], prev_in_squad[i], prev_in_squad[i])
            else:
                add_row([(S(i, t), 1.0), (S(i, t - 1), -1.0), (BUY(i, t), -1.0), (SELL(i, t), 1.0)], 0, 0)

        # paid(t) >= made(t) - ft(t); ft(0) is the constant `free_transfers`
        buy_terms = [(BUY(i, t), -1.0) for i in range(n)]
        if t == 0:
            add_row([(PAID(0), 1.0)] + buy_terms, -free_transfers, np.inf)
        else:
            add_row([(PAID(t), 1.0)] + buy_terms + [(FT(t), 1.0)], 0, np.inf)

        # bank(t) = bank(t-1) + sum(sell_price*sell) - sum(price*buy); bank(-1) = given `bank`
        bank_terms = [(SELL(i, t), effective_sell_price[i]) for i in range(n)] + [(BUY(i, t), -price[i]) for i in range(n)]
        if t == 0:
            add_row([(BANK(0), 1.0)] + [(c_, -v) for c_, v in bank_terms], bank, bank)
        else:
            add_row([(BANK(t), 1.0), (BANK(t - 1), -1.0)] + [(c_, -v) for c_, v in bank_terms], 0, 0)

    for t in range(1, h):
        # ft(t) <= ft(t-1) - made(t-1) + paid(t-1) + 1, i.e.
        # ft(t) - ft(t-1) + made(t-1) - paid(t-1) <= 1 (ft(0) is the constant `free_transfers`, moved to the RHS)
        pairs = [(FT(t), 1.0)]
        if t - 1 >= 1:
            pairs.append((FT(t - 1), -1.0))
            rhs = 1.0
        else:
            rhs = 1.0 + free_transfers
        pairs += [(BUY(i, t - 1), 1.0) for i in range(n)]
        pairs.append((PAID(t - 1), -1.0))
        add_row(pairs, -np.inf, rhs)

    n_constraints = r
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars)).tocsr()
    constraints = [LinearConstraint(A, np.array(lb), np.array(ub))]

    integrality = np.zeros(n_vars)
    integrality[:5 * block] = 1  # s, y, cap, buy, sell all binary
    integrality[off_ft:off_paid] = 1  # ft integer
    integrality[off_paid:off_bank] = 1  # paid integer
    # bank stays continuous

    lower = np.zeros(n_vars)
    upper = np.ones(n_vars)
    upper[off_ft:off_paid] = max_free_transfers
    lower[off_ft:off_paid] = 1
    upper[off_paid:off_bank] = SQUAD_SIZE
    upper[off_bank:] = np.inf
    bounds = Bounds(lb=lower, ub=upper)

    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = mip_rel_gap

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options or None)
    if res.status not in (0, 1) or res.x is None:
        raise RuntimeError(f"transfer plan optimization infeasible: {res.message}")

    x = res.x
    gameweeks: list[GameweekPlan] = []
    prev_squad_ids = current_set
    ft_before = free_transfers
    bank_val = bank
    total_net = 0.0
    for t in range(h):
        squad_ids = {ids[i] for i in range(n) if round(x[S(i, t)]) == 1}
        starters_ids = {ids[i] for i in range(n) if round(x[Y(i, t)]) == 1}
        captain_id = next(ids[i] for i in range(n) if round(x[CAP(i, t)]) == 1)
        transfers_in = sorted(squad_ids - prev_squad_ids)
        transfers_out = sorted(prev_squad_ids - squad_ids)
        made = len(transfers_in)
        assert made == len(transfers_out), "squad size drifted — a bug in the continuity constraints"
        paid_t = max(0, made - ft_before)
        # cross-check against the LP's own internal PAID variable — a real
        # discrepancy here means the constraint math has a bug, not
        # something to silently paper over.
        lp_paid = round(x[PAID(t)])
        assert lp_paid == paid_t, f"gw{t}: LP paid={lp_paid} != recomputed paid={paid_t} from squad diff"
        hit_points = hit_cost * paid_t

        spend = sum(price[idx[pid]] for pid in transfers_in)
        gain = sum(effective_sell_price[idx[pid]] for pid in transfers_out)
        bank_val = bank_val + gain - spend
        assert bank_val >= -1e-6, f"gw{t}: bank went negative ({bank_val}) — a bug in the budget constraints"

        ep_t = ep[t]
        bench_ids = sorted(squad_ids - starters_ids, key=lambda pid: -ep_t[idx[pid]])
        total_net += sum(ep_t[idx[pid]] for pid in starters_ids) + ep_t[idx[captain_id]] + hit_points

        gameweeks.append(GameweekPlan(
            gw_index=t, squad=sorted(squad_ids), starters=sorted(starters_ids), captain=captain_id,
            bench_order=bench_ids, transfers_in=transfers_in, transfers_out=transfers_out,
            free_transfers_available=ft_before, paid_transfers=paid_t, hit_points=hit_points, bank_after=bank_val,
        ))

        ft_before = min(max_free_transfers, max(1, ft_before - made + paid_t + 1))
        prev_squad_ids = squad_ids

    return TransferPlan(
        horizon=h, gameweeks=gameweeks, total_net_expected_points=total_net,
        proven_optimal=res.status == 0, mip_gap=getattr(res, "mip_gap", None), message=res.message,
    )


def rolling_horizon_transfers(
    initial_squad_ids: list[str],
    initial_sell_price_by_id: dict[str, float],
    initial_bank: float,
    initial_free_transfers: int,
    gw_universes: list[list[HorizonPlayer]],
    horizon: int,
    hit_cost: float = -4.0,
    max_free_transfers: int = 5,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
    shortlist_per_position: int | None = 15,
) -> list[GameweekPlan]:
    """Receding-horizon driver — the pattern the module docstring
    prescribes for actually USING a plan: at each real gameweek t, solve
    plan_transfers() over the window [t, t+horizon) (shrinking near the
    end of `gw_universes`), commit ONLY that gameweek's decision, advance
    squad/bank/free-transfers, then re-solve fresh for t+1. horizon=1
    gives a purely myopic rolling policy; horizon>1 gives genuine
    multi-gameweek lookahead — both go through this same driver, so a
    myopic-vs-lookahead comparison differs only in the `horizon` argument,
    not in two different code paths.

    `gw_universes[t]` holds each available player as a HorizonPlayer with
    a length-1 `ep_by_gw` — that real gameweek's own forecast only. This
    function assembles the sliding multi-gameweek window internally by
    looking each player up across `gw_universes[t:t+horizon]`, treating a
    player absent from a given gameweek's universe as unavailable that
    week (ep=0.0 — e.g. their team has no fixture that gameweek, a
    postponement or a blank gameweek) — including a CURRENTLY-OWNED
    player who is absent from every universe in the current window: they
    are still a real squad member occupying a real slot (they just score
    0 that week), not a player who has ceased to exist, so they are kept
    in the window with ep=0.0 rather than silently dropped (which would
    otherwise make `plan_transfers` reject the call outright, since it
    requires every current-squad player to be present in `players`).
    Their position/team/price are recovered from the most recent
    gameweek in which they DID appear anywhere in `gw_universes`.
    """
    for g in gw_universes:
        for p in g:
            if len(p.ep_by_gw) != 1:
                raise ValueError("gw_universes entries must have a length-1 ep_by_gw (one real gameweek's own forecast)")

    known_meta: dict[str, HorizonPlayer] = {}
    for g in gw_universes:
        for p in g:
            known_meta[p.player_id] = p

    squad_ids = list(initial_squad_ids)
    sell_price_by_id = dict(initial_sell_price_by_id)
    bank = initial_bank
    free_transfers = initial_free_transfers
    committed: list[GameweekPlan] = []

    for t in range(len(gw_universes)):
        eff_horizon = min(horizon, len(gw_universes) - t)
        by_offset = [{p.player_id: p for p in gw_universes[t + offset]} for offset in range(eff_horizon)]
        all_ids = set().union(*(set(lookup) for lookup in by_offset)) | set(squad_ids)

        window_players = []
        for pid in all_ids:
            if any(pid in lookup for lookup in by_offset):
                ref = next(lookup[pid] for lookup in by_offset if pid in lookup)
            elif pid in known_meta:
                ref = known_meta[pid]
            else:
                raise ValueError(f"current-squad player {pid} never appears in any gw_universe — cannot determine position/team/price")
            ep_tuple = tuple(by_offset[offset][pid].ep_by_gw[0] if pid in by_offset[offset] else 0.0 for offset in range(eff_horizon))
            window_players.append(HorizonPlayer(pid, ref.position, ref.team, ref.price, ep_tuple))

        plan = plan_transfers(
            squad_ids, sell_price_by_id, bank, free_transfers, window_players, eff_horizon,
            hit_cost=hit_cost, max_free_transfers=max_free_transfers, time_limit=time_limit,
            mip_rel_gap=mip_rel_gap, shortlist_per_position=shortlist_per_position,
        )
        step = plan.gameweeks[0]
        committed.append(step)

        window_price_by_id = {p.player_id: p.price for p in window_players}
        squad_ids = step.squad
        free_transfers = min(max_free_transfers, max(1, step.free_transfers_available - len(step.transfers_in) + step.paid_transfers + 1))
        bank = step.bank_after
        for pid in step.transfers_in:
            sell_price_by_id[pid] = window_price_by_id[pid]
        for pid in step.transfers_out:
            sell_price_by_id.pop(pid, None)

    return committed
