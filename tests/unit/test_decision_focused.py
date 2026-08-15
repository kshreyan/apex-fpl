from __future__ import annotations

from apex_fpl.models import decision_focused as df
from apex_fpl.optimization import squad as sq

POSITIONS_POOL = {"GK": 4, "DEF": 8, "MID": 8, "FWD": 5}


def _build_pool(ep_overrides: dict[str, float] | None = None, actual_overrides: dict[str, int] | None = None):
    """A candidate pool with real squad.select_squad quotas satisfiable
    with surplus (so selection decisions are non-trivial), default EP and
    actual both equal to a flat baseline per player, overridable per id."""
    ep_overrides = ep_overrides or {}
    actual_overrides = actual_overrides or {}
    candidates_meta, ep_by_id, actual_by_player = {}, {}, {}
    for pos, n in POSITIONS_POOL.items():
        for i in range(n):
            pid = f"{pos}{i}"
            candidates_meta[pid] = {"position": pos, "team": f"club_{pos}_{i}", "price": 5.0, "name": pid}
            base = 4.0
            ep_by_id[pid] = ep_overrides.get(pid, base)
            actual_by_player[pid] = actual_overrides.get(pid, int(base))
    return candidates_meta, ep_by_id, actual_by_player


def test_apply_shrinkage_identity_at_one():
    _, ep_by_id, _ = _build_pool(ep_overrides={"MID0": 9.0, "DEF0": 1.0})
    position_by_id = {"MID0": "MID", "DEF0": "DEF"} | {f"{pos}{i}": pos for pos, n in POSITIONS_POOL.items() for i in range(n)}
    adjusted = df.apply_shrinkage(ep_by_id, position_by_id, shrinkage=1.0)
    assert adjusted == ep_by_id


def test_apply_shrinkage_collapses_to_median_at_zero():
    position_by_id = {f"{pos}{i}": pos for pos, n in POSITIONS_POOL.items() for i in range(n)}
    ep_by_id = {pid: float(i) for i, pid in enumerate(position_by_id)}  # spread of distinct values per position
    adjusted = df.apply_shrinkage(ep_by_id, position_by_id, shrinkage=0.0)

    import statistics
    by_pos: dict[str, list[float]] = {}
    for pid, ep in ep_by_id.items():
        by_pos.setdefault(position_by_id[pid], []).append(ep)
    medians = {pos: statistics.median(vals) for pos, vals in by_pos.items()}
    for pid in ep_by_id:
        assert abs(adjusted[pid] - medians[position_by_id[pid]]) < 1e-9


def test_hybrid_ep_blend_correctness():
    ep_a = {"p1": 10.0, "p2": 4.0}
    ep_b = {"p1": 6.0, "p2": 8.0}
    assert df.hybrid_ep(ep_a, ep_b, weight_b=0.0) == ep_a
    assert df.hybrid_ep(ep_a, ep_b, weight_b=1.0) == ep_b
    blended = df.hybrid_ep(ep_a, ep_b, weight_b=0.5)
    assert blended["p1"] == 8.0 and blended["p2"] == 6.0


def test_tune_shrinkage_prefers_no_shrinkage_when_predictions_are_perfect():
    candidates_meta, ep_by_id, actual_by_player = _build_pool()
    # give a few players distinct, ACCURATE ep==actual values so selection isn't a flat tie
    for pid, val in [("MID0", 9), ("MID1", 8), ("FWD0", 10), ("DEF0", 2)]:
        ep_by_id[pid] = float(val)
        actual_by_player[pid] = val

    tuning_gameweeks = [(candidates_meta, ep_by_id, actual_by_player)]
    chosen = df.tune_shrinkage(tuning_gameweeks, shrinkage_grid=[1.0, 0.7, 0.3, 0.0], select_squad_fn=sq.select_squad, select_starting_xi_fn=sq.select_starting_xi, player_candidate_cls=sq.PlayerCandidate)
    assert chosen == 1.0


def test_tune_shrinkage_prefers_shrinkage_when_a_player_is_a_persistent_overestimate():
    """A player whose predicted EP is consistently much higher than their
    actual realized points across every tuning gameweek is exactly the
    case shrinkage should help with: the optimizer keeps getting lured
    into starting/captaining them, and shrinking toward the position
    median (dragged down by everyone else's accurate, modest EP) should
    reduce that regret enough that a lower shrinkage wins on TOTAL
    realized points across the tuning set."""
    tuning_gameweeks = []
    for _ in range(3):
        candidates_meta, ep_by_id, actual_by_player = _build_pool()
        ep_by_id["MID0"] = 15.0  # looks like a clear captaincy-worthy star
        actual_by_player["MID0"] = 1  # but actually blanks
        tuning_gameweeks.append((candidates_meta, ep_by_id, actual_by_player))

    chosen = df.tune_shrinkage(tuning_gameweeks, shrinkage_grid=[1.0, 0.5, 0.0], select_squad_fn=sq.select_squad, select_starting_xi_fn=sq.select_starting_xi, player_candidate_cls=sq.PlayerCandidate)
    assert chosen < 1.0
