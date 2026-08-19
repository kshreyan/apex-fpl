"""Rank-aware squad selector (Phase 13 — item 3 of the "explicitly
disclosed, not built" gaps: the biggest and most novel, deliberately
built last).

Every squad optimizer through Phase 12 (`apex_fpl.optimization.squad.
select_squad`) maximizes MY squad's own expected points — none of them
know anything about the competitive field. Phase 10's field/rank
simulator (`apex_fpl.simulation.field`) exists to model exactly that,
but only ever MEASURED a given squad's rank/percentile after the fact
(`scripts/run_phase10_field_simulation_demo.py`); nothing in this
project has ever used that measurement to CHOOSE a squad. That's the
real gap this module closes.

**Why this isn't just "add rank as another linear term in the existing
squad MILP."** A squad's percentile against the field is NOT a linear
function of its players' individual EP values — it depends on the
JOINT, correlated, scenario-by-scenario outcome of the whole squad
relative to the whole field (`fsim.simulate_field_scores` /
`fsim.my_percentile_per_scenario`), which a linear MILP objective
cannot express. Rather than inventing a new, fragile nonlinear
optimization formulation, this reuses the two building blocks that
already exist and are already validated (`select_squad`'s max-EV MILP,
and the field Monte Carlo) as a SEARCH-AND-SCORE procedure instead:

1. Generate a small set of CANDIDATE squads: the existing max-EV squad,
   plus a handful of DIFFERENTIALS — each one swaps exactly one of the
   max-EV squad's most heavily-owned members for the lowest-ownership
   same-position, budget-and-club-legal alternative available, at a
   bounded EV cost (`max_ev_loss_fraction` of the max-EV squad's own
   total EV — not an unbounded chase for uniqueness).
2. Score every candidate against the SAME synthetic field (reusing the
   correlated per-scenario samples, exactly as Phase 10 already does)
   — mean EV, mean simulated score, mean percentile, P(top 10%/25%).
3. Rank-aware selection picks whichever LEGAL candidate maximizes a
   rank-oriented target metric (default: P(top decile) of the field)
   — not a silent swap: every candidate's full tradeoff table is
   returned, so a human (or the caller) sees exactly what EV was
   traded for what rank gain.

Deliberately ONE swap per candidate, not a combinatorial search over
several simultaneous swaps — that would need its own MILP to stay
tractable and would defeat the "reuse validated building blocks"
premise this module is built on. A genuinely differentiable or
multi-swap formulation is a real, stated next step, not attempted here.

This is a genuinely different question from "what's the best squad,"
which is why it stays a SEPARATE capability rather than replacing
predict.py's own from-scratch squad — CLAUDE.md's standing rule is that
a new capability doesn't silently become the production default; it
earns that only through this project's own promotion discipline
(pre-registered comparison, statistical significance), which a single
gameweek's swap-or-not decision cannot provide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apex_fpl.optimization import squad as sq
from apex_fpl.simulation import field as fsim


@dataclass(frozen=True)
class CandidateSquadResult:
    label: str
    squad_ids: tuple[str, ...]
    swapped_out: str | None  # player_id, or None for the max-EV squad itself
    swapped_in: str | None
    mean_ev: float
    mean_simulated_score: float
    mean_percentile: float
    p_top10pct: float
    p_top25pct: float


@dataclass(frozen=True)
class RankAwareSelectionResult:
    candidates: tuple[CandidateSquadResult, ...]  # max-EV squad first, then differentials
    selected: CandidateSquadResult  # the candidate maximizing target_metric
    target_metric: str


def generate_differential_candidates(
    ev_squad: list[sq.PlayerCandidate],
    all_candidates: list[sq.PlayerCandidate],
    ownership_fractions: dict[str, float],
    budget: float,
    max_candidates: int = 5,
    max_ev_loss_fraction: float = 0.05,
) -> list[tuple[list[sq.PlayerCandidate], str, str]]:
    """Returns up to `max_candidates` (new_squad, swapped_out_id,
    swapped_in_id) tuples, one swap each, tried in order of the max-EV
    squad's most heavily-owned members first (the players a rank-aware
    manager most wants to differentiate away from)."""
    ev_squad_ids = {p.player_id for p in ev_squad}
    total_ev = sum(p.expected_points * p.availability_probability for p in ev_squad)
    spend = sum(p.price for p in ev_squad)
    club_counts: dict[str, int] = {}
    for p in ev_squad:
        club_counts[p.team] = club_counts.get(p.team, 0) + 1

    owned_sorted = sorted(ev_squad, key=lambda p: -ownership_fractions.get(p.player_id, 0.0))

    out: list[tuple[list[sq.PlayerCandidate], str, str]] = []
    for out_player in owned_sorted:
        if len(out) >= max_candidates:
            break
        budget_after_sale = budget - (spend - out_player.price)
        club_count_without_out = club_counts.get(out_player.team, 0) - 1
        replacements = [
            p for p in all_candidates
            if p.position == out_player.position
            and p.player_id not in ev_squad_ids
            and p.price <= budget_after_sale
            and p.availability_probability > 0.0
            and (p.team != out_player.team or club_count_without_out < sq.MAX_PER_CLUB)
            and (club_counts.get(p.team, 0) + (0 if p.team == out_player.team else 1) <= sq.MAX_PER_CLUB)
        ]
        if not replacements:
            continue
        replacements.sort(key=lambda p: ownership_fractions.get(p.player_id, 1.0))
        for in_player in replacements:
            ev_delta = (out_player.expected_points * out_player.availability_probability) - (in_player.expected_points * in_player.availability_probability)
            if total_ev > 0 and ev_delta / total_ev <= max_ev_loss_fraction:
                new_squad = [p for p in ev_squad if p.player_id != out_player.player_id] + [in_player]
                out.append((new_squad, out_player.player_id, in_player.player_id))
                break
    return out


def score_candidate_squad(
    label: str,
    squad: list[sq.PlayerCandidate],
    swapped_out: str | None,
    swapped_in: str | None,
    sim_results: dict[str, object],
    candidates_meta: dict[str, dict],
    ownership_fractions: dict[str, float],
    n_rivals: int,
    seed: int,
) -> CandidateSquadResult:
    """Simulates one candidate squad's rank/percentile against a fresh
    synthetic field (same seed across candidates, so differences
    between candidates reflect the squad, not field-sampling noise)."""
    xi = sq.select_starting_xi(squad)
    my_samples = np.sum([sim_results[p.player_id].samples for p in xi.starters], axis=0) + sim_results[xi.captain.player_id].samples
    rival_squads = fsim.sample_synthetic_rival_squads(ownership_fractions, candidates_meta, n_rivals=n_rivals, seed=seed)
    field_scores = fsim.simulate_field_scores(rival_squads, sim_results, candidates_meta)
    percentiles = fsim.my_percentile_per_scenario(my_samples, field_scores)
    mean_ev = sum(p.expected_points * p.availability_probability for p in squad)
    return CandidateSquadResult(
        label=label,
        squad_ids=tuple(p.player_id for p in squad),
        swapped_out=swapped_out,
        swapped_in=swapped_in,
        mean_ev=float(mean_ev),
        mean_simulated_score=float(my_samples.mean()),
        mean_percentile=float(percentiles.mean()),
        p_top10pct=float((percentiles >= 0.90).mean()),
        p_top25pct=float((percentiles >= 0.75).mean()),
    )


def select_rank_aware_squad(
    all_candidates: list[sq.PlayerCandidate],
    ownership_fractions: dict[str, float],
    sim_results: dict[str, object],
    candidates_meta: dict[str, dict],
    budget: float = sq.BUDGET,
    n_rivals: int = 2000,
    seed: int = 2026,
    max_candidates: int = 5,
    max_ev_loss_fraction: float = 0.05,
    target_metric: str = "p_top10pct",
) -> RankAwareSelectionResult:
    """The max-EV squad is always candidate zero — a rank-aware
    selection that never beats it on `target_metric` legitimately
    selects the max-EV squad itself, not a forced differential."""
    ev_squad = sq.select_squad(all_candidates, budget=budget)
    ev_result = score_candidate_squad("max_ev", ev_squad, None, None, sim_results, candidates_meta, ownership_fractions, n_rivals, seed)

    diffs = generate_differential_candidates(ev_squad, all_candidates, ownership_fractions, budget, max_candidates, max_ev_loss_fraction)
    diff_results = [
        score_candidate_squad(f"differential_{i}", new_squad, out_id, in_id, sim_results, candidates_meta, ownership_fractions, n_rivals, seed)
        for i, (new_squad, out_id, in_id) in enumerate(diffs)
    ]

    candidates = (ev_result, *diff_results)
    selected = max(candidates, key=lambda r: getattr(r, target_metric))
    return RankAwareSelectionResult(candidates=candidates, selected=selected, target_metric=target_metric)
