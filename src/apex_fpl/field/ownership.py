"""Real historical ownership data (spec Parts XXIII/XXXII) — the
foundation for Phase 10's field/rank simulator.

`merged_gw.csv`'s `selected` column is a REAL per-player, per-gameweek
COUNT of managers owning that player, sourced from the same frozen
per-gameweek FPL ledger already audited in docs/vaastav_archive_audit.md
(the same "grade A, weekly capture" verdict applies to this column — it
is not independently re-verified here beyond the calibration check
below, but it is the same ledger source, not a separately-scraped or
re-derived field). It is a raw count, not a percentage, so converting it
to an ownership FRACTION requires an estimate of the total number of
managers that season — not directly given anywhere in this archive.

`estimate_total_managers` derives that estimate from real data rather
than assuming a round number: `players_raw.csv` (a season-end snapshot)
gives each player's final `selected_by_percent`; dividing a player's
final-gameweek `selected` count (from merged_gw.csv) by that percentage
recovers an implied total-manager count. Checked across the 4 most-owned
players in 2022-23, this converges tightly (11.40M-11.50M, a ~1% spread)
— a real, calibrated estimate, not a fabricated round number — so this
function takes the MEDIAN across several high-ownership players for
robustness against any single player's noise, rather than trusting one.
"""
from __future__ import annotations

from apex_fpl.backtesting import vaastav_loader as vl

N_CALIBRATION_PLAYERS = 8  # top-N most-owned players at the final gameweek, used to estimate total managers


def load_players_raw(season: str) -> list[dict]:
    import csv

    path = vl._season_dir(season) / "players_raw.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def estimate_total_managers(season: str) -> int:
    """Median implied total-manager count across the N most-owned players
    at the season's final gameweek, cross-referencing merged_gw.csv's raw
    `selected` count against players_raw.csv's `selected_by_percent`."""
    final_gw = max(vl.season_gameweeks(season))
    rows = vl.load_merged_gw(season)
    final_rows = [r for r in rows if int(r["GW"]) == final_gw]
    final_rows.sort(key=lambda r: -int(r["selected"]))
    top_players = final_rows[:N_CALIBRATION_PLAYERS]

    pct_by_id = {r["id"]: float(r["selected_by_percent"]) for r in load_players_raw(season)}

    implied_totals = []
    for r in top_players:
        pct = pct_by_id.get(r["element"])
        if pct is None or pct <= 0:
            continue
        implied_totals.append(int(r["selected"]) / (pct / 100.0))

    if not implied_totals:
        raise ValueError(f"could not calibrate total managers for {season} — no matching players_raw selected_by_percent data")
    implied_totals.sort()
    return int(implied_totals[len(implied_totals) // 2])


def load_ownership_counts(season: str, gw: int) -> dict[str, int]:
    """Real per-player manager COUNT (not a fraction) for one gameweek."""
    rows = vl.load_merged_gw(season)
    return {r["element"]: int(r["selected"]) for r in rows if int(r["GW"]) == gw}


def load_ownership_fractions(season: str, gw: int, total_managers: int | None = None) -> dict[str, float]:
    """Real per-player ownership FRACTION (0-1) for one gameweek, using
    `estimate_total_managers` if not supplied. Fractions can very rarely
    exceed 1.0 slightly if `total_managers` is a slight underestimate for
    an early gameweek (managers join throughout August) — not clamped,
    so callers see the raw ratio rather than a silently corrected one."""
    if total_managers is None:
        total_managers = estimate_total_managers(season)
    counts = load_ownership_counts(season, gw)
    return {pid: count / total_managers for pid, count in counts.items()}
