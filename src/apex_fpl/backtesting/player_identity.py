"""Cross-season, cross-source player identity resolution via FPL's own
stable `code` field (Phase 13, Block 1.3).

A player's `id` (bootstrap-static) / `element` (merged_gw.csv) is NOT
stable across seasons, or between the live API and the Vaastav
historical archive -- the same real person gets a different number each
season, in both sources, by FPL's own convention. `code`, in contrast,
is FPL's stable per-person identifier and is present in the live
bootstrap-static payload (`elements[].code`) and in every season's
`players_raw.csv` that has been fetched.

Name-matching was tried first, in the field, and found unreliable
before this module existed: a last-name-only join between a live
payload and 2024-25's archive produced false matches from surname
collisions (mapped "Cole Palmer" to an unrelated live goalkeeper also
named Palmer). `code` is the correct join key, not a convenience -- see
the investigation this module resolves in CLAUDE.md / the Block 1
promotion-schedule report.

Only `players_raw.csv` carries `code`; `merged_gw.csv` does not. A
`merged_gw.csv` row needs its own season's `players_raw.csv` to resolve
`code`, via `element == id`. Not every committed season has
`players_raw.csv` -- only where an actual code-based join or
set-piece/ownership feature needs it (see CLAUDE.md's data taxonomy).
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = REPO_ROOT / "data" / "external" / "vaastav"


def _players_raw_path(season: str) -> Path:
    return EXTERNAL_ROOT / season / "players_raw.csv"


def load_code_map(season: str) -> dict[int, dict]:
    """code -> {id, web_name, first_name, second_name} for one season's
    players_raw.csv."""
    path = _players_raw_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist -- players_raw.csv isn't committed for every "
            "season, only where a code-based join or set-piece/ownership feature "
            "actually needs it (see CLAUDE.md's data taxonomy)"
        )
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {
        int(r["code"]): {"id": r["id"], "web_name": r["web_name"], "first_name": r["first_name"], "second_name": r["second_name"]}
        for r in rows
    }


def resolve_live_code_to_season_id(code: int, season: str) -> str | None:
    """Given a player's live, stable `code`, returns their season-
    specific `id` (players_raw.csv) / `element` (merged_gw.csv) value in
    `season`, or None if they have no entry that season -- a real,
    legitimate outcome (didn't play FPL that year, no code assigned),
    not an error."""
    entry = load_code_map(season).get(code)
    return entry["id"] if entry else None


def build_live_to_season_id_map(live_elements: list[dict], season: str) -> dict[str, str]:
    """Batch version for a whole live roster at once: live bootstrap-
    static `elements` (each needs `id` and `code`) -> {live_id:
    season_id}, omitting any live player absent from that season
    entirely rather than raising -- most of a live roster legitimately
    won't have played in an older season (transfers, debuts, retirees)."""
    code_map = load_code_map(season)
    out = {}
    for el in live_elements:
        entry = code_map.get(el["code"])
        if entry is not None:
            out[str(el["id"])] = entry["id"]
    return out
