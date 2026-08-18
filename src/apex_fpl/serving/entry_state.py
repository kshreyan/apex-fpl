"""Real FPL entry state (Phase 13 Block 2.5) -- reads the actual squad
held by this project's real, human-operated FPL entry, via FPL's public
read-only per-entry endpoints. No credentials needed (see CLAUDE.md's
"The real FPL entry" section: this project deliberately never stores
FPL session credentials as a secret) -- `entry/{id}/...` only exposes
what any public tool showing a given entry ID already shows, and a
gameweek's picks only become visible via this endpoint once that
gameweek's deadline has passed (an anti-copying measure by FPL itself,
confirmed live: entry/{id}/event/1/picks/ returns 404 before GW1's
deadline, not partial or provisional data) -- the same "not available
yet, not an error" semantics already used everywhere else pre-deadline
in this project.

ENTRY_ID identifies THIS project's real entry ("Apex FPL", managed by
Kumar Shreyansh, registered from GW1) -- verified live against the
public API on 2026-08-18 (name, player, started_event=1 all confirmed).
Not a secret: an FPL entry ID is a public identifier, the same one
visible in that entry's own public team-page URL. Never treat it as
sensitive or gitignore anything derived from it.

**Sell-price approximation, honestly disclosed, not silently assumed
correct.** FPL's real sell-price rule (configs/seasons/2026_27.yaml's
`transfers.sell_on_fee=0.5`, `element_sell_at_purchase_price=false`):
selling a player who has RISEN in price since purchase only returns the
purchase price plus half the rise (rounded down); a fallen price has no
such protection. Computing this exactly needs each player's actual
purchase price, which requires replaying `entry/{id}/transfers/`
(confirmed public, empty pre-season) alongside a captured pre-deadline
price for the untouched initial squad -- real, buildable work, not
attempted in this first version. `sell_price_by_id` here uses each
player's CURRENT price instead, which slightly OVERSTATES sell value
for risers and is exact for players whose price hasn't moved or has
fallen. Flagged in every transfer recommendation's caveats, not hidden.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from apex_fpl.data import bronze

ENTRY_ID = 4432389

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_ROOT = REPO_ROOT / "data" / "raw"

PICKS_URL_TEMPLATE = "https://fantasy.premierleague.com/api/entry/{entry_id}/event/{gw}/picks/"
HISTORY_URL_TEMPLATE = "https://fantasy.premierleague.com/api/entry/{entry_id}/history/"

# Standard FPL banking rule (configs/seasons/2026_27.yaml's transfers
# section): 1 free transfer granted per gameweek from GW2 onward,
# banking up to this cap. No entry has any free transfers to spend
# before its own first-ever transfer opportunity (GW2).
STARTING_FREE_TRANSFERS = 1
MAX_BANKED_FREE_TRANSFERS = 5


class EntryStateError(Exception):
    """Raised when a captured entry payload doesn't match the shape this
    module depends on, or when state is requested for a gameweek that
    structurally cannot have it yet (e.g. free-transfer count before
    GW2). Same fail-loudly-rather-than-guess discipline as
    pipeline/fpl_client.py's SchemaValidationError."""


@dataclass(frozen=True)
class CurrentSquadState:
    squad_ids: list[str]
    bank: float
    free_transfers: int
    sell_price_by_id: dict[str, float]
    as_of_gw: int  # the most recently settled gameweek this state reflects


def _get_with_retry(url: str) -> requests.Response | None:
    """Same backoff as apex_fpl.data.bronze.fetch_raw, but a 404 is
    returned (not retried, not raised) -- it is this API's own signal
    for "not available yet," a real and expected state for any
    gameweek whose deadline hasn't passed, not a transient failure."""
    last_exc: Exception | None = None
    for attempt in range(bronze.MAX_FETCH_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": bronze.USER_AGENT})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < bronze.MAX_FETCH_ATTEMPTS - 1:
                time.sleep(min(bronze.BACKOFF_BASE_SECONDS * (2 ** attempt), bronze.BACKOFF_CAP_SECONDS))
    assert last_exc is not None
    raise last_exc


def _write_raw_capture(raw_bytes: bytes, source: str, gameweek_dir: int) -> Path:
    out_dir = RAW_DATA_ROOT / f"gw{gameweek_dir:02d}" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}.json"
    suffix = 0
    while path.exists():
        suffix += 1
        path = out_dir / f"{ts}_{suffix}.json"
    path.write_bytes(raw_bytes)
    return path


def validate_entry_picks(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise EntryStateError(f"entry_picks: expected a dict at top level, got {type(payload).__name__}")
    if "picks" not in payload or not isinstance(payload["picks"], list):
        raise EntryStateError("entry_picks.picks: missing or not a list")
    for i, pick in enumerate(payload["picks"]):
        for field in ("element", "position", "multiplier", "is_captain", "is_vice_captain"):
            if field not in pick:
                raise EntryStateError(f"entry_picks.picks[{i}].{field}: missing")
    if "entry_history" not in payload or not isinstance(payload["entry_history"], dict):
        raise EntryStateError("entry_picks.entry_history: missing or not a dict")
    for field in ("bank", "value", "event_transfers", "event_transfers_cost"):
        if field not in payload["entry_history"]:
            raise EntryStateError(f"entry_picks.entry_history.{field}: missing")


def fetch_entry_picks(gw: int, entry_id: int = ENTRY_ID) -> dict | None:
    """Fetches and validates the real entry's picks for gameweek `gw`.
    Returns None if that gameweek's picks aren't available yet (real,
    expected pre-deadline state -- not an error). Writes an immutable
    raw capture on success, matching this project's other raw-capture
    provenance conventions (data/raw/gw{gw:02d}/entry_picks/)."""
    url = PICKS_URL_TEMPLATE.format(entry_id=entry_id, gw=gw)
    resp = _get_with_retry(url)
    if resp is None:
        return None
    payload = json.loads(resp.content)
    validate_entry_picks(payload)
    _write_raw_capture(resp.content, "entry_picks", gw)
    return payload


def fetch_entry_history(entry_id: int = ENTRY_ID) -> dict:
    """Fetches the entry's whole-season history in one call (`current`:
    one entry per settled gameweek this season, with event_transfers/
    event_transfers_cost/bank/value; `past`: prior seasons; `chips`:
    chips played). Always available (confirmed live pre-season: returns
    empty lists, not 404) -- unlike picks, there's nothing to wait for
    here. Writes an immutable raw capture under
    data/raw/gw{as-of-latest-known-gw:02d}/entry_history/ is NOT done
    here since this payload isn't scoped to one gameweek -- callers that
    need provenance for a specific gameweek's free-transfer computation
    should capture picks (which are gameweek-scoped) instead."""
    url = HISTORY_URL_TEMPLATE.format(entry_id=entry_id)
    resp = _get_with_retry(url)
    if resp is None:
        raise EntryStateError(f"entry/{entry_id}/history/ returned 404 -- unexpected, this endpoint should always resolve for a real entry")
    payload = json.loads(resp.content)
    if not isinstance(payload, dict) or "current" not in payload:
        raise EntryStateError("entry_history: expected a dict with a 'current' key")
    return payload


def compute_free_transfers(history_current: list[dict], settled_gw: int) -> int:
    """Replays the standard FPL banking rule (1/gameweek from GW2
    onward, capped at MAX_BANKED_FREE_TRANSFERS, minus whatever was
    actually spent each gameweek) across `history_current` (the
    entry_history.current list) up through `settled_gw`, returning the
    free transfers available going INTO the gameweek after `settled_gw`.
    Same bookkeeping formula already used and tested in
    apex_fpl.optimization.transfers.rolling_horizon_transfers, replayed
    here against REAL entry_history instead of a backtest's own
    self-tracked state -- one formula, two callers, not two
    implementations that could silently drift apart.

    Not computable before any gameweek has settled (there's no GW2 to
    have a free transfer for yet) -- raises EntryStateError rather than
    returning a guessed number."""
    if settled_gw < 1:
        raise EntryStateError(f"cannot compute free transfers before any gameweek has settled (settled_gw={settled_gw})")
    by_gw = {row["event"]: row for row in history_current}
    free_transfers = STARTING_FREE_TRANSFERS
    for gw in range(2, settled_gw + 1):
        row = by_gw.get(gw)
        if row is None:
            raise EntryStateError(f"entry_history.current has no row for gameweek {gw} (needed to replay free-transfer banking through gameweek {settled_gw})")
        transfers_made = row["event_transfers"]
        paid_transfers = max(0, transfers_made - free_transfers)
        free_transfers = min(MAX_BANKED_FREE_TRANSFERS, max(STARTING_FREE_TRANSFERS, free_transfers - transfers_made + paid_transfers + 1))
    return free_transfers


def build_current_squad_state(now_cost_by_element: dict[int, int], entry_id: int = ENTRY_ID) -> CurrentSquadState | None:
    """The single entry point the transfer-recommendation script needs:
    the real entry's current squad, bank, and free transfers, as of the
    most recently settled gameweek. Returns None if no gameweek has
    settled yet for this entry (there is no "current squad" to build a
    transfer recommendation against before GW1 has actually happened --
    this is a real precondition, not a bug to work around).

    `now_cost_by_element`: element_id (int, matching the picks payload's
    `element` field, NOT this project's usual player_id string) -> live
    now_cost (tenths of £m), used for the disclosed sell-price
    approximation -- see module docstring."""
    history = fetch_entry_history(entry_id)
    settled_rows = history["current"]
    if not settled_rows:
        return None
    settled_gw = max(row["event"] for row in settled_rows)

    picks_payload = fetch_entry_picks(settled_gw, entry_id)
    if picks_payload is None:
        raise EntryStateError(f"entry_history reports gameweek {settled_gw} settled, but its picks are still unavailable -- inconsistent API state, investigate before trusting either")

    squad_ids = [str(p["element"]) for p in picks_payload["picks"]]
    free_transfers = compute_free_transfers(settled_rows, settled_gw)
    bank = picks_payload["entry_history"]["bank"] / 10.0
    sell_price_by_id = {
        str(p["element"]): now_cost_by_element[p["element"]] / 10.0
        for p in picks_payload["picks"] if p["element"] in now_cost_by_element
    }
    return CurrentSquadState(squad_ids=squad_ids, bank=bank, free_transfers=free_transfers, sell_price_by_id=sell_price_by_id, as_of_gw=settled_gw)
