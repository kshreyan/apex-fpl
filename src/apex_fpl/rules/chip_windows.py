"""Real 2026/27 chip-window rules (Phase 13 Block 2.7) — the redesign
the earlier ruleset-validity review called for: Phase 9's chip valuation
(`apex_fpl.optimization.chips`) is correct, general-purpose math (given
a sequence of observed EP values, when should a one-shot option be
played) but was only ever demonstrated against a single, OPEN GW2-38
window (that demo's own docstring already discloses this explicitly —
"a methodological demonstration... not a claim about what was actually
legal"). The REAL 2026/27 structure is two independent halves, each
with its own full set of 4 chips (WC/FH/BB/TC), no carryover of an
unused first-half chip into the second — `apply_1e_stopping_rule` must
be applied SEPARATELY per (chip, half) window, not once across a whole
season. This module supplies the correct windows so it can be.

Source of truth: `configs/seasons/2026_27.yaml`'s `chips.inventory`
(parsed directly from the live API's own `chips` array — not
re-derived or guessed here).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ChipWindow:
    name: str  # "wildcard" | "freehit" | "bboost" | "3xc"
    half: int  # 1 or 2
    start_event: int
    stop_event: int
    chip_type: str  # "transfer" | "team"

    def contains(self, gw: int) -> bool:
        return self.start_event <= gw <= self.stop_event

    def gameweeks_remaining(self, gw: int) -> int:
        """0 once the window has closed or hasn't opened yet -- callers
        must check .contains(gw) separately if they need to distinguish
        those two cases from 'the last decidable gameweek'."""
        if not self.contains(gw):
            return 0
        return self.stop_event - gw + 1


def load_chip_windows(season: str = "2026_27") -> list[ChipWindow]:
    path = REPO_ROOT / "configs" / "seasons" / f"{season}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return [
        ChipWindow(name=c["name"], half=c["half"], start_event=c["start_event"], stop_event=c["stop_event"], chip_type=c["chip_type"])
        for c in cfg["chips"]["inventory"]
    ]


def active_window(chip_name: str, gw: int, windows: list[ChipWindow] | None = None, season: str = "2026_27") -> ChipWindow | None:
    """The window for `chip_name` that contains `gw`, if any -- None if
    `gw` falls in neither half's window for that chip (e.g. wildcard at
    GW1, which is real: wildcard/freehit both start_event=2)."""
    windows = windows if windows is not None else load_chip_windows(season)
    for w in windows:
        if w.name == chip_name and w.contains(gw):
            return w
    return None


def half_for_gameweek(gw: int, windows: list[ChipWindow] | None = None, season: str = "2026_27") -> int:
    """1 or 2 -- derived from bboost's own window boundaries (it's the
    one chip usable from GW1 in both halves, so its stop_event cleanly
    marks the half-1/half-2 boundary) rather than hardcoding the GW19/20
    split a second time."""
    windows = windows if windows is not None else load_chip_windows(season)
    bboost_half1 = next(w for w in windows if w.name == "bboost" and w.half == 1)
    return 1 if gw <= bboost_half1.stop_event else 2
