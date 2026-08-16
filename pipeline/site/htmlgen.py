"""Central HTML-escaping for the static site builder (Phase 13, Stage 5).

Hand-built HTML plus FPL player/team names (arbitrary, externally-supplied
strings we don't control) is the classic setup for an XSS break. The fix
here is structural, not disciplinary: every dynamic value that reaches a
page goes through render(), and render() escapes every keyword argument
by default. The ONLY way to put unescaped markup on a page is to wrap it
in raw() first -- which is a visible, greppable decision (`grep -rn
"raw(" pipeline/site pipeline/build_site.py`), and should only ever wrap
HTML this codebase assembled itself out of other render()/esc() calls,
never a value that came from the FPL API or a ledger record.
"""
from __future__ import annotations

import html


class Raw(str):
    """A string that is already safe, final HTML. Never construct this
    directly from external data -- only via esc() or raw()."""


def esc(value: object) -> Raw:
    """Escapes `value` for safe inclusion in HTML text or a quoted
    attribute. Idempotent on values already marked Raw (so render() can
    apply it unconditionally to every kwarg without double-escaping
    fragments a caller already built safely)."""
    if isinstance(value, Raw):
        return value
    return Raw(html.escape(str(value), quote=True))


def raw(value: str) -> Raw:
    """Marks `value` as trusted, already-safe HTML, bypassing escaping.
    Use ONLY for HTML assembled from other render()/esc()/raw() calls in
    this codebase -- never for a raw external string (player names, team
    names, free-text fields from a ledger record)."""
    return Raw(value)


def join(*parts: object, sep: str = "") -> Raw:
    """Concatenates `parts`, escaping any part that isn't already Raw."""
    return Raw(esc(sep).join(esc(p) for p in parts))


def render(template: str, **kwargs: object) -> Raw:
    """Fills `{name}` placeholders in `template` (an str.format template).
    Every kwarg is escaped via esc() before substitution -- the only way
    for a caller to inject unescaped markup is to pass a value already
    wrapped in raw(), which is explicit and auditable at the call site.
    `template` itself is always a trusted literal written by this
    codebase, never external data."""
    safe_kwargs = {k: esc(v) for k, v in kwargs.items()}
    return Raw(template.format(**safe_kwargs))
