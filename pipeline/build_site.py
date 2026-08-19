#!/usr/bin/env python3
"""Static site builder (Phase 13, Stage 5) — renders /docs from the
committed ledgers and data/calibration.json. No live fetch: everything
here is read from git-tracked files already on disk by the time this
runs (predict.py, score.py, metrics.py all run earlier in the same
pipeline invocation).

Owns a fixed set of output paths under /docs/ and only ever writes those
-- nothing else in /docs/ is touched, including this project's own
Phase 0-12 research reports that already live there
(docs/phase6_joint_simulation_report.md and siblings):

    index.html                          season scorecard, hero chart
    current/index.html                  this week's live picks
    methodology/index.html              hand-written, static prose
    gameweek/gw{n:02d}/index.html       one per gameweek that has a
                                         prediction, a result, or a
                                         recorded missing-prediction gap
    assets/site.css, assets/staleness.js

Determinism, restated. The original requirement was "same input data ->
byte-identical HTML." That's now "same ledgers + calibration.json + git
state -> byte-identical HTML" -- not a weaker claim, a more precise one.
A gameweek page's commit link depends on `git blame` (see
pipeline/site/git_commits.py), and blame output legitimately changes the
moment a previously-pending line's commit lands: that's real information
becoming available (the record just became independently verifiable),
not noise. Stage 7's healthcheck should treat that as expected, not flag
it as a determinism regression.

Staleness is handled entirely client-side, on purpose. A build-time
staleness check can only ever fire on a build that actually ran -- which
means it can never detect the one failure that matters most (the
scheduled workflow silently stopped firing at all: disabled after 60
days of GitHub inactivity, a YAML error, a runner never dispatching).
Instead this embeds `rebuilt_at_utc` in the page and lets
assets/staleness.js compare it against the *browser's* clock on load --
a browser loading the page days after the last real rebuild is itself
the detector. No JS means no banner; that's an accepted, deliberate
degrade (progressive enhancement), not a coverage gap in the core
content.

A permanently missing prediction (a deadline that passed with nothing
ever recorded for that gameweek) is a different, PERMANENT thing, not a
staleness banner -- it is rendered as a standing fact, at build time, on
that gameweek's own page and in the homepage history, because a silently
missing page reads as a hidden one.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.site import charts, git_commits
from pipeline.site.htmlgen import Raw, esc, raw

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = REPO_ROOT / "data" / "predictions"
TRANSFER_RECOMMENDATIONS_DIR = REPO_ROOT / "data" / "transfer_recommendations"
CHIP_OBSERVATIONS_DIR = REPO_ROOT / "data" / "chip_observations"
EXECUTION_DIVERGENCE_DIR = REPO_ROOT / "data" / "execution_divergence"
RESULTS_DIR = REPO_ROOT / "data" / "results"
CALIBRATION_PATH = REPO_ROOT / "data" / "calibration.json"
DOCS_ROOT = REPO_ROOT / "docs"

STALENESS_WARN_HOURS = 36
STALENESS_ESCALATE_HOURS = 72

# GitHub Pages serves a project site (not a custom domain or a
# username.github.io user site) at https://<user>.github.io/<repo>/ --
# every internal link and asset reference MUST carry this prefix, or it
# resolves against the domain root instead and 404s. Found for real: the
# live site's every link 404'd except the in-page "skip to content"
# anchor, which doesn't leave the page. If this ever moves to a custom
# domain (a bare CNAME, served at the root), set this to "".
SITE_BASE_PATH = "/apex-fpl"

SITE_TITLE = "APEX FPL"
DISCLAIMER = (
    "APEX FPL is an independent research project. It is not affiliated with, "
    "endorsed by, or connected to the Premier League or Fantasy Premier League. "
    "Fantasy Premier League, Premier League, and associated marks are trademarks "
    "of their respective owners."
)

NAV_ITEMS = [
    ("/", "Record"),
    ("/current/", "This week"),
    ("/methodology/", "Methodology"),
]


# ----------------------------------------------------------------- data -----

def _read_ledger_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_predictions() -> dict[int, list[dict]]:
    if not PREDICTIONS_DIR.exists():
        return {}
    return {int(p.stem[2:]): _read_ledger_lines(p) for p in sorted(PREDICTIONS_DIR.glob("gw*.jsonl"))}


def _all_results() -> dict[int, list[dict]]:
    if not RESULTS_DIR.exists():
        return {}
    return {int(p.stem[2:]): _read_ledger_lines(p) for p in sorted(RESULTS_DIR.glob("gw*.jsonl"))}


def _all_transfer_recommendations() -> dict[int, list[dict]]:
    if not TRANSFER_RECOMMENDATIONS_DIR.exists():
        return {}
    return {int(p.stem[2:]): _read_ledger_lines(p) for p in sorted(TRANSFER_RECOMMENDATIONS_DIR.glob("gw*.jsonl"))}


CHIP_DISPLAY_NAMES = {"bboost": "Bench Boost", "3xc": "Triple Captain", "freehit": "Free Hit", "wildcard": "Wildcard"}


def _all_chip_observations() -> dict[str, list[dict]]:
    """One file PER CHIP (data/chip_observations/{chip_name}.jsonl), not
    per gameweek -- see pipeline/predict_chips.py's own docstring for
    why (this ledger's natural key is a time series within a chip's
    window, not a single gameweek's decision)."""
    if not CHIP_OBSERVATIONS_DIR.exists():
        return {}
    return {p.stem: _read_ledger_lines(p) for p in sorted(CHIP_OBSERVATIONS_DIR.glob("*.jsonl"))}


def _all_execution_divergence() -> dict[int, dict]:
    """One record per gameweek, written at most once (see
    pipeline/check_execution_divergence.py's own docstring on why a
    checked gameweek is never re-checked) -- a gameweek missing here
    simply hasn't been checked yet, not a MATCHED result."""
    if not EXECUTION_DIVERGENCE_DIR.exists():
        return {}
    return {int(p.stem[2:]): _read_ledger_lines(p)[-1] for p in sorted(EXECUTION_DIVERGENCE_DIR.glob("gw*.jsonl")) if _read_ledger_lines(p)}


def _url(path: str) -> str:
    """Every internal href/src must go through this -- see
    SITE_BASE_PATH's own comment for why a bare '/foo/' 404s on a GitHub
    Pages project site."""
    return f"{SITE_BASE_PATH}{path}"


def _commit_link_for_latest(ledger_dir: Path, gw: int, lines: list[dict]) -> str | None:
    if not lines:
        return None
    path = ledger_dir / f"gw{gw:02d}.jsonl"
    resolved = git_commits.resolve_commit_shas_for_ledger(path, lines)
    sha = resolved.get(lines[-1]["record_id"])
    return git_commits.commit_url(sha) if sha else None


# ------------------------------------------------------------- html shell ---

def _nav(active_path: str) -> Raw:
    items = []
    for href, label in NAV_ITEMS:
        current = ' aria-current="page"' if href == active_path else ""
        link = str(esc(label)).join([f'<a href="{_url(href)}"{current}>', "</a>"])
        items.append(f"<li>{link}</li>")
    return raw(f'<nav aria-label="Primary"><ul class="nav-list">{"".join(items)}</ul></nav>')


def _page(title: str, description: str, active_path: str, body: Raw, rebuilt_at_utc: str) -> str:
    year = datetime.now(timezone.utc).year
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{esc(title)} — {esc(SITE_TITLE)}</title>\n"
        f'<meta name="description" content="{esc(description)}">\n'
        f'<link rel="stylesheet" href="{_url("/assets/site.css")}">\n'
        "</head>\n"
        f'<body data-rebuilt-at="{esc(rebuilt_at_utc)}">\n'
        '<a class="skip-link" href="#main">Skip to main content</a>\n'
        "<header>\n"
        f'<div class="header-inner"><a class="site-title" href="{_url("/")}">{esc(SITE_TITLE)}</a>'
        f"{_nav(active_path)}</div>\n"
        "</header>\n"
        f'<div id="staleness-banner" class="staleness-banner" hidden></div>\n'
        f'<main id="main">\n{body}\n</main>\n'
        "<footer>\n"
        f'<p class="disclaimer">{esc(DISCLAIMER)}</p>\n'
        f'<p class="build-note">Site last rebuilt {esc(rebuilt_at_utc)} UTC · &copy; {year} APEX FPL</p>\n'
        "</footer>\n"
        f'<script src="{_url("/assets/staleness.js")}" defer></script>\n'
        "</body>\n</html>\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# --------------------------------------------------------------- fragments --

def _status_badge(label: str, kind: str) -> Raw:
    # kind: "good" | "warning" | "critical" | "muted" -- reserved status colors,
    # always icon-equivalent text + label, never colour alone.
    icons = {"good": "✓", "warning": "!", "critical": "✕", "muted": "–"}
    return raw(f'<span class="badge badge-{esc(kind)}">{esc(icons.get(kind, "•"))} {esc(label)}</span>')


def _gw_status(gw: int, prediction: dict | None, result: dict | None, is_missing: bool) -> tuple[str, str]:
    if is_missing:
        return "MISSING_PREDICTION", "critical"
    if prediction is None:
        return "UNKNOWN", "muted"
    if prediction["status"] == "BLANK_GAMEWEEK":
        return "BLANK_GAMEWEEK", "muted"
    if result is None:
        return "PUBLISHED_AWAITING_RESULT", "warning"
    if result["status"] == "SCORED":
        return "SCORED", "good"
    if result["status"] == "BLANK_GAMEWEEK_NO_SCORING":
        return "BLANK_GAMEWEEK", "muted"
    return result["status"], "muted"


def _execution_divergence_section(divergence: dict | None) -> Raw:
    """CLAUDE.md's "real FPL entry: execution is human-in-the-loop"
    rule, second half -- a settled gameweek's real entry picks compared
    against what was published, shown the same way a missing prediction
    already is: as a permanent fact, not corrected or reconciled.
    `divergence` is None when the gameweek hasn't been checked yet
    (nothing to show, not a MATCHED result)."""
    if divergence is None:
        return raw("")
    if divergence["status"] == "MATCHED":
        return raw(
            f'<p class="commit-proof">{_status_badge("Execution matched", "good")} '
            "The real FPL entry's actual picks for this gameweek matched what was published.</p>"
        )
    reasons = []
    if divergence["squad_diverged"]:
        reasons.append("the 15-man squad differs")
    if divergence["captain_diverged"]:
        reasons.append("the captain differs")
    return raw(
        '<div class="notice notice-critical" role="note">'
        f'{_status_badge("Execution diverged", "critical")} '
        "<div>The real FPL entry's actual picks for this gameweek diverged from what was published "
        f"({esc(' and '.join(reasons))}) — a manual execution failure (the transfer or lineup wasn't "
        "entered in time, or was entered differently), not a model decision. This is a permanent "
        "record, not corrected or reconciled.</div></div>"
    )


def _squad_table(squad: dict) -> Raw:
    rows = []
    starters = {p["player_id"] for p in squad["starting_xi"]}
    for p in squad["starting_xi"] + squad["bench_order"]:
        is_starter = p["player_id"] in starters
        role = ""
        if p["player_id"] == squad["captain_player_id"]:
            role = " (C)"
        elif p["player_id"] == squad["vice_captain_player_id"]:
            role = " (VC)"
        rows.append(
            "<tr{cls}><td>{name}{role}</td><td>{pos}</td><td>{team}</td><td>£{price}</td></tr>".format(
                cls=' class="bench-row"' if not is_starter else "",
                name=esc(p["name"]), role=esc(role), pos=esc(p["position"]),
                team=esc(p["team"]), price=esc(f"{p['price']:.1f}"),
            )
        )
    return raw(
        "<table class=\"data-table squad-table\"><caption>Selected squad</caption>"
        "<thead><tr><th scope=\"col\">Player</th><th scope=\"col\">Pos</th>"
        "<th scope=\"col\">Team</th><th scope=\"col\">Price</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _calls_table(calls: list[dict], call_results_by_id: dict[str, dict], squad: dict | None = None) -> Raw:
    names_by_id: dict[str, str] = {}
    if squad:
        for p in squad["starting_xi"] + squad["bench_order"]:
            names_by_id[p["player_id"]] = p["name"]

    rows = []
    for call in calls:
        cr = call_results_by_id.get(call["id"])
        subject = call["subject"]
        kind = subject.get("kind", "")
        player_name = names_by_id.get(subject.get("player_id"))
        if call.get("claim"):
            claim = call["claim"]
        elif kind == "captain" and player_name:
            claim = f"{player_name} (captain, doubled)"
        elif kind == "squad_total":
            claim = "Squad total (projected)"
        else:
            claim = kind
        if call["type"] == "points_forecast":
            predicted = call["value"]
            actual = cr["actual"] if cr else None
            error = cr["error"] if cr else None
            actual_cell = esc(actual) if actual is not None else raw('<span class="pending">not yet scored</span>')
            error_cell = esc(f"{error:+.2f}") if error is not None else esc("–")
        else:
            predicted = f"{call['probability']:.0%}"
            outcome = cr["outcome"] if cr else None
            actual_cell = (esc("Yes") if outcome else esc("No")) if outcome is not None else raw('<span class="pending">not yet scored</span>')
            error_cell = esc("–")
        rows.append(
            "<tr><td>{claim}</td><td>{predicted}</td><td>{actual}</td><td>{error}</td></tr>".format(
                claim=esc(claim), predicted=esc(predicted), actual=actual_cell, error=error_cell,
            )
        )
    return raw(
        "<table class=\"data-table\"><caption>What was called vs. what happened</caption>"
        "<thead><tr><th scope=\"col\">Call</th><th scope=\"col\">Predicted</th>"
        "<th scope=\"col\">Actual</th><th scope=\"col\">Error</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _calibration_bins_table(bins: list[dict]) -> Raw:
    rows = []
    for b in bins:
        lo, hi = b["bin_range"]
        pm = f"{b['predicted_mean']:.0%}" if b["predicted_mean"] is not None else "suppressed (fewer than 10 calls)"
        ar = f"{b['actual_rate']:.0%}" if b["actual_rate"] is not None else "suppressed (fewer than 10 calls)"
        rows.append(
            "<tr><td>{lo:.0%}–{hi:.0%}</td><td>{n}</td><td>{pm}</td><td>{ar}</td></tr>".format(
                lo=lo, hi=hi, n=b["n"], pm=esc(pm), ar=esc(ar),
            )
        )
    return raw(
        "<table class=\"data-table\"><caption>Calibration bins (text equivalent of the chart above)</caption>"
        "<thead><tr><th scope=\"col\">Predicted probability</th><th scope=\"col\">n</th>"
        "<th scope=\"col\">Predicted mean</th><th scope=\"col\">Actual rate</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _missing_predictions_notice(missing_gws: list[int]) -> Raw:
    if not missing_gws:
        return raw("")
    links = ", ".join(f'<a href="{_url(f"/gameweek/gw{gw:02d}/")}">GW{gw}</a>' for gw in missing_gws)
    return raw(
        '<div class="notice notice-critical" role="note">'
        f'{_status_badge("Pipeline gap", "critical")} '
        f"No prediction was ever recorded for: {links}. "
        "This means the automated pipeline failed to run before that gameweek's deadline "
        "— it is not a model decision, and it is not hidden here."
        "</div>"
    )


def _squad_recomputation_caveat() -> Raw:
    """Phase 13 Block 0.2: an honesty caveat, not a bug disclosure. The
    squad shown is recomputed unconstrained every gameweek -- it has no
    memory of a prior squad, so it is not telling you which players to
    transfer in or out, and the points-vs-baselines comparison elsewhere
    on this site does not deduct transfer costs (\"hits\") because the
    model does not take any transfers to get there. Shown on both / and
    /current/ -- see CLAUDE.md's execution-failure-policy section for
    the related standing rule this pairs with."""
    return raw(
        '<div class="notice notice-warning" role="note">'
        f'{_status_badge("Read this first", "warning")} '
        "This squad is recomputed from scratch every gameweek — it has no memory of a "
        "previous squad, so it is not a transfer recommendation (it will not tell you who to "
        "sell or buy) and it is not a playable week-to-week plan. Points shown vs. baselines "
        "elsewhere on this site do not account for transfer costs or hits, because the model "
        "does not model taking any."
        "</div>"
    )


def _biggest_misses_section(misses: dict) -> Raw:
    def _points_row(m: dict) -> str:
        subj = m["subject"]
        label = f"GW{m['gameweek']} · {subj.get('kind')} {subj.get('player_id', '')}".strip()
        error_str = f"{m['error']:+.2f}"
        return f"<li><strong>{esc(label)}</strong> — predicted {esc(m['predicted'])}, actual {esc(m['actual'])} (error {esc(error_str)})</li>"

    def _prob_row(m: dict) -> str:
        subj = m["subject"]
        label = f"GW{m['gameweek']} · captain haul (≥{subj.get('threshold')} pts)"
        outcome = "happened" if m["outcome"] else "did not happen"
        prob_str = f"{m['predicted_probability']:.0%}"
        return f"<li><strong>{esc(label)}</strong> — called {esc(prob_str)} likely, {esc(outcome)}</li>"

    points_items = "".join(_points_row(m) for m in misses["points_forecast"])
    prob_items = "".join(_prob_row(m) for m in misses["binary_probability"])
    if not points_items and not prob_items:
        body = "<p>No scored calls yet — this section populates once gameweeks are scored.</p>"
    else:
        body = (
            "<div class=\"misses-grid\">"
            f'<div><h3>Biggest points misses</h3><ul>{points_items or "<li>None yet.</li>"}</ul></div>'
            f'<div><h3>Biggest confident-and-wrong calls</h3><ul>{prob_items or "<li>None yet.</li>"}</ul></div>'
            "</div>"
        )
    return raw(
        "<section aria-labelledby=\"misses-heading\">"
        '<h2 id="misses-heading">Biggest misses this season</h2>'
        "<p class=\"section-note\">Ranked automatically, every rebuild, by size of error "
        "(points calls) or confidence times wrongness (probability calls). This section "
        "cannot be edited to leave out an embarrassing week.</p>"
        f"{body}</section>"
    )


# ------------------------------------------------------------------ pages --

def _record_summary_fragment(calibration: dict) -> Raw:
    cov = calibration["coverage"]
    prob = calibration["probability_metrics"]
    pts = calibration["points_forecast_metrics"]
    base = calibration["points_vs_baselines"]
    cap = calibration["captaincy"]

    if not cov["gameweeks_scored"]:
        return raw(
            '<section class="record-card" aria-labelledby="record-heading">'
            '<h2 id="record-heading">Season record</h2>'
            "<p>No gameweeks have been scored yet — there is no live track record to show. "
            "The first scored gameweek will populate this section automatically.</p>"
            "</section>"
        )

    brier_str = prob["brier_score"] if prob["brier_score"] is not None else "n/a"
    mae_str = pts["mae"] if pts["mae"] is not None else "n/a"
    diff_str = f"{base['diff_vs_average_manager']:+.0f}"
    hit_rate_str = cap["hit_rate"] if cap["hit_rate"] is not None else f"{cap['hits']}/{cap['n']}"

    return raw(
        '<section class="record-card" aria-labelledby="record-heading">'
        '<h2 id="record-heading">Season record</h2>'
        '<div class="stat-row">'
        f'<div class="stat-tile"><span class="stat-value">{esc(brier_str)}</span><span class="stat-label">Brier score ({esc(prob["n"])} calls)</span></div>'
        f'<div class="stat-tile"><span class="stat-value">{esc(mae_str)}</span><span class="stat-label">Points MAE ({esc(pts["n"])} calls)</span></div>'
        f'<div class="stat-tile"><span class="stat-value">{esc(diff_str)}</span><span class="stat-label">Points vs. average manager</span></div>'
        f'<div class="stat-tile"><span class="stat-value">{esc(hit_rate_str)}</span><span class="stat-label">Captain hit rate</span></div>'
        "</div>"
        f'<p><a href="{_url("/")}">See the full calibration curve and history →</a></p>'
        "</section>"
    )


def build_homepage(calibration: dict, predictions: dict[int, list[dict]], results: dict[int, list[dict]]) -> str:
    cov = calibration["coverage"]
    prob = calibration["probability_metrics"]
    chart = charts.render_calibration_chart(prob["calibration_bins"], prob["n"], prob["brier_score"], calibration["small_sample_policy"]["min_n_for_rate_display"])
    bins_table = _calibration_bins_table(prob["calibration_bins"])
    base_chart = charts.render_baselines_chart(calibration["points_vs_baselines"]["by_gameweek"])

    all_gws = sorted(set(predictions) | set(results) | set(cov["gameweeks_missing_prediction"] or []))
    history_rows = []
    for gw in all_gws:
        pred = predictions.get(gw, [None])[-1] if predictions.get(gw) else None
        res = results.get(gw, [None])[-1] if results.get(gw) else None
        is_missing = gw in (cov["gameweeks_missing_prediction"] or [])
        status, kind = _gw_status(gw, pred, res, is_missing)
        history_rows.append(f'<tr><td><a href="{_url(f"/gameweek/gw{gw:02d}/")}">GW{gw}</a></td><td>{_status_badge(status.replace("_", " ").title(), kind)}</td></tr>')
    history_rows_html = "".join(history_rows) or '<tr><td colspan="2">No gameweeks yet.</td></tr>'

    body = raw(
        '<div class="hero">'
        '<p class="eyebrow">Live, verifiable, in public</p>'
        "<h1>The record, not just the picks</h1>"
        "<p class=\"lede\">Every prediction this model has made is committed to git before its "
        "gameweek deadline, and every outcome is scored automatically once the gameweek settles — "
        "hits and misses alike, shown here plainly.</p>"
        "</div>"
        f"{_missing_predictions_notice(cov['gameweeks_missing_prediction'] or [])}"
        f"{_squad_recomputation_caveat()}"
        f"{_record_summary_fragment(calibration)}"
        '<section aria-labelledby="calib-heading">'
        '<h2 id="calib-heading">Calibration: predicted probability vs. reality</h2>'
        f"{chart}{bins_table}"
        "</section>"
        '<section aria-labelledby="baselines-heading">'
        '<h2 id="baselines-heading">Cumulative points vs. baselines</h2>'
        f"{base_chart}"
        "</section>"
        f"{_biggest_misses_section(calibration['biggest_misses'])}"
        '<section aria-labelledby="history-heading">'
        '<h2 id="history-heading">Gameweek history</h2>'
        f'<table class="data-table"><thead><tr><th scope="col">Gameweek</th><th scope="col">Status</th></tr></thead><tbody>{history_rows_html}</tbody></table>'
        "</section>"
    )
    return _page(
        "Season record",
        "APEX FPL's running Fantasy Premier League forecasting record — calibration, points vs. baselines, and every miss.",
        "/", body, calibration["rebuilt_at_utc"],
    )


def _play_now_chips_for_gw(gw: int, chip_observations: dict[str, list[dict]]) -> list[dict]:
    """The only chip status worth surfacing on this page (Phase 13
    Block 2.8 (a)) -- every other status (OBSERVING, WAIT, WINDOW_NOT_
    OPEN, ALREADY_PLAYED_THIS_HALF, NO_VALUATION_AVAILABLE) is a real,
    expected, low-stakes internal state for most of the season, not
    something worth surfacing as if it were actionable."""
    fired = []
    for chip_name, records in chip_observations.items():
        by_gw = {r["gameweek"]: r for r in records}  # last write per gw wins (append order)
        rec = by_gw.get(gw)
        if rec is not None and rec["decision"] == "PLAY_NOW":
            fired.append(rec)
    return fired


def _this_weeks_action_section(gw: int, transfer_recommendations: dict[int, list[dict]], chip_observations: dict[str, list[dict]]) -> Raw:
    """Phase 13 Block 2.8 (b) — ONE unified "what to do this week"
    story, combining the transfer recommendation (Block 2.5) and any
    chip that just cleared its 1/e stopping-rule threshold (Block 2.8
    (a)), instead of two separately-computed signals competing for a
    visitor's attention next to the full from-scratch squad below.
    Silent (empty string) unless there's a PUBLISHED transfer
    recommendation or a chip fired PLAY_NOW this gameweek -- see
    _play_now_chips_for_gw's own docstring for why every other chip
    status stays silent here."""
    transfer_lines = transfer_recommendations.get(gw)
    transfer_rec = transfer_lines[-1] if transfer_lines and transfer_lines[-1]["status"] == "PUBLISHED" else None
    fired_chips = _play_now_chips_for_gw(gw, chip_observations)

    if transfer_rec is None and not fired_chips:
        return raw("")

    def _player_row(p: dict) -> str:
        return f"<li>{esc(p['name'])} ({esc(p['position'])}, {esc(p['team'])})</li>"

    parts = []
    caveats: list[str] = []

    if fired_chips:
        chip_items = "".join(f"<li><strong>{esc(CHIP_DISPLAY_NAMES.get(c['chip_name'], c['chip_name']))}</strong></li>" for c in fired_chips)
        parts.append(f"<h3>Play this chip</h3><ul>{chip_items}</ul>")
        parts.append(
            "<p class=\"section-note\">Timing decided by a real optimal-stopping rule (the classical 1/e "
            "\"secretary problem\" rule) over this half-season's chip window, not a guess.</p>"
        )

    if transfer_rec is not None:
        r = transfer_rec["recommendation"]
        if not r["transfers_in"]:
            parts.append("<h3>Transfer</h3><p>No transfer recommended this gameweek.</p>")
        else:
            parts.append(
                "<h3>Transfer</h3>"
                f'<div class="misses-grid"><div><h4>In</h4><ul>{"".join(_player_row(p) for p in r["transfers_in"])}</ul></div>'
                f'<div><h4>Out</h4><ul>{"".join(_player_row(p) for p in r["transfers_out"])}</ul></div></div>'
                f"<p>Free transfers available: {esc(r['free_transfers_available'])}. "
                + (f"Hit taken: {esc(r['hit_points'])} points." if r["hit_points"] else "No hit taken.")
                + "</p>"
            )
        caveats.extend(transfer_rec["caveats"])
    elif fired_chips:
        parts.append("<h3>Transfer</h3><p>No transfer recommendation available yet this gameweek.</p>")

    caveats_html = "".join(f"<li>{esc(c)}</li>" for c in caveats)
    caveats_block = f"<p><strong>Scope, disclosed:</strong></p><ul>{caveats_html}</ul>" if caveats else ""
    return raw(
        '<section class="notice notice-warning" aria-labelledby="action-heading">'
        '<h2 id="action-heading">This week\'s recommended action</h2>'
        f"{''.join(parts)}"
        f"{caveats_block}"
        "</section>"
    )


def build_current_page(calibration: dict, predictions: dict[int, list[dict]],
                        transfer_recommendations: dict[int, list[dict]] | None = None,
                        chip_observations: dict[str, list[dict]] | None = None) -> str:
    record = _record_summary_fragment(calibration)
    caveat = _squad_recomputation_caveat()
    if not predictions:
        body = raw(f"{record}{caveat}<h1>This week</h1><p>No live prediction has been published yet.</p>")
        return _page("This week", "This week's live FPL picks from APEX FPL.", "/current/", body, calibration["rebuilt_at_utc"])

    gw = max(predictions)
    lines = predictions[gw]
    prediction = lines[-1]
    commit_url = _commit_link_for_latest(PREDICTIONS_DIR, gw, lines)

    if prediction["status"] == "BLANK_GAMEWEEK":
        picks_html = raw("<p>No fixtures this gameweek for any team — a blank gameweek. No squad was selected.</p>")
    else:
        squad = prediction["squad"]
        commit_note = (
            f'<p class="commit-proof"><a href="{esc(commit_url)}">View the git commit</a> that recorded this prediction, before the GW{gw} deadline.</p>'
            if commit_url else
            '<p class="commit-proof pending">Commit pending — will be recorded in the next automated commit.</p>'
        )
        picks_html = raw(f"{_squad_table(squad)}{commit_note}")

    # Phase 13 Block 2.8 (b): ONE unified "what to do this week" story leads
    # the page, right under the deadline -- the from-scratch squad below is
    # explicitly demoted to reference material, not competing for attention
    # as if it were itself a weekly instruction. See _this_weeks_action_
    # section's own docstring for why it's silent when there's nothing
    # actionable yet (the common case for most of the season so far).
    action_section = _this_weeks_action_section(gw, transfer_recommendations or {}, chip_observations or {})

    body = raw(
        f"{record}"
        f"{caveat}"
        f"<h1>This week: GW{gw}</h1>"
        f'<p class="deadline-note">Deadline: {esc(prediction["deadline_time_utc"])} UTC</p>'
        f"{action_section}"
        '<h2>Reference: squad if building from scratch</h2>'
        '<p class="section-note">Recomputed independently every week, with no memory of any prior squad — '
        'not a transfer plan. If a recommended action is shown above, that is the actual thing to do this week, '
        "not this squad.</p>"
        f"{picks_html}"
    )
    return _page("This week", "This week's live FPL picks from APEX FPL.", "/current/", body, calibration["rebuilt_at_utc"])


def build_gameweek_page(gw: int, prediction: dict | None, result: dict | None, is_missing: bool,
                         pred_commit_url: str | None, result_commit_url: str | None, rebuilt_at_utc: str,
                         divergence: dict | None = None) -> str:
    status, kind = _gw_status(gw, prediction, result, is_missing)
    parts = [f"<h1>Gameweek {gw}</h1>", str(_status_badge(status.replace("_", " ").title(), kind))]

    if is_missing:
        parts.append(
            '<div class="notice notice-critical" role="note">'
            f"No prediction was ever recorded for GW{gw} despite its deadline having passed. "
            "This is a pipeline failure, not a model decision — it is being shown, not hidden."
            "</div>"
        )
    elif prediction is None:
        parts.append("<p>No prediction has been made for this gameweek yet.</p>")
    elif prediction["status"] == "BLANK_GAMEWEEK":
        parts.append("<p>No fixtures this gameweek for any team — a blank gameweek. The model was not invoked.</p>")
    else:
        commit_note = (
            f'<p class="commit-proof"><a href="{esc(pred_commit_url)}">View the git commit</a> that recorded this prediction, before the deadline ({esc(prediction["deadline_time_utc"])} UTC).</p>'
            if pred_commit_url else
            '<p class="commit-proof pending">Commit pending — will be recorded in the next automated commit.</p>'
        )
        parts.append(str(commit_note))
        parts.append(str(_squad_table(prediction["squad"])))

        call_results_by_id = {c["call_id"]: c for c in (result["call_results"] if result and result["status"] == "SCORED" else [])}
        parts.append(str(_calls_table(prediction["calls"], call_results_by_id, prediction["squad"])))

        if result is not None and result["status"] == "SCORED":
            sa = result["squad_actual"]
            bl = result["baselines"]
            result_note = (
                f'<p class="commit-proof"><a href="{esc(result_commit_url)}">View the git commit</a> that recorded this result.</p>'
                if result_commit_url else
                '<p class="commit-proof pending">Result commit pending — will be recorded in the next automated commit.</p>'
            )
            parts.append(
                "<section aria-labelledby=\"outcome-heading\"><h2 id=\"outcome-heading\">Outcome</h2>"
                f"<p>Squad scored {esc(sa['total_points'])} points "
                f"(vs. average manager {esc(bl['average_manager_score'])}, "
                f"template team {esc(bl['template_team_score'])}"
                + (f", top 10k average {esc(bl['top_10k_average'])}" if bl["top_10k_average"] is not None else "")
                + f").</p>{result_note}</section>"
            )
        elif prediction["status"] != "BLANK_GAMEWEEK":
            parts.append('<p class="pending">Not yet scored — this gameweek has not settled.</p>')

        parts.append(str(_execution_divergence_section(divergence)))

    body = raw("".join(parts))
    return _page(f"Gameweek {gw}", f"APEX FPL's prediction and result for gameweek {gw}.", "", body, rebuilt_at_utc)


def build_methodology_page(rebuilt_at_utc: str) -> str:
    body = raw(
        "<h1>Methodology</h1>"
        "<p>APEX FPL forecasts Fantasy Premier League points using a Monte Carlo simulation "
        "over modelled team goal expectations and per-player minutes and attacking-involvement "
        "shares, tournament-selected against simpler baselines on multi-season historical replay "
        "before being trusted in production. It does not currently model bonus points at all — "
        "see the disclosure below, which is not a footnote.</p>"
        '<section class="notice notice-warning" aria-labelledby="bias-heading">'
        '<h2 id="bias-heading">Known, current bias — read this before trusting a projection</h2>'
        "<p>Every points projection on this site assumes <strong>zero bonus points</strong>, for "
        "every player, every gameweek. This is not a rounding simplification: the live model "
        "literally never adds a bonus term. A BPS-aware simulator was built during this project's "
        "research phase, but a blanket promotion was tested and rejected on real evidence — it made "
        "aggregate accuracy worse, not better, because most players win zero bonus in any given "
        "gameweek and the model can only add noise there. Measured directly against a full real "
        "season (2025/26): bonus points are 7.0% of all points scored by players who appeared, and "
        "that share is <strong>highest for forwards (11.4%) and lowest for defenders (5.6%)</strong>, "
        "rising with price at every position. So this bias systematically <em>underrates attacking "
        "players, forwards most of all</em> — not defenders, contrary to an earlier version of this "
        "page.</p>"
        "<p>Separately: 2025/26 introduced Defensive Contribution (DefCon) points, a real scoring "
        "category this model does not compute at all. DefCon specifically rewards defensive actions "
        "(tackles, clearances, blocks, interceptions, recoveries), so — unlike the bonus bias above — "
        "this one does concentrate in defensively-active players, mostly defenders. The raw stats it's "
        "based on are already being captured from the live API but are not yet validated or used "
        "anywhere in this pipeline.</p>"
        "<p><strong>These two biases point in different directions and should not be summed into a "
        "single \"worst position.\"</strong> Missing bonus points underrates forwards and premium "
        "attackers most; missing DefCon underrates defensively-active defenders. Treat both directions "
        "with scepticism rather than assuming one position is uniquely unreliable.</p>"
        "</section>"
        "<h2>What it predicts</h2>"
        "<ul><li>Expected points per starting player for the upcoming gameweek</li>"
        "<li>A captaincy pick and the probability the captain scores 6 or more points</li>"
        "<li>A projected total for the selected squad</li></ul>"
        "<h2>What it does not model</h2>"
        "<ul>"
        "<li>Bonus points and Defensive Contribution points — see the disclosure above.</li>"
        "<li>Double gameweeks are only partially handled: a team's two fixtures are combined "
        "into one summed-goals match rather than simulated independently — a stated "
        "approximation, not a silent gap.</li>"
        "<li>Rotation risk and press-conference team news are not incorporated intraday; the "
        "model runs once daily against whatever the FPL API currently reflects.</li>"
        "<li>Price changes and transfer-market timing are not modelled — this project scores "
        "forecast accuracy and squad selection, not a transfer strategy.</li>"
        "<li>The squad shown is not a transfer plan: it is recomputed from scratch every "
        "gameweek with no memory of a previous squad, so it never tells you who to sell or buy, "
        "and points shown vs. baselines elsewhere on this site do not deduct transfer costs.</li>"
        "</ul>"
        "<h2>Known weaknesses</h2>"
        "<ul>"
        "<li>\"Actual points this gameweek\" is read from the FPL API's most-recently-settled "
        "gameweek field, which is only reliable if scoring runs within days of a gameweek "
        "finishing — a real limitation if the automated pipeline is ever down across more "
        "than one full gameweek cycle.</li>"
        "<li>The top-10k baseline is only available for gameweeks where a separate standings "
        "capture job has run — it will correctly show as unavailable, not zero, for earlier "
        "ones.</li>"
        "</ul>"
        "<h2>Backtests vs. live calls</h2>"
        "<p>Any backfilled or backtested results referenced in this project's research reports "
        "are historical simulation, generated with hindsight, and are never combined with or "
        "presented alongside the live calibration record on this site. The record shown here is "
        "exclusively real predictions, committed before their deadlines.</p>"
    )
    return _page("Methodology", "How the APEX FPL model works, and what it doesn't model.", "/methodology/", body, rebuilt_at_utc)


# ------------------------------------------------------------------- css/js -

SITE_CSS = """
:root {
  --page: #f5f6f8; --surface: #ffffff; --surface-2: #fbfbfd;
  --ink: #0b0e14; --ink-2: #454b57; --muted: #767c89;
  --gridline: #e6e8ec; --border: rgba(11,14,20,0.08); --border-strong: rgba(11,14,20,0.14);
  --good: #0ca30c; --good-soft: #e6f6e6;
  --warning: #b3760a; --warning-soft: #fdf1d8;
  --critical: #c22a2a; --critical-soft: #fce9e9;
  --muted-soft: #eef0f3;
  --brand: #2a78d6; --brand-ink: #ffffff; --brand-dark: #184f95;
  --shadow-sm: 0 1px 2px rgba(11,14,20,0.06);
  --shadow-md: 0 4px 14px rgba(11,14,20,0.07), 0 1px 3px rgba(11,14,20,0.06);
  --shadow-lg: 0 12px 32px rgba(11,14,20,0.10), 0 2px 6px rgba(11,14,20,0.06);
  --radius-sm: 8px; --radius-md: 14px; --radius-lg: 20px;
  --max-width: 1080px;
  --space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px;
  --space-5: 24px; --space-6: 32px; --space-7: 48px; --space-8: 64px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0a0c10; --surface: #14171d; --surface-2: #191c23;
    --ink: #f4f5f7; --ink-2: #c3c7d1; --muted: #8b909c;
    --gridline: #262a33; --border: rgba(255,255,255,0.08); --border-strong: rgba(255,255,255,0.14);
    --good: #34c759; --good-soft: rgba(52,199,89,0.14);
    --warning: #f0a83a; --warning-soft: rgba(240,168,58,0.14);
    --critical: #f0554f; --critical-soft: rgba(240,85,79,0.14);
    --muted-soft: rgba(255,255,255,0.05);
    --brand: #4c93e8; --brand-ink: #051225; --brand-dark: #9cc4f2;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 14px rgba(0,0,0,0.35), 0 1px 3px rgba(0,0,0,0.3);
    --shadow-lg: 0 16px 40px rgba(0,0,0,0.45), 0 2px 8px rgba(0,0,0,0.3);
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.55; -webkit-font-smoothing: antialiased;
}
.skip-link { position: absolute; left: -999px; top: 0; background: var(--brand); color: var(--brand-ink); padding: 10px 16px; z-index: 20; border-radius: 0 0 var(--radius-sm) 0; font-weight: 600; }
.skip-link:focus { left: 0; top: 0; }

header {
  background: var(--surface); /* fallback for browsers without color-mix() */
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter: saturate(180%) blur(12px); -webkit-backdrop-filter: saturate(180%) blur(12px);
  border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10;
}
.header-inner { max-width: var(--max-width); margin: 0 auto; padding: var(--space-3) var(--space-5); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: var(--space-3); }
.site-title { font-weight: 800; text-decoration: none; color: var(--ink); font-size: 1.15rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: 8px; }
.site-title::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; background: linear-gradient(135deg, var(--brand), #1baf7a); }
.nav-list { list-style: none; display: flex; gap: var(--space-1); margin: 0; padding: 0; }
.nav-list a { color: var(--ink-2); text-decoration: none; padding: 8px 14px; border-radius: 999px; font-weight: 600; font-size: 0.92rem; transition: background 0.15s ease, color 0.15s ease; }
.nav-list a:hover, .nav-list a:focus-visible { color: var(--ink); background: var(--muted-soft); }
.nav-list a[aria-current="page"] { color: var(--brand-ink); background: var(--brand); }

main { max-width: var(--max-width); margin: 0 auto; padding: var(--space-6) var(--space-5) var(--space-8); }

h1 { font-size: clamp(1.9rem, 1.5rem + 1.6vw, 2.7rem); line-height: 1.1; letter-spacing: -0.02em; margin: 0 0 var(--space-3); font-weight: 800; }
h2 { font-size: 1.35rem; letter-spacing: -0.01em; margin: 0 0 var(--space-4); font-weight: 750; }
h3 { font-size: 1.05rem; margin: 0 0 var(--space-2); font-weight: 700; }
p { margin: 0 0 var(--space-3); }
a { color: var(--brand); text-underline-offset: 3px; }
a:hover { text-decoration-thickness: 2px; }

.eyebrow { text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.78rem; font-weight: 700; color: var(--brand); margin: 0 0 var(--space-2); }
.hero { padding: var(--space-6) 0 var(--space-5); }
.lede { color: var(--ink-2); font-size: 1.12rem; max-width: 62ch; margin: 0; }

/* Every top-level content block is a card by default -- no per-call CSS
   class needed in build_site.py's Python, since every one of them is
   already a <section> or one of these specific containers. */
main > section, .record-card, .notice {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
  padding: var(--space-5); margin: 0 0 var(--space-5); box-shadow: var(--shadow-sm);
}
main > section:hover, .record-card:hover { box-shadow: var(--shadow-md); }
main > section, .record-card, .notice { transition: box-shadow 0.2s ease; }

.record-card { background: linear-gradient(165deg, var(--surface) 0%, var(--surface-2) 100%); }
.record-card h2 { margin-bottom: var(--space-4); }

.stat-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--space-4); margin: 0 0 var(--space-4); }
@media (min-width: 640px) { .stat-row { grid-template-columns: repeat(4, 1fr); } }
.stat-tile {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: var(--space-4); position: relative; overflow: hidden;
}
.stat-tile::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--brand); }
.stat-value { display: block; font-size: clamp(1.5rem, 1.2rem + 1vw, 2.1rem); font-weight: 800; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; line-height: 1.15; }
.stat-label { display: block; color: var(--muted); font-size: 0.82rem; margin-top: var(--space-1); font-weight: 600; }

.data-table { width: 100%; border-collapse: collapse; margin: 0; font-size: 0.94rem; overflow-x: auto; display: block; border: 1px solid var(--gridline); border-radius: var(--radius-sm); }
.data-table thead, .data-table tbody { display: table; width: 100%; table-layout: fixed; }
.data-table caption { text-align: left; color: var(--muted); font-size: 0.8rem; padding: var(--space-2) var(--space-3); border-bottom: 1px solid var(--gridline); background: var(--surface-2); caption-side: top; }
.data-table th { text-align: left; padding: 10px var(--space-3); background: var(--surface-2); color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700; border-bottom: 1px solid var(--gridline); }
.data-table td { text-align: left; padding: 10px var(--space-3); border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover td { background: var(--surface-2); }
.squad-table .bench-row td { color: var(--muted); }

.badge { display: inline-flex; align-items: center; gap: 5px; padding: 4px 11px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
.badge-good { color: var(--good); background: var(--good-soft); }
.badge-warning { color: var(--warning); background: var(--warning-soft); }
.badge-critical { color: var(--critical); background: var(--critical-soft); }
.badge-muted { color: var(--muted); background: var(--muted-soft); }

.notice { border: 1px solid var(--critical-soft); background: var(--critical-soft); display: flex; gap: var(--space-3); align-items: flex-start; }
.notice .badge { flex-shrink: 0; }
.notice-warning { border-color: var(--warning-soft); background: var(--warning-soft); }

.section-note { color: var(--muted); font-size: 0.9rem; margin-top: calc(var(--space-4) * -1 + var(--space-2)); margin-bottom: var(--space-4); }
.misses-grid { display: grid; grid-template-columns: 1fr; gap: var(--space-5); }
@media (min-width: 640px) { .misses-grid { grid-template-columns: 1fr 1fr; } }
.misses-grid h3, .misses-grid h4 { color: var(--ink-2); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }
.misses-grid ul { margin: 0; padding-left: 1.1em; }
.misses-grid li { margin-bottom: var(--space-2); }

.pending { color: var(--muted); font-style: italic; }
.commit-proof { font-size: 0.88rem; color: var(--muted); }
.commit-proof a { font-weight: 600; }
.commit-proof.pending { font-style: italic; }
.chart-empty { color: var(--muted); font-style: italic; padding: var(--space-6) 0; text-align: center; }
.deadline-note { color: var(--muted); font-size: 0.92rem; margin-top: calc(var(--space-3) * -1); }

footer { max-width: var(--max-width); margin: var(--space-7) auto 0; padding: var(--space-5); color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border); }
footer p { margin: 0 0 var(--space-2); }
footer p:last-child { margin-bottom: 0; }

.staleness-banner { text-align: center; padding: var(--space-3) var(--space-4); font-size: 0.9rem; font-weight: 600; }
.staleness-banner.staleness-warning { background: var(--warning-soft); color: var(--warning); }
.staleness-banner.staleness-critical { background: var(--critical); color: #ffffff; }
"""

STALENESS_JS = f"""
(function () {{
  var WARN_HOURS = {STALENESS_WARN_HOURS};
  var ESCALATE_HOURS = {STALENESS_ESCALATE_HOURS};
  var banner = document.getElementById("staleness-banner");
  var body = document.body;
  if (!banner || !body || !body.dataset.rebuiltAt) return;
  var rebuiltAt = new Date(body.dataset.rebuiltAt);
  if (isNaN(rebuiltAt.getTime())) return;
  var hours = (Date.now() - rebuiltAt.getTime()) / 3600000;
  if (hours >= ESCALATE_HOURS) {{
    banner.textContent = "Data has not updated in over " + Math.floor(hours) + " hours. The automated pipeline may have stopped running.";
    banner.className = "staleness-banner staleness-critical";
    banner.hidden = false;
  }} else if (hours >= WARN_HOURS) {{
    banner.textContent = "Data was last updated " + Math.floor(hours) + " hours ago.";
    banner.className = "staleness-banner staleness-warning";
    banner.hidden = false;
  }}
}})();
"""


# ------------------------------------------------------------------- build --

def run() -> None:
    if not CALIBRATION_PATH.exists():
        raise RuntimeError(f"{CALIBRATION_PATH} does not exist — run pipeline/metrics.py first. Refusing to build a site with no calibration data rather than rendering a misleading empty one.")
    calibration = json.loads(CALIBRATION_PATH.read_text())

    predictions = _all_predictions()
    results = _all_results()
    transfer_recommendations = _all_transfer_recommendations()
    chip_observations = _all_chip_observations()
    execution_divergence = _all_execution_divergence()
    rebuilt_at_utc = calibration["rebuilt_at_utc"]

    _write(DOCS_ROOT / "index.html", build_homepage(calibration, predictions, results))
    _write(DOCS_ROOT / "current" / "index.html", build_current_page(calibration, predictions, transfer_recommendations, chip_observations))
    _write(DOCS_ROOT / "methodology" / "index.html", build_methodology_page(rebuilt_at_utc))

    missing = set(calibration["coverage"]["gameweeks_missing_prediction"] or [])
    all_gws = sorted(set(predictions) | set(results) | missing)
    for gw in all_gws:
        pred_lines = predictions.get(gw, [])
        result_lines = results.get(gw, [])
        prediction = pred_lines[-1] if pred_lines else None
        result = result_lines[-1] if result_lines else None
        pred_commit_url = _commit_link_for_latest(PREDICTIONS_DIR, gw, pred_lines) if pred_lines else None
        result_commit_url = _commit_link_for_latest(RESULTS_DIR, gw, result_lines) if result_lines else None
        page = build_gameweek_page(gw, prediction, result, gw in missing, pred_commit_url, result_commit_url, rebuilt_at_utc, execution_divergence.get(gw))
        _write(DOCS_ROOT / "gameweek" / f"gw{gw:02d}" / "index.html", page)

    _write(DOCS_ROOT / "assets" / "site.css", SITE_CSS)
    _write(DOCS_ROOT / "assets" / "staleness.js", STALENESS_JS)

    print(f"Built site into {DOCS_ROOT}: index, current, methodology, {len(all_gws)} gameweek page(s).")


if __name__ == "__main__":
    run()
