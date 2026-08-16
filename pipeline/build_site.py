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
RESULTS_DIR = REPO_ROOT / "data" / "results"
CALIBRATION_PATH = REPO_ROOT / "data" / "calibration.json"
DOCS_ROOT = REPO_ROOT / "docs"

STALENESS_WARN_HOURS = 36
STALENESS_ESCALATE_HOURS = 72

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
        link = str(esc(label)).join([f'<a href="{href}"{current}>', "</a>"])
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
        '<link rel="stylesheet" href="/assets/site.css">\n'
        "</head>\n"
        f'<body data-rebuilt-at="{esc(rebuilt_at_utc)}">\n'
        '<a class="skip-link" href="#main">Skip to main content</a>\n'
        "<header>\n"
        f'<div class="header-inner"><a class="site-title" href="/">{esc(SITE_TITLE)}</a>'
        f"{_nav(active_path)}</div>\n"
        "</header>\n"
        f'<div id="staleness-banner" class="staleness-banner" hidden></div>\n'
        f'<main id="main">\n{body}\n</main>\n'
        "<footer>\n"
        f'<p class="disclaimer">{esc(DISCLAIMER)}</p>\n'
        f'<p class="build-note">Site last rebuilt {esc(rebuilt_at_utc)} UTC · &copy; {year} APEX FPL</p>\n'
        "</footer>\n"
        '<script src="/assets/staleness.js" defer></script>\n'
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
    links = ", ".join(f'<a href="/gameweek/gw{gw:02d}/">GW{gw}</a>' for gw in missing_gws)
    return raw(
        '<div class="notice notice-critical" role="note">'
        f'{_status_badge("Pipeline gap", "critical")} '
        f"No prediction was ever recorded for: {links}. "
        "This means the automated pipeline failed to run before that gameweek's deadline "
        "— it is not a model decision, and it is not hidden here."
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
        '<p><a href="/">See the full calibration curve and history →</a></p>'
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
        history_rows.append(f'<tr><td><a href="/gameweek/gw{gw:02d}/">GW{gw}</a></td><td>{_status_badge(status.replace("_", " ").title(), kind)}</td></tr>')
    history_rows_html = "".join(history_rows) or '<tr><td colspan="2">No gameweeks yet.</td></tr>'

    body = raw(
        "<h1>The record, not just the picks</h1>"
        "<p class=\"lede\">Every prediction this model has made is committed to git before its "
        "gameweek deadline, and every outcome is scored automatically once the gameweek settles — "
        "hits and misses alike, shown here plainly.</p>"
        f"{_missing_predictions_notice(cov['gameweeks_missing_prediction'] or [])}"
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


def build_current_page(calibration: dict, predictions: dict[int, list[dict]]) -> str:
    record = _record_summary_fragment(calibration)
    if not predictions:
        body = raw(f"{record}<h1>This week</h1><p>No live prediction has been published yet.</p>")
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

    body = raw(
        f"{record}"
        f"<h1>This week: GW{gw}</h1>"
        f'<p class="deadline-note">Deadline: {esc(prediction["deadline_time_utc"])} UTC</p>'
        f"{picks_html}"
    )
    return _page("This week", "This week's live FPL picks from APEX FPL.", "/current/", body, calibration["rebuilt_at_utc"])


def build_gameweek_page(gw: int, prediction: dict | None, result: dict | None, is_missing: bool,
                         pred_commit_url: str | None, result_commit_url: str | None, rebuilt_at_utc: str) -> str:
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

    body = raw("".join(parts))
    return _page(f"Gameweek {gw}", f"APEX FPL's prediction and result for gameweek {gw}.", "", body, rebuilt_at_utc)


def build_methodology_page(rebuilt_at_utc: str) -> str:
    body = raw(
        "<h1>Methodology</h1>"
        "<p>APEX FPL forecasts Fantasy Premier League points using a Monte Carlo simulation "
        "over modelled team goal expectations, per-player minutes and attacking-involvement "
        "shares, and a reduced-form bonus-points model, tournament-selected against simpler "
        "baselines on multi-season historical replay before being trusted in production.</p>"
        "<h2>What it predicts</h2>"
        "<ul><li>Expected points per starting player for the upcoming gameweek</li>"
        "<li>A captaincy pick and the probability the captain scores 6 or more points</li>"
        "<li>A projected total for the selected squad</li></ul>"
        "<h2>What it does not model</h2>"
        "<ul>"
        "<li>Double gameweeks are only partially handled: a team's two fixtures are combined "
        "into one summed-goals match rather than simulated independently — a stated "
        "approximation, not a silent gap.</li>"
        "<li>Rotation risk and press-conference team news are not incorporated intraday; the "
        "model runs once daily against whatever the FPL API currently reflects.</li>"
        "<li>Price changes and transfer-market timing are not modelled — this project scores "
        "forecast accuracy and squad selection, not a transfer strategy.</li>"
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
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --link: #2a78d6;
  --max-width: 880px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --gridline: #2c2c2a; --border: rgba(255,255,255,0.10); --link: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.5;
}
.skip-link { position: absolute; left: -999px; top: 0; background: var(--surface); color: var(--ink); padding: 8px 12px; z-index: 10; }
.skip-link:focus { left: 8px; top: 8px; }
header { background: var(--surface); border-bottom: 1px solid var(--border); }
.header-inner { max-width: var(--max-width); margin: 0 auto; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.site-title { font-weight: 700; text-decoration: none; color: var(--ink); font-size: 1.1rem; }
.nav-list { list-style: none; display: flex; gap: 16px; margin: 0; padding: 0; }
.nav-list a { color: var(--ink-2); text-decoration: none; padding: 4px 2px; }
.nav-list a:hover, .nav-list a:focus { color: var(--link); }
.nav-list a[aria-current="page"] { color: var(--ink); font-weight: 600; border-bottom: 2px solid var(--link); }
main { max-width: var(--max-width); margin: 0 auto; padding: 16px; }
h1 { font-size: 1.6rem; margin-top: 0.2em; }
h2 { font-size: 1.2rem; margin-top: 2em; }
.lede { color: var(--ink-2); font-size: 1.05rem; }
a { color: var(--link); }
.record-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin: 16px 0; }
.stat-row { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }
.stat-tile { min-width: 140px; }
.stat-value { display: block; font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat-label { display: block; color: var(--ink-2); font-size: 0.85rem; }
.data-table { width: 100%; border-collapse: collapse; margin: 12px 0; }
.data-table caption { text-align: left; color: var(--ink-2); font-size: 0.85rem; margin-bottom: 4px; }
.data-table th, .data-table td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--gridline); font-variant-numeric: tabular-nums; }
.data-table { overflow-x: auto; display: block; }
.data-table thead, .data-table tbody { display: table; width: 100%; table-layout: fixed; }
.squad-table .bench-row { color: var(--ink-2); }
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; }
.badge-good { color: var(--good); }
.badge-warning { color: var(--warning); }
.badge-critical { color: var(--critical); }
.badge-muted { color: var(--muted); }
.notice { border-left: 4px solid var(--critical); background: var(--surface); padding: 12px 16px; margin: 16px 0; border-radius: 4px; }
.misses-grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 640px) { .misses-grid { grid-template-columns: 1fr 1fr; } }
.pending { color: var(--muted); font-style: italic; }
.commit-proof { font-size: 0.9rem; }
.commit-proof.pending { color: var(--muted); }
.chart-empty { color: var(--muted); font-style: italic; padding: 24px 0; }
footer { max-width: var(--max-width); margin: 32px auto 0; padding: 16px; color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border); }
.staleness-banner { text-align: center; padding: 8px 16px; font-size: 0.9rem; }
.staleness-banner.staleness-warning { background: var(--warning); color: #1a1a19; }
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
    rebuilt_at_utc = calibration["rebuilt_at_utc"]

    _write(DOCS_ROOT / "index.html", build_homepage(calibration, predictions, results))
    _write(DOCS_ROOT / "current" / "index.html", build_current_page(calibration, predictions))
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
        page = build_gameweek_page(gw, prediction, result, gw in missing, pred_commit_url, result_commit_url, rebuilt_at_utc)
        _write(DOCS_ROOT / "gameweek" / f"gw{gw:02d}" / "index.html", page)

    _write(DOCS_ROOT / "assets" / "site.css", SITE_CSS)
    _write(DOCS_ROOT / "assets" / "staleness.js", STALENESS_JS)

    print(f"Built site into {DOCS_ROOT}: index, current, methodology, {len(all_gws)} gameweek page(s).")


if __name__ == "__main__":
    run()
