"""SVG chart builders for the static site (Phase 13, Stage 5).

Hand-computed inline SVG, not <canvas>/JS -- real DOM (<title>/<desc> for
screen readers), renders with zero script, matches "loads fast on a bad
connection." Every chart here is paired, by build_site.py, with a plain
HTML <table> carrying the same numbers -- the text-equivalent lives next
to the chart, not inside it.

Colors are the dataviz skill's validated reference palette
(references/palette.md), used unmodified: categorical slots in their
fixed documented order (1=blue, 2=orange, 3=aqua, 4=yellow), chart-chrome
grays for axes/gridlines/reference lines. This is the reference instance,
not a re-derived one, so it did not need re-validation -- only a
substituted brand palette would.
"""
from __future__ import annotations

from pipeline.site.htmlgen import Raw, esc, raw

# Reference palette (references/palette.md), light-mode values.
SERIES = {
    "model": "#2a78d6",           # slot 1: blue
    "average_manager": "#eb6834",  # slot 2: orange
    "template_team": "#1baf7a",    # slot 3: aqua
    "top_10k": "#eda100",          # slot 4: yellow
}
COLOR_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_PRIMARY = "#0b0b0b"

_STYLE = """
<style>
.chart-svg { max-width: 100%; height: auto; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.chart-svg text { fill: var(--chart-muted, #898781); }
.chart-axis-label { fill: var(--chart-text-secondary, #52514e); font-size: 12px; }
.chart-title-label { fill: var(--chart-text-primary, #0b0b0b); font-size: 13px; font-weight: 600; }
@media (prefers-color-scheme: dark) {
  .chart-svg { --chart-muted: #898781; --chart-text-secondary: #c3c2b7; --chart-text-primary: #ffffff; }
  .chart-svg .chart-gridline { stroke: #2c2c2a; }
  .chart-svg .chart-baseline { stroke: #383835; }
}
</style>
"""


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class _Scale:
    def __init__(self, domain: tuple[float, float], px_range: tuple[float, float]):
        self.d0, self.d1 = domain
        self.r0, self.r1 = px_range

    def __call__(self, value: float) -> float:
        span = (self.d1 - self.d0) or 1.0
        t = (value - self.d0) / span
        return _lerp(self.r0, self.r1, t)


def render_calibration_chart(calibration_bins: list[dict], n_total: int, brier_score: float | None, min_n: int) -> Raw:
    """A reliability plot: x = predicted probability, y = actual outcome
    rate, one point per non-suppressed bin, against a y=x reference line.
    Suppressed bins (n < min_n) are rendered as small hollow markers
    pinned to the axis instead of being silently omitted -- the small-
    sample honesty this project committed to needs to stay visible even
    when there isn't yet a rate to plot."""
    if n_total == 0:
        return raw(
            '<div class="chart-empty" role="img" '
            'aria-label="No scored probability calls yet">'
            "No scored predictions yet — the calibration curve will appear here "
            "once the first gameweek is scored."
            "</div>"
        )

    W, H = 560, 400
    ML, MR, MT, MB = 56, 20, 24, 44
    plot_w, plot_h = W - ML - MR, H - MT - MB
    sx = _Scale((0.0, 1.0), (ML, ML + plot_w))
    sy = _Scale((0.0, 1.0), (MT + plot_h, MT))  # inverted: y grows upward

    parts: list[str] = [_STYLE]
    parts.append(
        f'<svg class="chart-svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-labelledby="calib-title calib-desc" xmlns="http://www.w3.org/2000/svg">'
    )
    parts.append(str(esc(f"Model calibration: predicted vs. actual — Brier score {brier_score if brier_score is not None else 'n/a'}")).join([
        '<title id="calib-title">', '</title>',
    ]))
    parts.append(str(esc(
        "Reliability plot. X axis: predicted probability, binned in tenths. "
        "Y axis: actual rate the outcome happened. Points on the diagonal "
        "reference line mean the model's stated confidence matched reality."
    )).join(['<desc id="calib-desc">', '</desc>']))

    # gridlines + axis ticks at 0, 0.25, 0.5, 0.75, 1.0
    for i in range(5):
        v = i / 4
        x, y = sx(v), sy(v)
        parts.append(f'<line class="chart-gridline" x1="{x:.1f}" y1="{MT:.1f}" x2="{x:.1f}" y2="{MT+plot_h:.1f}" stroke="{COLOR_GRIDLINE}" stroke-width="1"/>')
        parts.append(f'<line class="chart-gridline" x1="{ML:.1f}" y1="{y:.1f}" x2="{ML+plot_w:.1f}" y2="{y:.1f}" stroke="{COLOR_GRIDLINE}" stroke-width="1"/>')
        parts.append(f'<text class="chart-axis-label" x="{x:.1f}" y="{MT+plot_h+18:.1f}" text-anchor="middle">{v:.2f}</text>')
        parts.append(f'<text class="chart-axis-label" x="{ML-8:.1f}" y="{y+4:.1f}" text-anchor="end">{v:.2f}</text>')

    parts.append(f'<line class="chart-baseline" x1="{ML}" y1="{MT+plot_h}" x2="{ML+plot_w}" y2="{MT+plot_h}" stroke="{COLOR_BASELINE}" stroke-width="1.5"/>')
    parts.append(f'<line class="chart-baseline" x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+plot_h}" stroke="{COLOR_BASELINE}" stroke-width="1.5"/>')

    # y = x reference line, dashed, recessive
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}" '
                  f'stroke="{COLOR_MUTED}" stroke-width="1.5" stroke-dasharray="4 4"/>')
    parts.append(f'<text class="chart-axis-label" x="{sx(0.72):.1f}" y="{sy(0.78):.1f}" transform="rotate(-38 {sx(0.72):.1f} {sy(0.78):.1f})">perfectly calibrated</text>')

    plotted = [b for b in calibration_bins if not b["suppressed"]]
    plotted.sort(key=lambda b: b["predicted_mean"])
    if len(plotted) >= 2:
        pts = " ".join(f"{sx(b['predicted_mean']):.1f},{sy(b['actual_rate']):.1f}" for b in plotted)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{SERIES["model"]}" stroke-width="2"/>')
    for b in plotted:
        cx, cy = sx(b["predicted_mean"]), sy(b["actual_rate"])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{SERIES["model"]}"/>')

    suppressed = [b for b in calibration_bins if b["suppressed"]]
    for b in suppressed:
        mid = sum(b["bin_range"]) / 2
        cx = sx(mid)
        cy = MT + plot_h + 30
        parts.append(f'<path d="M {cx-4:.1f} {cy:.1f} L {cx:.1f} {cy-4:.1f} L {cx+4:.1f} {cy:.1f} L {cx:.1f} {cy+4:.1f} Z" '
                      f'fill="none" stroke="{COLOR_MUTED}" stroke-width="1.5"/>')
    if suppressed:
        parts.append(f'<text class="chart-axis-label" x="{ML:.1f}" y="{MT+plot_h+42:.1f}">'
                      f'◇ = bin has fewer than {min_n} calls (rate suppressed, not shown)</text>')

    legend_y = 14
    parts.append(f'<circle cx="{ML+6}" cy="{legend_y}" r="5" fill="{SERIES["model"]}"/>')
    parts.append(f'<text class="chart-title-label" x="{ML+16}" y="{legend_y+4}">Model</text>')
    parts.append("</svg>")
    return raw("".join(parts))


def render_baselines_chart(by_gameweek: list[dict]) -> Raw:
    """Cumulative season points: model vs. the three baselines, one line
    per series in fixed categorical order. top_10k is only drawn over the
    contiguous stretch of gameweeks where the standings capture has
    actually run -- a gap is left visibly blank rather than interpolated
    or backfilled."""
    if not by_gameweek:
        return raw(
            '<div class="chart-empty" role="img" '
            'aria-label="No scored gameweeks yet">'
            "No scored gameweeks yet — cumulative points will appear here "
            "once the first gameweek is scored."
            "</div>"
        )

    gws = [row["gameweek"] for row in by_gameweek]
    series_defs = [("model", "Model"), ("average_manager", "Average manager"), ("template_team", "Template team"), ("top_10k", "Top 10k average")]

    cumulative: dict[str, list[tuple[int, float]]] = {}
    for key, _ in series_defs:
        running = 0.0
        pts: list[tuple[int, float]] = []
        for row, gw in zip(by_gameweek, gws):
            val = row.get(key)
            if val is None:
                continue
            running += val
            pts.append((gw, running))
        cumulative[key] = pts

    all_values = [v for pts in cumulative.values() for _, v in pts]
    if not all_values:
        return raw('<div class="chart-empty">No scored gameweeks yet.</div>')

    y_min, y_max = min(0.0, min(all_values)), max(all_values)
    if y_max == y_min:
        y_max += 1.0

    W, H = 560, 400
    ML, MR, MT, MB = 56, 20, 24, 44
    plot_w, plot_h = W - ML - MR, H - MT - MB
    sx = _Scale((min(gws), max(gws)) if len(gws) > 1 else (min(gws) - 0.5, max(gws) + 0.5), (ML, ML + plot_w))
    sy = _Scale((y_min, y_max), (MT + plot_h, MT))

    parts: list[str] = [_STYLE]
    parts.append(f'<svg class="chart-svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="base-title base-desc" xmlns="http://www.w3.org/2000/svg">')
    parts.append(str(esc("Cumulative season points: model vs. baselines")).join(['<title id="base-title">', '</title>']))
    parts.append(str(esc(
        "Line chart. X axis: gameweek. Y axis: cumulative points scored so far this season, "
        "one line per series: model, average manager, template team, top 10k average."
    )).join(['<desc id="base-desc">', '</desc>']))

    n_y_ticks = 5
    for i in range(n_y_ticks):
        v = y_min + (y_max - y_min) * i / (n_y_ticks - 1)
        y = sy(v)
        parts.append(f'<line class="chart-gridline" x1="{ML}" y1="{y:.1f}" x2="{ML+plot_w}" y2="{y:.1f}" stroke="{COLOR_GRIDLINE}" stroke-width="1"/>')
        parts.append(f'<text class="chart-axis-label" x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{v:.0f}</text>')
    for gw in gws:
        x = sx(gw)
        parts.append(f'<text class="chart-axis-label" x="{x:.1f}" y="{MT+plot_h+18}" text-anchor="middle">{gw}</text>')

    parts.append(f'<line class="chart-baseline" x1="{ML}" y1="{MT+plot_h}" x2="{ML+plot_w}" y2="{MT+plot_h}" stroke="{COLOR_BASELINE}" stroke-width="1.5"/>')
    parts.append(f'<line class="chart-baseline" x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+plot_h}" stroke="{COLOR_BASELINE}" stroke-width="1.5"/>')

    legend_x = ML
    for key, label in series_defs:
        pts = cumulative[key]
        color = SERIES[key]
        if len(pts) >= 2:
            poly = " ".join(f"{sx(gw):.1f},{sy(v):.1f}" for gw, v in pts)
            parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>')
        for gw, v in pts:
            parts.append(f'<circle cx="{sx(gw):.1f}" cy="{sy(v):.1f}" r="4" fill="{color}"/>')
        if pts:
            parts.append(f'<circle cx="{legend_x+6}" cy="{H-6}" r="5" fill="{color}"/>')
            parts.append(str(esc(label)).join([f'<text class="chart-title-label" x="{legend_x+16}" y="{H-2}">', "</text>"]))
            legend_x += 24 + 9 * len(label)

    parts.append("</svg>")
    return raw("".join(parts))
