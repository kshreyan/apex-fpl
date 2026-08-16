from __future__ import annotations

from pipeline.site import htmlgen


def test_esc_escapes_html_special_characters():
    out = htmlgen.esc("<script>alert(1)</script>")
    assert "<script>" not in out
    assert out == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_esc_escapes_quotes_for_attribute_safety():
    out = htmlgen.esc('"><img src=x onerror=alert(1)>')
    assert "<img" not in out
    assert "&quot;" in out
    assert "&gt;" in out


def test_esc_is_idempotent_on_already_raw_values():
    trusted = htmlgen.raw("<b>bold</b>")
    assert htmlgen.esc(trusted) == "<b>bold</b>"


def test_render_escapes_every_kwarg_by_default():
    out = htmlgen.render("<div>{name}</div>", name="<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_does_not_escape_a_value_explicitly_wrapped_in_raw():
    trusted_fragment = htmlgen.raw("<b>Arsenal</b>")
    out = htmlgen.render("<div>{content}</div>", content=trusted_fragment)
    assert "<b>Arsenal</b>" in out


def test_render_handles_a_realistic_hostile_fpl_name():
    # a real, if unlikely, case: an FPL display/team name containing markup --
    # this must never reach the page unescaped.
    out = htmlgen.render("<td>{name}</td>", name='Bukayo <img src=x onerror=alert(1)> Saka')
    assert "<img" not in out
    assert "&lt;img" in out


def test_join_escapes_plain_parts_but_not_raw_parts():
    out = htmlgen.join("<a>", htmlgen.raw("<b>"), "<c>")
    assert out == "&lt;a&gt;<b>&lt;c&gt;"


def test_render_output_is_a_raw_string_reusable_without_double_escaping():
    inner = htmlgen.render("<span>{v}</span>", v="A & B")
    outer = htmlgen.render("<div>{inner}</div>", inner=inner)
    assert outer == "<div><span>A &amp; B</span></div>"
