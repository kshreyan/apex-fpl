from __future__ import annotations

from datetime import datetime, timezone

from pipeline import gw_state as gs

from api_payloads import TEAM_NAMES, make_bootstrap_static, make_event, make_fixture

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)


def _single_gw_fixtures(event=1, kickoff="2026-08-21T14:00:00Z"):
    return [
        make_fixture(1, event, team_h=1, team_a=2, kickoff_time=kickoff),
        make_fixture(2, event, team_h=3, team_a=4, kickoff_time=kickoff),
    ]


def test_pre_deadline_phase():
    bs = make_bootstrap_static([make_event(1, "2026-08-21T17:30:00Z", is_next=True)])
    info = gs.gameweek_phase(bs, _single_gw_fixtures(), gameweek=1, now=NOW)
    assert info.phase == gs.Phase.PRE_DEADLINE
    assert info.hours_until_deadline > 0
    assert info.teams_without_fixture == frozenset()
    assert info.teams_with_double_fixture == frozenset()


def test_in_progress_phase_deadline_passed_not_finished():
    bs = make_bootstrap_static([make_event(1, "2026-08-19T17:30:00Z", finished=False, data_checked=False, is_current=True)])
    info = gs.gameweek_phase(bs, _single_gw_fixtures(kickoff="2026-08-19T14:00:00Z"), gameweek=1, now=NOW)
    assert info.phase == gs.Phase.IN_PROGRESS
    assert info.hours_until_deadline < 0


def test_in_progress_phase_finished_but_not_data_checked():
    """The explicit gap the brief calls out: last kickoff finished, bonus
    not yet confirmed. Must NOT be reported as SETTLED."""
    bs = make_bootstrap_static([make_event(1, "2026-08-19T17:30:00Z", finished=True, data_checked=False, is_current=True)])
    info = gs.gameweek_phase(bs, _single_gw_fixtures(kickoff="2026-08-19T14:00:00Z"), gameweek=1, now=NOW)
    assert info.phase == gs.Phase.IN_PROGRESS
    assert info.finished is True
    assert info.data_checked is False


def test_settled_phase_requires_both_finished_and_data_checked():
    bs = make_bootstrap_static([make_event(1, "2026-08-19T17:30:00Z", finished=True, data_checked=True, is_current=True)])
    info = gs.gameweek_phase(bs, _single_gw_fixtures(kickoff="2026-08-19T14:00:00Z"), gameweek=1, now=NOW)
    assert info.phase == gs.Phase.SETTLED


def test_blank_gameweek_reports_teams_without_fixture():
    # Everton and Fulham have no fixture this gameweek
    bs = make_bootstrap_static([make_event(1, "2026-08-21T17:30:00Z", is_next=True)])
    fixtures = [make_fixture(1, 1, team_h=1, team_a=2, kickoff_time="2026-08-21T14:00:00Z")]
    info = gs.gameweek_phase(bs, fixtures, gameweek=1, now=NOW)
    assert info.teams_without_fixture == frozenset({"Everton", "Fulham"})
    assert info.teams_with_double_fixture == frozenset()


def test_fully_blank_gameweek_all_teams_missing():
    bs = make_bootstrap_static([make_event(1, "2026-08-21T17:30:00Z", is_next=True)])
    info = gs.gameweek_phase(bs, fixtures=[], gameweek=1, now=NOW)
    assert info.teams_without_fixture == frozenset(TEAM_NAMES)


def test_double_gameweek_reports_teams_with_double_fixture():
    # Arsenal plays twice (vs Chelsea, then vs Everton); Fulham has none this gameweek
    bs = make_bootstrap_static([make_event(1, "2026-08-21T17:30:00Z", is_next=True)])
    fixtures = [
        make_fixture(1, 1, team_h=1, team_a=2, kickoff_time="2026-08-21T14:00:00Z"),
        make_fixture(2, 1, team_h=1, team_a=3, kickoff_time="2026-08-24T19:00:00Z"),
    ]
    info = gs.gameweek_phase(bs, fixtures, gameweek=1, now=NOW)
    assert info.teams_with_double_fixture == frozenset({"Arsenal"})
    assert info.teams_without_fixture == frozenset({"Fulham"})


def test_next_prediction_gameweek_skips_an_earlier_unsettled_gameweek():
    """The core subtlety this module was redesigned around: GW1 finished
    but not yet data_checked must NOT block predict.py from correctly
    seeing GW2 as the next gameweek needing a prediction."""
    bs = make_bootstrap_static([
        make_event(1, "2026-08-14T17:30:00Z", finished=True, data_checked=False, is_current=True),
        make_event(2, "2026-08-28T17:30:00Z", is_next=True),
    ])
    assert gs.next_prediction_gameweek(bs, NOW) == 2
    # and GW1's own phase is independently still IN_PROGRESS, not conflated with GW2's state
    info_gw1 = gs.gameweek_phase(bs, [], gameweek=1, now=NOW)
    assert info_gw1.phase == gs.Phase.IN_PROGRESS


def test_end_of_season_returns_none():
    # NOW is 2026-08-20; both deadlines must be genuinely in the past for this
    # to represent "season over" rather than an event with a future deadline
    # that happens to already (nonsensically) claim finished=True.
    bs = make_bootstrap_static([
        make_event(1, "2026-08-07T17:30:00Z", finished=True, data_checked=True),
        make_event(2, "2026-08-14T17:30:00Z", finished=True, data_checked=True),
    ])
    assert gs.next_prediction_gameweek(bs, NOW) is None


def test_cross_check_logs_warning_on_flag_disagreement_but_does_not_raise(caplog):
    # deadline passed and settled, but FPL still (implausibly) flags is_next=True
    bs = make_bootstrap_static([make_event(1, "2026-08-14T17:30:00Z", finished=True, data_checked=True, is_next=True)])
    with caplog.at_level("WARNING"):
        info = gs.gameweek_phase(bs, [], gameweek=1, now=NOW)
    assert info.phase == gs.Phase.SETTLED  # derived from hard facts, not the (disagreeing) flag
    assert any("is_next=True" in r.message for r in caplog.records)
