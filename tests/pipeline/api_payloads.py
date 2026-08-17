"""Hand-built, offline FPL-API-shaped payloads for pipeline tests — no
live API call anywhere in this module or in anything that imports it.
Field names and types match pipeline/fpl_client.py's spec, which was
checked directly against a real, already-captured live bootstrap-static/
fixtures payload (see that module's docstring) — these are synthetic
values in a verified-real shape, not a literal saved response, since
hand-maintaining several complete ~600-player JSON blobs for each phase/
edge-case variant would be far more error-prone than a small factory
that can only ever produce internally-consistent payloads.
"""
from __future__ import annotations

import copy

TEAM_NAMES = ["Arsenal", "Chelsea", "Everton", "Fulham"]


def _default_teams():
    return [
        {"id": 1, "name": "Arsenal", "strength_overall_home": 4, "strength_overall_away": 5},
        {"id": 2, "name": "Chelsea", "strength_overall_home": 4, "strength_overall_away": 4},
        {"id": 3, "name": "Everton", "strength_overall_home": 3, "strength_overall_away": 3},
        {"id": 4, "name": "Fulham", "strength_overall_home": 3, "strength_overall_away": 3},
    ]


def _default_element_types():
    return [
        {"id": 1, "squad_select": 2, "squad_min_play": 1, "squad_max_play": 1},
        {"id": 2, "squad_select": 5, "squad_min_play": 3, "squad_max_play": 5},
        {"id": 3, "squad_select": 5, "squad_min_play": 2, "squad_max_play": 5},
        {"id": 4, "squad_select": 3, "squad_min_play": 1, "squad_max_play": 3},
    ]


def make_element(id_, team, element_type, web_name="Player", now_cost=50, selected_by_percent="10.0", status="a", event_points=0,
                  chance_of_playing_this_round=None, chance_of_playing_next_round=None):
    return {
        "id": id_, "web_name": web_name, "team": team, "element_type": element_type,
        "status": status, "now_cost": now_cost, "selected_by_percent": selected_by_percent, "event_points": event_points,
        # None is the real, confirmed default for a healthy ('a') player -- not "unknown".
        # See apex_fpl.serving.live_data.player_availability_probability.
        "chance_of_playing_this_round": chance_of_playing_this_round,
        "chance_of_playing_next_round": chance_of_playing_next_round,
    }


def _default_elements():
    return [make_element(i, team=((i - 1) % 4) + 1, element_type=((i - 1) % 4) + 1) for i in range(1, 9)]


def make_event(id_, deadline_time, finished=False, data_checked=False, is_current=False, is_next=False, average_entry_score=0):
    return {
        "id": id_, "deadline_time": deadline_time, "finished": finished,
        "data_checked": data_checked, "is_current": is_current, "is_next": is_next,
        "average_entry_score": average_entry_score,
    }


def make_bootstrap_static(events, teams=None, elements=None, element_types=None):
    """Every default is a FRESH copy per call -- a caller mutating the
    returned dict (e.g. to test a schema violation) must never leak that
    mutation into a different test via a shared module-level list. This
    was a real bug caught by actually running the suite, not a
    hypothetical: an earlier version returned the same list objects by
    reference, and one test's `del bad["events"][0]["data_checked"]`-style
    mutation silently corrupted every test that ran after it."""
    return {
        "events": copy.deepcopy(events),
        "teams": copy.deepcopy(teams) if teams is not None else _default_teams(),
        "elements": copy.deepcopy(elements) if elements is not None else _default_elements(),
        "element_types": copy.deepcopy(element_types) if element_types is not None else _default_element_types(),
    }


def make_fixture(id_, event, team_h, team_a, kickoff_time, finished=False,
                  team_h_score=None, team_a_score=None, team_h_difficulty=3, team_a_difficulty=3):
    return {
        "id": id_, "event": event, "team_h": team_h, "team_a": team_a, "kickoff_time": kickoff_time,
        "finished": finished, "team_h_score": team_h_score, "team_a_score": team_a_score,
        "team_h_difficulty": team_h_difficulty, "team_a_difficulty": team_a_difficulty,
    }
