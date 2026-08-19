"""Offline coverage of predict_chips.py's decision logic (evaluate_chip)
-- no live API call, no Monte Carlo simulation. run()'s own data-
fetching orchestration is exercised only by the live dry-run smoke test
documented in the commit, not here (matching predict_transfers.py's own
test-file split: this module's job is the DECISION given a value, not
producing that value)."""
from __future__ import annotations

from pipeline import predict_chips as pc


def _install_tmp_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "LEDGER_DIR", tmp_path / "chip_observations")


def test_window_not_open_for_freehit_at_gw1(monkeypatch, tmp_path):
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    record = pc.evaluate_chip("freehit", target_gw=1, marginal_value=5.0, model_version="abc")

    assert record["decision"] == "WINDOW_NOT_OPEN"
    assert record["marginal_value"] is None
    assert record["window"] is None


def test_no_valuation_available_is_stored_as_none_not_a_fake_zero(monkeypatch, tmp_path):
    """The real bug this guards against: a None marginal_value must
    never be silently treated as an observed zero, which would corrupt
    the 1/e rule's threshold calibration for later gameweeks."""
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    record = pc.evaluate_chip("freehit", target_gw=5, marginal_value=None, model_version="abc")

    assert record["decision"] == "NO_VALUATION_AVAILABLE"
    assert record["marginal_value"] is None


def test_already_played_this_half_blocks_further_evaluation(monkeypatch, tmp_path):
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [{"name": "bboost", "event": 3}])

    record = pc.evaluate_chip("bboost", target_gw=10, marginal_value=8.0, model_version="abc")

    assert record["decision"] == "ALREADY_PLAYED_THIS_HALF"
    assert record["marginal_value"] is None


def test_a_chip_played_in_the_other_half_does_not_block_this_half(monkeypatch, tmp_path):
    """wildcard played at GW3 (half 1) must not block half 2's
    independent wildcard instance -- the real rule grants one per half."""
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [{"name": "bboost", "event": 3}])

    record = pc.evaluate_chip("bboost", target_gw=25, marginal_value=8.0, model_version="abc")

    assert record["decision"] != "ALREADY_PLAYED_THIS_HALF"


def test_observing_during_the_calibration_phase(monkeypatch, tmp_path):
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    record = pc.evaluate_chip("bboost", target_gw=1, marginal_value=5.0, model_version="abc")

    assert record["decision"] == "OBSERVING"
    assert record["window"]["n_observed_including_this_gw"] == 1


def test_sequence_reconstruction_reads_prior_gameweeks_from_the_ledger(monkeypatch, tmp_path):
    """A real, end-to-end check of the ledger round-trip: append several
    low observations (still within/just past the observation phase),
    then a high one that should trigger PLAY_NOW once past it."""
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    # bboost half-1 window is GW1-19 -> r = round(19/e) = 7 observation gameweeks
    low_values = [1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0]  # GWs 1-7, all within/ending the observation phase
    for gw, v in enumerate(low_values, start=1):
        record = pc.evaluate_chip("bboost", target_gw=gw, marginal_value=v, model_version="abc")
        pc._append_ledger("bboost", record)

    high_record = pc.evaluate_chip("bboost", target_gw=8, marginal_value=99.0, model_version="abc")
    assert high_record["decision"] == "PLAY_NOW"
    assert high_record["window"]["n_observed_including_this_gw"] == 8


def test_wait_when_past_observation_but_value_does_not_clear_threshold(monkeypatch, tmp_path):
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    low_values = [5.0, 4.0, 6.0, 3.0, 2.0, 1.0, 6.0]  # threshold = 6.0
    for gw, v in enumerate(low_values, start=1):
        record = pc.evaluate_chip("bboost", target_gw=gw, marginal_value=v, model_version="abc")
        pc._append_ledger("bboost", record)

    record = pc.evaluate_chip("bboost", target_gw=8, marginal_value=5.5, model_version="abc")
    assert record["decision"] == "WAIT"


def test_supersedes_points_at_the_prior_record_for_the_same_gameweek(monkeypatch, tmp_path):
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    first = pc.evaluate_chip("3xc", target_gw=1, marginal_value=4.0, model_version="abc")
    pc._append_ledger("3xc", first)

    second = pc.evaluate_chip("3xc", target_gw=1, marginal_value=4.5, model_version="def")
    assert second["supersedes"] == first["record_id"]


def test_a_missing_gameweek_in_the_sequence_is_skipped_not_treated_as_zero(monkeypatch, tmp_path):
    """If GW3's record for some reason stored marginal_value=None (a
    real NO_VALUATION_AVAILABLE week), it must be excluded from the
    reconstructed sequence entirely, not counted as an observed 0.0."""
    _install_tmp_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(pc.es, "already_played_chips", lambda: [])

    for gw, v in [(1, 5.0), (2, 5.0)]:
        record = pc.evaluate_chip("bboost", target_gw=gw, marginal_value=v, model_version="abc")
        pc._append_ledger("bboost", record)
    # GW3: no valuation available this run
    no_val = pc.evaluate_chip("bboost", target_gw=3, marginal_value=None, model_version="abc")
    pc._append_ledger("bboost", no_val)
    assert no_val["decision"] == "NO_VALUATION_AVAILABLE"

    record = pc.evaluate_chip("bboost", target_gw=4, marginal_value=6.0, model_version="abc")
    # sequence should be [5.0, 5.0, 6.0] (3 observed), NOT [5.0, 5.0, None-as-0, 6.0] (4 observed)
    assert record["window"]["n_observed_including_this_gw"] == 3
