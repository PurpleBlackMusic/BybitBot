from datetime import datetime, timezone

from bybit_app.utils.timestamps import extract_exec_timestamp, normalise_exec_time


def test_normalise_exec_time_handles_epoch_millis():
    raw = "1690000000000"
    assert normalise_exec_time(raw) == 1690000000000


def test_normalise_exec_time_handles_epoch_seconds_with_fraction():
    raw = "1759762441.934818"
    assert normalise_exec_time(raw) == 1759762441934


def test_normalise_exec_time_handles_epoch_nanos():
    raw = 1690000000000000000
    assert normalise_exec_time(raw) == 1690000000000


def test_normalise_exec_time_handles_iso_timestamp():
    raw = "2024-05-16T12:34:56Z"
    expected = int(datetime(2024, 5, 16, 12, 34, 56, tzinfo=timezone.utc).timestamp() * 1000)
    assert normalise_exec_time(raw) == expected


def test_normalise_exec_time_rejects_invalid_values():
    assert normalise_exec_time("not-a-timestamp") is None
    assert normalise_exec_time(None) is None


def test_extract_exec_timestamp_uses_first_valid_candidate():
    payload = {"execTime": "1690000000000", "ts": 42}
    assert extract_exec_timestamp(payload) == 1690000000000


def test_extract_exec_timestamp_falls_back_to_other_fields():
    now_seconds = 1717000000
    payload = {"execTime": "", "execTimeNs": None, "ts": now_seconds}
    assert extract_exec_timestamp(payload) == now_seconds * 1000


def test_extract_exec_timestamp_returns_none_when_all_fail():
    payload = {"execTime": "", "ts": "bad"}
    assert extract_exec_timestamp(payload) is None
