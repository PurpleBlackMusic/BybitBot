from __future__ import annotations

import json

from bybit_app.utils import log as log_module


def _use_temp_log(tmp_path, monkeypatch):
    log_file = tmp_path / "app.log"
    monkeypatch.setattr(log_module, "LOG_FILE", log_file)
    return log_file


def test_read_tail_returns_empty_for_missing_file(tmp_path, monkeypatch):
    _use_temp_log(tmp_path, monkeypatch)

    assert log_module.read_tail(10) == []


def test_read_tail_respects_limit(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)

    records = [
        {"event": "one", "payload": {"value": 1}},
        {"event": "two", "payload": {"value": 2}},
        {"event": "three", "payload": {"value": 3}},
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(json.dumps(rec) for rec in records), encoding="utf-8")

    assert log_module.read_tail(2) == [json.dumps(records[-2]), json.dumps(records[-1])]


def test_read_tail_defaults_to_safe_limit(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)

    entries = [
        {"event": "alpha", "payload": {}},
        {"event": "beta", "payload": {}},
    ]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    tail = log_module.read_tail("not-a-number")

    assert tail == [json.dumps(entry) for entry in entries]


def test_read_tail_handles_non_positive_limits(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)
    log_file.write_text("{}\n".format(json.dumps({"event": "only"})), encoding="utf-8")

    assert log_module.read_tail(0) == []
    assert log_module.read_tail(-5) == []
