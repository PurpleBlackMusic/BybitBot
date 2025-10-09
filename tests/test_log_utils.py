from __future__ import annotations

import json

import pytest

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


def test_read_tail_can_parse_json_and_validate(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)

    entries = [
        {"event": "ok", "payload": {"value": 1}},
        {"event": "also_ok", "payload": {"value": 2}},
    ]
    invalid_line = "{not-json}"

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
        handle.write(invalid_line + "\n")

    parsed = log_module.read_tail(5, parse=True)
    assert [item["event"] for item in parsed] == ["ok", "also_ok"]

    with pytest.raises(ValueError):
        log_module.read_tail(5, parse=True, drop_invalid=False)


def test_log_prunes_when_file_exceeds_limit(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)
    monkeypatch.setattr(log_module, "MAX_LOG_BYTES", 150)
    monkeypatch.setattr(log_module, "RETAIN_LOG_LINES", 3)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    existing_records = [
        {"event": f"existing-{idx}", "payload": {"value": idx}}
        for idx in range(6)
    ]
    log_file.write_text(
        "\n".join(json.dumps(record) for record in existing_records) + "\n",
        encoding="utf-8",
    )

    before_size = log_file.stat().st_size

    log_module.log("new-entry", value=99)

    after_size = log_file.stat().st_size
    assert after_size <= before_size

    with log_file.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    assert len(lines) <= 3
    events = [json.loads(line)["event"] for line in lines]
    assert events[-1] == "new-entry"
    assert all(event.startswith("existing") or event == "new-entry" for event in events)


def test_log_infers_severity_and_thread(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)

    log_module.log("guardian.refresh.error", err="boom")

    record = json.loads(log_file.read_text(encoding="utf-8"))
    assert record["severity"] == "error"
    assert record["thread"]
    assert record["payload"]["err"] == "boom"


def test_log_respects_explicit_severity(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)

    log_module.log("custom.event", severity="WARNING", value=1)

    record = json.loads(log_file.read_text(encoding="utf-8"))
    assert record["severity"] == "warning"
    assert record["payload"]["value"] == 1


def test_log_serialises_exception_details(tmp_path, monkeypatch):
    log_file = _use_temp_log(tmp_path, monkeypatch)

    try:
        raise RuntimeError("kaboom")
    except RuntimeError as exc:
        log_module.log("runtime.failure", exc=exc)

    record = json.loads(log_file.read_text(encoding="utf-8"))
    assert record["severity"] == "error"
    exception = record["exception"]
    assert exception["type"].endswith("RuntimeError")
    assert "kaboom" in exception["message"]
    assert "RuntimeError: kaboom" in exception["traceback"]
