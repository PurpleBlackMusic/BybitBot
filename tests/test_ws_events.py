import pytest

from bybit_app.utils.ws_events import (
    EventQueue,
    event_queue_stats,
    fetch_events,
    publish_event,
    reset_event_queue,
)


def test_event_queue_publish_and_fetch_order() -> None:
    queue = EventQueue(maxlen=4)
    first = queue.publish(scope="private", topic="orders", payload={"id": 1})
    second = queue.publish(scope="private", topic="orders", payload={"id": 2})

    events = queue.fetch(scope="private")
    assert [event["id"] for event in events] == [first["id"], second["id"]]
    assert events[0]["payload"] == {"id": 1}


def test_event_queue_filters_scope_and_since() -> None:
    queue = EventQueue(maxlen=4)
    first = queue.publish(scope="private", topic="orders", payload={"id": 1})
    queue.publish(scope="public", topic="tickers", payload={"id": 2})
    queue.publish(scope="private", topic="executions", payload={"id": 3})

    filtered = queue.fetch(scope="private", since=first["id"])
    assert len(filtered) == 1
    assert filtered[0]["topic"] == "executions"


def test_event_queue_tracks_overflow() -> None:
    queue = EventQueue(maxlen=1)
    queue.publish(scope="private", topic="orders", payload={"id": 1})
    queue.publish(scope="private", topic="orders", payload={"id": 2})
    stats = queue.stats()
    assert stats["dropped"] == 1
    assert stats["latest_id"] == 2


def test_global_queue_helpers() -> None:
    reset_event_queue()
    publish_event(scope="private", topic="orders", payload={"id": "abc"})
    publish_event(scope="private", topic="executions", payload={"id": "def"})

    events = fetch_events(scope="private")
    assert len(events) == 2
    assert events[-1]["topic"] == "executions"

    stats = event_queue_stats()
    assert stats["latest_id"] >= 2

    reset_event_queue()
    assert fetch_events() == []


def test_event_queue_rejects_empty_topic() -> None:
    queue = EventQueue(maxlen=2)
    with pytest.raises(ValueError):
        queue.publish(scope="private", topic="", payload={})
