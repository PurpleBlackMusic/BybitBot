import json
import os
from types import SimpleNamespace


def _make_dummy_response(payload):
    class _Response:
        status_code = 200
        text = ""

        def json(self):
            return payload

    return _Response()


def test_load_model_reloads_when_weight_changes(tmp_path, monkeypatch):
    from bybit_app.utils.ai import deepseek_local

    deepseek_local._load_model.cache_clear()

    created_instances: list[SimpleNamespace] = []

    class DummyLlama:
        def __init__(self, **kwargs):
            created_instances.append(SimpleNamespace(**kwargs))
            self._response = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"direction": "long", "confidence": 0.5}
                            )
                        }
                    }
                ]
            }

        def create_chat_completion(self, **_kwargs):
            return self._response

    monkeypatch.setattr(deepseek_local, "Llama", DummyLlama)
    monkeypatch.setattr(deepseek_local, "_IMPORT_ERROR", None)

    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"0")

    adapter = deepseek_local.DeepSeekLocalAdapter(
        str(weight), max_tokens=200, n_ctx=512
    )

    first = adapter.get_signal("BTCUSDT")
    assert first["direction"] == "long"
    assert len(created_instances) == 1

    # Cached model should be reused while the weight file is unchanged.
    _ = adapter.get_signal("ETHUSDT")
    assert len(created_instances) == 1

    stat_result = weight.stat()
    os.utime(weight, (stat_result.st_atime, stat_result.st_mtime + 1))

    third = adapter.get_signal("LTCUSDT")
    assert third["direction"] == "long"
    assert len(created_instances) == 2

    deepseek_local._load_model.cache_clear()


def test_adapter_disables_local_after_dependency_error(tmp_path, monkeypatch):
    from bybit_app.utils.ai import deepseek_adapter, deepseek_local

    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"0")

    call_counter = {"count": 0}

    class DummyLocalAdapter:
        def __init__(self, *_args, **_kwargs):
            call_counter["count"] += 1

        def get_signal(self, _symbol):
            raise deepseek_local.DeepSeekLocalDependencyError("missing llama-cpp")

    monkeypatch.setattr(deepseek_local, "DeepSeekLocalAdapter", DummyLocalAdapter)

    cache_path = tmp_path / "cache.json"
    adapter = deepseek_adapter.DeepSeekAdapter(
        local_model_path=str(weight), cache_path=cache_path
    )
    adapter.api_key = "token"

    responses: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_post(*args, **kwargs):
        responses.append((args, kwargs))
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"direction": "short", "confidence": 0.25}
                        )
                    }
                }
            ]
        }
        return _make_dummy_response(payload)

    adapter.session = SimpleNamespace(post=fake_post)

    first = adapter.get_signal("BTCUSDT")
    assert first["deepseek_direction"] == "short"
    assert call_counter["count"] == 1
    assert adapter._local_model_enabled is False

    second = adapter.get_signal("ETHUSDT")
    assert second["deepseek_direction"] == "short"
    assert call_counter["count"] == 1
    assert len(responses) == 2
