"""Local DeepSeek adapter backed by ``llama-cpp-python`` models."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover - optional dependency import
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - optional dependency import
    Llama = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class DeepSeekLocalDependencyError(RuntimeError):
    """Raised when llama.cpp bindings are unavailable on the host."""

    pass


@dataclass(frozen=True)
class _RuntimeHints:
    """Container with llama.cpp tuning parameters."""

    n_ctx: int
    n_gpu_layers: int
    n_threads: int
    n_batch: int
    max_tokens: int


def _is_apple_silicon() -> bool:
    """Return ``True`` when running on an Apple Silicon (ARM) macOS host."""

    if platform.system() != "Darwin":
        return False
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine.startswith("aarch64")


def _read_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _detect_total_memory() -> int | None:
    """Best-effort detection of physical memory for tuning heuristics."""

    if platform.system() != "Darwin":
        return None
    try:
        completed = subprocess.run(  # noqa: S603, S607
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - platform specific
        return None
    output = completed.stdout.strip()
    try:
        return int(output)
    except ValueError:
        return None


def _tune_runtime(
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int | None,
    max_tokens: int,
) -> _RuntimeHints:
    """Adjust llama.cpp parameters for constrained Apple Silicon laptops."""

    base_ctx = max(int(n_ctx), 512)
    base_tokens = max(int(max_tokens), 128)
    tuned_layers = int(n_gpu_layers)
    is_apple = _is_apple_silicon()
    mem_bytes = _detect_total_memory() if is_apple else None
    cpu_count = os.cpu_count() or 2

    threads: int
    if n_threads is not None:
        threads = max(1, min(int(n_threads), cpu_count))
    elif is_apple:
        threads = max(2, min(cpu_count, 6))
    else:
        threads = max(1, cpu_count)

    tuned_ctx = base_ctx
    tuned_max_tokens = min(base_tokens, base_ctx)
    tuned_batch = min(max(128, base_ctx // 2), 512)

    if is_apple:
        low_mem_threshold = 9 * 1024**3  # ~9 GiB allows for OS overhead
        mid_mem_threshold = 17 * 1024**3
        if mem_bytes is not None and mem_bytes <= low_mem_threshold:
            tuned_ctx = min(tuned_ctx, 1792)
            tuned_max_tokens = min(tuned_max_tokens, 512)
            tuned_batch = min(tuned_batch, 192)
            if tuned_layers <= 0:
                tuned_layers = 0  # keep inference on CPU to avoid memory thrash
            threads = max(2, min(threads, 4))
        elif mem_bytes is not None and mem_bytes <= mid_mem_threshold:
            tuned_ctx = min(tuned_ctx, 3072)
            tuned_max_tokens = min(tuned_max_tokens, 640)
            tuned_batch = min(tuned_batch, 256)
            if tuned_layers == 0:
                tuned_layers = -1  # let llama.cpp decide Metal offload
            threads = max(2, min(threads, 6))
        else:
            tuned_ctx = min(tuned_ctx, 4096)
            tuned_max_tokens = min(tuned_max_tokens, 768)
            if tuned_layers == 0:
                tuned_layers = -1
            tuned_batch = min(tuned_batch, 384)
    else:
        tuned_ctx = min(tuned_ctx, 4096)
        tuned_max_tokens = min(tuned_max_tokens, tuned_ctx - 64)

    env_ctx = _read_env_int("DEEPSEEK_LOCAL_N_CTX")
    if env_ctx is not None and env_ctx > 0:
        tuned_ctx = max(128, env_ctx)
    env_gpu_layers = _read_env_int("DEEPSEEK_LOCAL_N_GPU_LAYERS")
    if env_gpu_layers is not None:
        tuned_layers = env_gpu_layers
    env_threads = _read_env_int("DEEPSEEK_LOCAL_THREADS")
    if env_threads is not None and env_threads > 0:
        threads = env_threads
    env_batch = _read_env_int("DEEPSEEK_LOCAL_N_BATCH")
    if env_batch is not None and env_batch > 0:
        tuned_batch = env_batch
    env_max_tokens = _read_env_int("DEEPSEEK_LOCAL_MAX_TOKENS")
    if env_max_tokens is not None and env_max_tokens > 0:
        tuned_max_tokens = env_max_tokens

    tuned_ctx = max(256, tuned_ctx)
    tuned_max_tokens = max(128, min(tuned_max_tokens, tuned_ctx - 64))
    tuned_batch = max(64, min(tuned_batch, tuned_ctx))
    threads = max(1, threads)

    return _RuntimeHints(
        n_ctx=int(tuned_ctx),
        n_gpu_layers=int(tuned_layers),
        n_threads=int(threads),
        n_batch=int(tuned_batch),
        max_tokens=int(tuned_max_tokens),
    )


def _coerce_runtime_hints(
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int | None,
    n_batch: int,
    max_tokens: int,
) -> _RuntimeHints:
    """Return normalised runtime hints without reapplying heuristics."""

    threads = int(n_threads) if n_threads is not None else (os.cpu_count() or 2)
    return _RuntimeHints(
        n_ctx=int(n_ctx),
        n_gpu_layers=int(n_gpu_layers),
        n_threads=max(1, int(threads)),
        n_batch=int(n_batch),
        max_tokens=int(max_tokens),
    )


@lru_cache(maxsize=1)
def _load_model(
    model_path: str,
    *,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    n_threads: int | None = None,
    n_batch: int = 256,
    max_tokens: int = 600,
    _weight_mtime: int | None = None,
    _apply_tuning: bool = True,
) -> "Llama":
    """Load and cache a ``Llama`` instance for local inference."""

    if _IMPORT_ERROR is not None:  # pragma: no cover - defensive guard
        raise DeepSeekLocalDependencyError(
            "llama-cpp-python is required for local DeepSeek inference"
        ) from _IMPORT_ERROR
    # ``_weight_mtime`` is intentionally unused other than affecting the cache key
    # so a weight refresh triggers a reload without forcing a global cache clear.
    _ = _weight_mtime

    hints = (
        _tune_runtime(n_ctx, n_gpu_layers, n_threads, max_tokens)
        if _apply_tuning
        else _coerce_runtime_hints(n_ctx, n_gpu_layers, n_threads, n_batch, max_tokens)
    )
    kwargs = {
        "model_path": model_path,
        "n_ctx": hints.n_ctx,
        "n_gpu_layers": hints.n_gpu_layers,
        "n_threads": hints.n_threads,
        "n_batch": hints.n_batch,
        "use_mmap": True,
        "use_mlock": False,
    }
    try:
        return Llama(**kwargs)
    except TypeError as exc:  # pragma: no cover - compatibility shim
        if "n_batch" in str(exc):
            kwargs.pop("n_batch", None)
            return Llama(**kwargs)
        raise


class DeepSeekLocalAdapter:
    """Local adapter for generating trading signals via a GGUF model."""

    def __init__(
        self,
        model_path: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 600,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.temperature = float(temperature)
        hints = _tune_runtime(int(n_ctx), int(n_gpu_layers), n_threads, int(max_tokens))
        self.max_tokens = hints.max_tokens
        self.n_ctx = hints.n_ctx
        self.n_gpu_layers = hints.n_gpu_layers
        self.n_threads = hints.n_threads
        self.n_batch = hints.n_batch
        self._runtime_hints = hints

    def _build_messages(self, symbol: str) -> list[dict[str, str]]:
        system_prompt = (
            "You are an experienced quantitative crypto analyst. "
            "Return a concise JSON object describing a trading setup for the given symbol. "
            "Use the schema: {\"direction\": \"long|short|neutral\", \"confidence\": number (0-1), "
            "\"entry\": number or null, \"stop_loss\": number or null, \"take_profit\": number or null, \"summary\": string}. "
            "If insufficient data is available, return neutral with confidence 0 and leave numeric fields null. "
            "Only respond with JSON."
        )
        user_prompt = f"Symbol: {symbol}. Provide the trading setup JSON."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def get_signal(self, symbol: str) -> Dict[str, Any]:
        try:
            stat_result = self.model_path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"DeepSeek weight missing at {self.model_path}"
            ) from exc
        weight_mtime = getattr(stat_result, "st_mtime_ns", None)
        if weight_mtime is None:
            weight_mtime = int(stat_result.st_mtime * 1_000_000_000)
        model = _load_model(
            str(self.model_path),
            n_ctx=self._runtime_hints.n_ctx,
            n_gpu_layers=self._runtime_hints.n_gpu_layers,
            n_threads=self._runtime_hints.n_threads,
            n_batch=self._runtime_hints.n_batch,
            max_tokens=self._runtime_hints.max_tokens,
            _weight_mtime=weight_mtime,
            _apply_tuning=False,
        )
        messages = self._build_messages(symbol)
        response = model.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=["\n\n"],
        )
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict):
                        content_value = message.get("content")
                        if isinstance(content_value, str):
                            content = content_value
        if not content:
            content = "{\"direction\": \"neutral\", \"confidence\": 0.0}"
        try:
            data = json.loads(content)
        except Exception:
            data = {"direction": "neutral", "confidence": 0.0}
        if "direction" not in data:
            data["direction"] = "neutral"
        if "confidence" not in data:
            data["confidence"] = 0.0
        return data


__all__ = ["DeepSeekLocalAdapter", "DeepSeekLocalDependencyError"]
