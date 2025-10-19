from __future__ import annotations

import threading
import time
from collections import deque

from .log import log

__all__ = ["reserve_ws_connection_slot"]


class WSConnectionLimiter:
    """Enforce Bybit's limit of 500 websocket connections per five minutes."""

    def __init__(self, *, limit: int = 500, window_seconds: float = 300.0) -> None:
        self.limit = max(int(limit), 1)
        self.window = max(float(window_seconds), 1.0)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def reserve(self) -> None:
        while True:
            with self._condition:
                now = time.time()
                queue = self._timestamps
                while queue and now - queue[0] >= self.window:
                    queue.popleft()
                if len(queue) < self.limit:
                    queue.append(now)
                    self._condition.notify_all()
                    return
                wait_for = self.window - (now - queue[0]) if queue else self.window
                if wait_for > 0:
                    log(
                        "ws.connection.limit.wait",
                        seconds=round(wait_for, 2),
                        limit=self.limit,
                        window=self.window,
                    )
                    self._condition.wait(timeout=min(wait_for, self.window))
                else:
                    self._condition.wait(timeout=0.1)


_limiter = WSConnectionLimiter()


def reserve_ws_connection_slot() -> None:
    """Block until a websocket connection slot is available."""

    _limiter.reserve()
