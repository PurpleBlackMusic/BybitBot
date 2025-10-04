
from __future__ import annotations
import time, threading

class RateLimiter:
    def __init__(self, rate_per_sec: float = 50.0):
        self.rate = float(rate_per_sec)
        self.tokens = self.rate
        self.last = time.time()
        self.lock = threading.Lock()

    def acquire(self, cost: float = 1.0):
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.rate, self.tokens + elapsed*self.rate)
            if self.tokens >= cost:
                self.tokens -= cost
                return
            # wait
            need = (cost - self.tokens)/self.rate
        time.sleep(max(0.0, need))
        with self.lock:
            self.tokens = max(0.0, self.tokens - cost)
