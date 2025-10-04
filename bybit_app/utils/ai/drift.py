
from __future__ import annotations
class PageHinkley:
    """Простой детектор дрейфа (Page-Hinkley).
    Считаем поток ошибок (0/1), сигналим, когда среднее резко ухудшается.
    """
    def __init__(self, delta: float = 0.005, lamb: float = 50.0, alpha: float = 0.9999):
        self.delta = float(delta)
        self.lamb = float(lamb)
        self.alpha = float(alpha)
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, x: float) -> bool:
        # x — ошибка (0 для верно, 1 для неверно)
        self.mean = self.alpha*self.mean + (1.0-self.alpha)*x
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if (self.cum - self.min_cum) > self.lamb:
            # drift
            self.reset()
            return True
        return False

    def reset(self):
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
