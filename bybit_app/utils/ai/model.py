
from __future__ import annotations
import json
import numpy as np

class Logistic:
    def __init__(self, n_features: int):
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.mu = np.zeros(n_features, dtype=float)
        self.sigma = np.ones(n_features, dtype=float)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, epochs=200, lr=0.05, l2=1e-4):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # normalize
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        Xn = (X - self.mu) / self.sigma
        for _ in range(epochs):
            z = Xn.dot(self.w) + self.b
            p = self._sigmoid(z)
            grad_w = Xn.T.dot(p - y)/len(y) + l2*self.w
            grad_b = (p - y).mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xn = (X - self.mu) / self.sigma
        z = Xn.dot(self.w) + self.b
        return self._sigmoid(z)

    def to_json(self):
        return json.dumps({
            "w": self.w.tolist(),
            "b": float(self.b),
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist()
        })

    @staticmethod
    def from_json(s: str):
        obj = json.loads(s)
        m = Logistic(n_features=len(obj["w"]))
        m.w = np.array(obj["w"], dtype=float)
        m.b = float(obj["b"])
        m.mu = np.array(obj["mu"], dtype=float)
        m.sigma = np.array(obj["sigma"], dtype=float)
        return m
