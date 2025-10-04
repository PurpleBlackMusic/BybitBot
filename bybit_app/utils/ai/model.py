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

    @staticmethod
    def _logloss(y_true, y_pred):
        eps = 1e-9
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y, epochs=200, lr=0.05, l2=1e-4, patience: int | None = None, tol: float = 1e-5):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # normalize
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        Xn = (X - self.mu) / self.sigma
        best_loss = float("inf")
        patience_ctr = 0
        for _ in range(epochs):
            z = Xn.dot(self.w) + self.b
            p = self._sigmoid(z)
            grad_w = Xn.T.dot(p - y)/len(y) + l2*self.w
            grad_b = (p - y).mean()
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            if patience:
                loss = self._logloss(y, p)
                if loss + tol < best_loss:
                    best_loss = loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break

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
