from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import accuracy, rmse


@dataclass
class KNNClassifier:
    """
    Tiny baseline classifier (fast enough for small demos).
    API: fit/predict/score like sklearn.
    """

    k: int = 5
    X_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be [N, D]")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be [N]")
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X: np.ndarray):
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        # Compute squared L2 distance to all train points (brute force).
        # For large datasets, use sklearn / faiss; this is a demo baseline.
        d2 = ((X[:, None, :] - self.X_[None, :, :]) ** 2).sum(axis=-1)  # [N_test, N_train]
        nn = np.argpartition(d2, kth=min(self.k, d2.shape[1] - 1), axis=1)[:, : self.k]
        votes = self.y_[nn]
        # Majority vote
        out = []
        for row in votes:
            vals, counts = np.unique(row, return_counts=True)
            out.append(vals[counts.argmax()])
        return np.asarray(out)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy(y, self.predict(X))


@dataclass
class LinearRegression:
    """
    Simple ridge-less linear regression using normal equations (small D only).
    """

    w_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be [N, D]")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be [N]")
        # Add bias term
        Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        # Solve least squares
        self.w_, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        return self

    def predict(self, X: np.ndarray):
        if self.w_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float64)
        Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        return (Xb @ self.w_).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return -rmse(y, self.predict(X))
