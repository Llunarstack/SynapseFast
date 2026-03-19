from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class StandardScaler:
    """
    Minimal sklearn-like StandardScaler.
    Works on numpy arrays of shape [N, D].
    """

    with_mean: bool = True
    with_std: bool = True
    eps: float = 1e-12
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be rank-2 [N, D]")
        if self.with_mean:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = np.zeros((X.shape[1],), dtype=X.dtype)
        if self.with_std:
            var = X.var(axis=0)
            self.scale_ = np.sqrt(var + self.eps)
        else:
            self.scale_ = np.ones((X.shape[1],), dtype=X.dtype)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted")
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)
