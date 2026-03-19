from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..metrics import accuracy, rmse
from .utils import require


@dataclass
class LightGBMClassifier:
    params: Optional[dict] = None
    model_: Optional[object] = None

    def fit(self, X, y):
        require(
            "lightgbm",
            package="lightgbm",
            extra="lightgbm",
            hint="This wrapper provides a sklearn-like API over LightGBM.",
        )
        import lightgbm as lgb

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model_ = lgb.LGBMClassifier(**(self.params or {}))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X)

    def score(self, X, y) -> float:
        return accuracy(y, self.predict(X))


@dataclass
class CatBoostClassifier:
    params: Optional[dict] = None
    model_: Optional[object] = None

    def fit(self, X, y):
        require(
            "catboost",
            package="catboost",
            extra="catboost",
            hint="This wrapper provides a sklearn-like API over CatBoost.",
        )
        from catboost import CatBoostClassifier as _CBC

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model_ = _CBC(**({"verbose": False} | (self.params or {})))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        pred = self.model_.predict(X)
        return np.asarray(pred).astype(np.int64).reshape(-1)

    def score(self, X, y) -> float:
        return accuracy(y, self.predict(X))


@dataclass
class LightGBMRegressor:
    params: Optional[dict] = None
    model_: Optional[object] = None

    def fit(self, X, y):
        require(
            "lightgbm", package="lightgbm", extra="lightgbm", hint="This wrapper provides regression API."
        )
        import lightgbm as lgb

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.model_ = lgb.LGBMRegressor(**(self.params or {}))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        return self.model_.predict(X).astype(np.float32)

    def score(self, X, y) -> float:
        # Higher is better: negative RMSE
        return -rmse(y, self.predict(X))
