from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple


class Estimator:
    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError


@dataclass
class Pipeline(Estimator):
    """
    Minimal sklearn-like Pipeline:
    - all but last must implement fit/transform
    - last must implement fit/predict (and optionally score)
    """

    steps: Sequence[Tuple[str, Any]]

    def fit(self, X, y=None):
        if not self.steps:
            raise ValueError("Pipeline needs at least one step")
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, y)
            else:
                step.fit(Xt, y)
                Xt = step.transform(Xt)
        last_name, last = self.steps[-1]
        last.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    def score(self, X, y):
        last = self.steps[-1][1]
        if hasattr(last, "score"):
            Xt = X
            for name, step in self.steps[:-1]:
                Xt = step.transform(Xt)
            return last.score(Xt, y)
        raise AttributeError("Last estimator has no score()")
