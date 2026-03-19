from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .utils import require


@dataclass
class HFTextClassifier:
    """
    Minimal wrapper around Hugging Face Transformers for quick text classification inference.
    Training is intentionally not wrapped here (Transformers Trainer is already great).
    """

    model_id: str = "distilbert-base-uncased-finetuned-sst-2-english"
    device: str = "cpu"  # "cpu" or "cuda"
    pipe_: Optional[Any] = None

    def _ensure(self):
        require(
            "transformers",
            package="transformers",
            extra="nlp",
            hint='Install NLP extras: pip install -e ".[nlp]"',
        )
        from transformers import pipeline

        if self.pipe_ is None:
            dev = 0 if (self.device == "cuda") else -1
            self.pipe_ = pipeline("text-classification", model=self.model_id, device=dev)

    def predict(self, texts):
        self._ensure()
        out = self.pipe_(list(texts))
        return out


@dataclass
class SpacyNlp:
    model: str = "en_core_web_sm"
    nlp_: Optional[Any] = None

    def _ensure(self):
        require("spacy", package="spacy", extra="spacy", hint='Install: pip install -e ".[spacy]"')
        import spacy

        if self.nlp_ is None:
            self.nlp_ = spacy.load(self.model)

    def __call__(self, text: str):
        self._ensure()
        return self.nlp_(text)
