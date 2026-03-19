from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .utils import require


@dataclass
class OpenCV:
    """
    Small convenience wrapper for OpenCV import + common ops.
    """

    cv2_: Optional[Any] = None

    def _ensure(self):
        require("cv2", package="opencv-python", extra="cv", hint='Install: pip install -e ".[cv]"')
        import cv2

        self.cv2_ = cv2

    def imread(self, path: str):
        self._ensure()
        return self.cv2_.imread(path)

    def resize(self, img, size):
        self._ensure()
        return self.cv2_.resize(img, size)
