"""
core/base_detector.py
─────────────────────
Detector abstraction — every backend (Ultralytics, ONNX, future) implements
this interface so the rest of the app never imports a specific backend.

Breaking the interface free of ultralytics means:
  - .onnx weights load via onnxruntime (MIT) with zero AGPL surface
  - future backends (OpenVINO, TFLite, CoreML) drop in without UI changes
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from models.annotation_model import BoundingBox


class BaseDetector(ABC):
    """Shared interface for every detection backend."""

    confidence: float = 0.45
    iou:        float = 0.45

    # ── lifecycle ─────────────────────────────────────────────────────────────
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load weights from *model_path*.  Must be idempotent."""

    @abstractmethod
    def is_loaded(self) -> bool: ...

    # ── inference ─────────────────────────────────────────────────────────────
    @abstractmethod
    def detect(self, bgr_frame) -> List[BoundingBox]:
        """
        Run detection on a single BGR frame (numpy HWC uint8).
        Returns normalised BoundingBox list (x_center, y_center, w, h ∈ [0,1]).
        """

    # ── metadata ──────────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Return {class_id: class_name} dict."""

    # ── helpers ───────────────────────────────────────────────────────────────
    @property
    def backend_name(self) -> str:
        return self.__class__.__name__
