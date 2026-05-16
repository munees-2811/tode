"""
core/detectors/ultralytics_detector.py
────────────────────────────────────────
Detector backed by the Ultralytics library (YOLO26 / YOLO11 / YOLOv8).

LICENCE NOTE: Ultralytics is AGPL-3.0.  Any code that imports this module
inherits that obligation.  For closed-source or SaaS deployments use
ONNXDetector instead — it carries no AGPL surface.
"""
import os
from typing import Dict, List

from core.base_detector import BaseDetector
from models.annotation_model import BoundingBox
from utils.logger import get_logger

log = get_logger("core.detectors.UltralyticsDetector")


class UltralyticsDetector(BaseDetector):
    """Wraps ultralytics.YOLO.  Accepts any .pt model name or path."""

    def __init__(self, confidence: float = 0.45, iou: float = 0.45):
        self.confidence = confidence
        self.iou        = iou
        self._model     = None
        self._model_path: str = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        if self._model is not None and self._model_path == model_path:
            return
        self._load_weights(model_path)

    def _load_weights(self, weights: str) -> None:
        from ultralytics import YOLO  # import isolated here — AGPL surface

        if not os.path.exists(weights) and not weights.endswith(".pt"):
            weights = weights + ".pt"

        log.info(f"[Ultralytics] loading: {weights}")
        try:
            self._model      = YOLO(weights)
            self._model_path = weights
            log.info(
                f"[Ultralytics] ready — "
                f"{len(self._model.names)} classes"
            )
        except Exception as exc:
            log.error(f"[Ultralytics] load failed: {exc}", exc_info=True)
            raise

    def is_loaded(self) -> bool:
        return self._model is not None

    # ── inference ─────────────────────────────────────────────────────────────
    def detect(self, bgr_frame) -> List[BoundingBox]:
        if not self.is_loaded():
            return []
        try:
            results = self._model.predict(
                source  = bgr_frame,
                conf    = self.confidence,
                iou     = self.iou,
                verbose = False,
            )
        except Exception as exc:
            log.error(f"[Ultralytics] inference error: {exc}", exc_info=True)
            return []

        boxes: List[BoundingBox] = []
        for result in results:
            img_h, img_w = bgr_frame.shape[:2]
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append(BoundingBox(
                    class_id   = cls_id,
                    class_name = cls_name,
                    x_center   = ((x1 + x2) / 2) / img_w,
                    y_center   = ((y1 + y2) / 2) / img_h,
                    width      = (x2 - x1)        / img_w,
                    height     = (y2 - y1)        / img_h,
                    confidence = conf,
                ))
        log.info(f"[Ultralytics] {len(boxes)} detection(s)")
        return boxes

    # ── metadata ──────────────────────────────────────────────────────────────
    @property
    def class_names(self) -> Dict[int, str]:
        return self._model.names if self._model else {}

    @property
    def backend_name(self) -> str:
        return "Ultralytics"
