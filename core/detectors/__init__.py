"""
core/detectors/
────────────────
Detection backend implementations.

  UltralyticsDetector  — wraps ultralytics.YOLO  (AGPL-3.0)
  ONNXDetector         — pure onnxruntime         (MIT, AGPL-free)
"""
from core.detectors.ultralytics_detector import UltralyticsDetector
from core.detectors.onnx_detector        import ONNXDetector

__all__ = ["UltralyticsDetector", "ONNXDetector"]
