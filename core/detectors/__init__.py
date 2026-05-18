"""
core/detectors/
────────────────
Detection backend implementations.

  UltralyticsDetector  — wraps ultralytics.YOLO  (AGPL-3.0)
  ONNXDetector         — pure onnxruntime         (MIT, AGPL-free)

Imports are lazy so that the ONNX path never pulls in torch/ultralytics.
"""

__all__ = ["UltralyticsDetector", "ONNXDetector"]


def __getattr__(name: str):
    if name == "ONNXDetector":
        from core.detectors.onnx_detector import ONNXDetector
        return ONNXDetector
    if name == "UltralyticsDetector":
        from core.detectors.ultralytics_detector import UltralyticsDetector
        return UltralyticsDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
