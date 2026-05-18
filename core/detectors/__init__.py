"""
core/detectors/
────────────────
Detection backend implementations.

  UltralyticsDetector  — wraps ultralytics.YOLO  (AGPL-3.0)
  ONNXDetector         — pure onnxruntime         (MIT, AGPL-free)

Backend imports are deferred so that the ONNX path never pulls in
torch/ultralytics. Import directly from the submodules:

    from core.detectors.onnx_detector import ONNXDetector
    from core.detectors.ultralytics_detector import UltralyticsDetector
"""
