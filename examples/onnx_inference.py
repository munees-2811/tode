"""
examples/onnx_inference.py
──────────────────────────
Minimal AGPL-free inference script using the ONNX backend.

Usage:
    python examples/onnx_inference.py <image_path> <model.onnx>

Output: prints one line per detection in the form
    <class_name> <confidence> <x1> <y1> <x2> <y2>

Requires only opencv-python, numpy, onnxruntime (see requirements-onnx.txt).
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detectors.onnx_detector import ONNXDetector  # noqa: E402


def main(image_path: str, model_path: str, conf: float = 0.25) -> int:
    if not os.path.exists(image_path):
        print(f"image not found: {image_path}", file=sys.stderr)
        return 1
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"could not decode image: {image_path}", file=sys.stderr)
        return 1

    det = ONNXDetector(conf_threshold=conf)
    det.load(model_path)

    h, w = frame.shape[:2]
    for box in det.detect(frame):
        x1, y1, x2, y2 = box.to_pixel_coords(w, h)
        print(f"{box.class_name} {box.confidence:.3f} {x1} {y1} {x2} {y2}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
