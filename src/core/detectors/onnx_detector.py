"""
core/detectors/onnx_detector.py
────────────────────────────────
Detector backed by ONNX Runtime (MIT licensed).

Loading a .onnx file exported from a trained YOLO model carries NO Ultralytics
AGPL obligation — the ONNX graph is just a neural network, not a derivative
work of the Ultralytics library.  This is the correct path for:
  - closed-source / commercial deployments
  - air-gapped / edge environments (no pip install ultralytics)
  - Jetson / ARM devices where Ultralytics wheels are unavailable

Export your trained model once:
    yolo export model=todev1.pt format=onnx imgsz=640
Then place todev1.onnx + todev1_classes.json in weights/ and point the
model selector at it.  ultralytics is never imported at runtime.

Expected ONNX output format (YOLOv8 / YOLO11 / YOLO26 default export):
    input  : [1, 3, 640, 640]  float32  RGB  normalised [0, 1]
    output : [1, 4+nc, 8400]   float32
             first 4 rows → cx, cy, w, h  (pixel coords in 640-space)
             rows 4..     → per-class confidence scores
"""
import json
import os

import cv2
import numpy as np

from core.base_detector import BaseDetector
from models.annotation_model import BoundingBox
from utils.logger import get_logger

log = get_logger("core.detectors.ONNXDetector")

_INPUT_SIZE = 640  # YOLO default export resolution

# COCO-80 fallback class names (used when no sidecar JSON is found)
_COCO_80 = {
    0: "person",        1: "bicycle",       2: "car",
    3: "motorcycle",    4: "airplane",      5: "bus",
    6: "train",         7: "truck",         8: "boat",
    9: "traffic light", 10: "fire hydrant", 11: "stop sign",
    12: "parking meter",13: "bench",        14: "bird",
    15: "cat",          16: "dog",          17: "horse",
    18: "sheep",        19: "cow",          20: "elephant",
    21: "bear",         22: "zebra",        23: "giraffe",
    24: "backpack",     25: "umbrella",     26: "handbag",
    27: "tie",          28: "suitcase",     29: "frisbee",
    30: "skis",         31: "snowboard",    32: "sports ball",
    33: "kite",         34: "baseball bat", 35: "baseball glove",
    36: "skateboard",   37: "surfboard",    38: "tennis racket",
    39: "bottle",       40: "wine glass",   41: "cup",
    42: "fork",         43: "knife",        44: "spoon",
    45: "bowl",         46: "banana",       47: "apple",
    48: "sandwich",     49: "orange",       50: "broccoli",
    51: "carrot",       52: "hot dog",      53: "pizza",
    54: "donut",        55: "cake",         56: "chair",
    57: "couch",        58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet",       62: "tv",
    63: "laptop",       64: "mouse",        65: "remote",
    66: "keyboard",     67: "cell phone",   68: "microwave",
    69: "oven",         70: "toaster",      71: "sink",
    72: "refrigerator", 73: "book",         74: "clock",
    75: "vase",         76: "scissors",     77: "teddy bear",
    78: "hair drier",   79: "toothbrush",
}


class ONNXDetector(BaseDetector):
    """
    AGPL-free detector.  Requires: pip install onnxruntime
    For NVIDIA GPU acceleration:  pip install onnxruntime-gpu
    """

    def __init__(self, confidence: float = 0.45, iou: float = 0.45):
        self.confidence  = confidence
        self.iou         = iou
        self._session    = None
        self._model_path = ""
        self._names: dict[int, str] = {}

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def load(self, model_path: str) -> None:
        if self._session is not None and self._model_path == model_path:
            return
        self._load_session(model_path)

    def _load_session(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed.\n"
                "Run: pip install onnxruntime\n"
                "GPU:  pip install onnxruntime-gpu"
            ) from exc

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        log.info(f"[ONNX] loading: {model_path}")
        providers = ort.get_available_providers()
        log.debug(f"[ONNX] available providers: {providers}")

        # Prefer GPU if available
        preferred = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in providers
            else ["CPUExecutionProvider"]
        )
        self._session    = ort.InferenceSession(model_path, providers=preferred)
        self._model_path = model_path
        self._names      = self._load_class_names(model_path)
        log.info(
            f"[ONNX] ready — {len(self._names)} classes — "
            f"provider: {self._session.get_providers()[0]}"
        )

    def is_loaded(self) -> bool:
        return self._session is not None

    # ── inference ─────────────────────────────────────────────────────────────
    def detect(self, bgr_frame) -> list[BoundingBox]:
        if not self.is_loaded():
            return []

        img_h, img_w = bgr_frame.shape[:2]
        tensor, scale, pad_top, pad_left = self._preprocess(bgr_frame)

        input_name = self._session.get_inputs()[0].name
        try:
            raw = self._session.run(None, {input_name: tensor})
        except Exception as exc:
            log.error(f"[ONNX] inference error: {exc}", exc_info=True)
            return []

        return self._postprocess(
            raw[0], scale, pad_top, pad_left, img_w, img_h
        )

    # ── preprocessing ─────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(
        bgr_frame,
    ) -> tuple[np.ndarray, float, int, int]:
        """
        Letterbox resize → RGB → NCHW float32 [0, 1].
        Returns (tensor, scale, pad_top, pad_left).
        """
        h, w   = bgr_frame.shape[:2]
        scale  = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)

        resized = cv2.resize(bgr_frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        canvas   = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        pad_top  = (_INPUT_SIZE - new_h) // 2
        pad_left = (_INPUT_SIZE - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # BGR → RGB, HWC → NCHW, uint8 → float32 [0, 1]
        rgb    = canvas[:, :, ::-1].astype(np.float32) / 255.0
        tensor = rgb.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 640, 640]
        return tensor, scale, pad_top, pad_left

    # ── postprocessing ────────────────────────────────────────────────────────
    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_top: int,
        pad_left: int,
        img_w: int,
        img_h: int,
    ) -> list[BoundingBox]:
        """
        Decode YOLOv8-style ONNX output [1, 4+nc, 8400] and apply NMS.
        Converts surviving boxes to normalised BoundingBox format.
        """
        # [1, 4+nc, 8400] → [8400, 4+nc]
        preds = output.squeeze(0).T

        raw_boxes:    list[list[float]] = []
        confidences:  list[float]       = []
        class_ids:    list[int]         = []

        for pred in preds:
            cx, cy, bw, bh = pred[:4]
            class_scores   = pred[4:]
            cls_id         = int(np.argmax(class_scores))
            conf           = float(class_scores[cls_id])

            if conf < self.confidence:
                continue

            # Convert from padded-640 space → original image pixels
            x1 = (cx - bw / 2 - pad_left) / scale
            y1 = (cy - bh / 2 - pad_top)  / scale
            x2 = (cx + bw / 2 - pad_left) / scale
            y2 = (cy + bh / 2 - pad_top)  / scale

            # Clamp to image bounds
            x1 = max(0.0, min(x1, img_w))
            y1 = max(0.0, min(y1, img_h))
            x2 = max(0.0, min(x2, img_w))
            y2 = max(0.0, min(y2, img_h))

            raw_boxes.append([x1, y1, x2 - x1, y2 - y1])   # xywh for NMS
            confidences.append(conf)
            class_ids.append(cls_id)

        if not raw_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            raw_boxes, confidences, self.confidence, self.iou
        )

        boxes: list[BoundingBox] = []
        for i in indices:
            x, y, w, h = raw_boxes[i]
            cls_id     = class_ids[i]
            boxes.append(BoundingBox(
                class_id   = cls_id,
                class_name = self._names.get(cls_id, str(cls_id)),
                x_center   = (x + w / 2) / img_w,
                y_center   = (y + h / 2) / img_h,
                width      = w           / img_w,
                height     = h           / img_h,
                confidence = confidences[i],
            ))

        log.info(f"[ONNX] {len(boxes)} detection(s)")
        return boxes

    # ── class names ───────────────────────────────────────────────────────────
    @staticmethod
    def _load_class_names(model_path: str) -> dict[int, str]:
        """
        Look for a sidecar JSON next to the .onnx file:
            todev1.onnx  →  todev1_classes.json
        Format: {"0": "person", "1": "car", ...}
        Falls back to COCO-80 if not found.
        """
        stem     = os.path.splitext(model_path)[0]
        sidecar  = stem + "_classes.json"
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, encoding="utf-8") as f:
                    raw = json.load(f)
                names = {int(k): v for k, v in raw.items()}
                log.info(
                    f"[ONNX] loaded {len(names)} class names from {sidecar}"
                )
                return names
            except Exception as exc:
                log.warning(f"[ONNX] could not read {sidecar}: {exc}")

        log.info("[ONNX] no class sidecar found — using COCO-80 fallback")
        return dict(_COCO_80)

    @property
    def class_names(self) -> dict[int, str]:
        return self._names

    @property
    def backend_name(self) -> str:
        return "ONNX Runtime"
