"""Saves and loads annotations in YOLO .txt format (or JSON)."""
import json
import os

from models.annotation_model import BoundingBox, FrameAnnotation
from utils.config import LABEL_FORMAT, LABELS_DIR


class LabelStorage:
    """
    Persists FrameAnnotation labels.

    Supported formats
    -----------------
    "yolo"  — one .txt file per frame (class cx cy w h)
    "json"  — one .json file per frame

    Both formats also maintain a sidecar ``classes.json`` mapping
    class_id → class_name so labels reload with proper names.
    """

    CLASSES_FILE = "classes.json"

    def __init__(self, video_name: str, fmt: str = LABEL_FORMAT):
        self.video_name = video_name
        self.fmt        = fmt
        self.base_dir   = os.path.join(LABELS_DIR, video_name)
        os.makedirs(self.base_dir, exist_ok=True)
        self._class_map: dict = self._load_class_map()

    # ── save ──────────────────────────────────────────────────────────────────
    def save(self, annotation: FrameAnnotation) -> str:
        """Write annotation to disk. Returns path of written file."""
        for box in annotation.boxes:
            self._class_map[int(box.class_id)] = box.class_name
        self._write_class_map()
        if self.fmt == "json":
            return self._save_json(annotation)
        return self._save_yolo(annotation)

    def _save_yolo(self, ann: FrameAnnotation) -> str:
        path = self._label_path(ann.frame_index, ext=".txt")
        with open(path, "w") as f:
            for box in ann.boxes:
                f.write(box.to_yolo_line() + "\n")
        return path

    def _save_json(self, ann: FrameAnnotation) -> str:
        path = self._label_path(ann.frame_index, ext=".json")
        data = {
            "frame_index": ann.frame_index,
            "frame_path":  ann.frame_path,
            "boxes": [
                {
                    "class_id":   b.class_id,
                    "class_name": b.class_name,
                    "x_center":   b.x_center,
                    "y_center":   b.y_center,
                    "width":      b.width,
                    "height":     b.height,
                    "confidence": b.confidence,
                }
                for b in ann.boxes
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self, frame_path: str) -> list[BoundingBox] | None:
        """Try to load labels for a frame identified by its image path."""
        frame_index = self._index_from_path(frame_path)
        if frame_index is None:
            return None
        if self.fmt == "json":
            return self._load_json(frame_index)
        return self._load_yolo(frame_index)

    def _load_yolo(self, frame_index: int) -> list[BoundingBox]:
        path = self._label_path(frame_index, ext=".txt")
        boxes = []
        if not os.path.exists(path):
            return boxes
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid = int(parts[0])
                    boxes.append(
                        BoundingBox(
                            class_id   = cid,
                            class_name = self._class_map.get(cid, str(cid)),
                            x_center   = float(parts[1]),
                            y_center   = float(parts[2]),
                            width      = float(parts[3]),
                            height     = float(parts[4]),
                        )
                    )
        return boxes

    def _load_json(self, frame_index: int) -> list[BoundingBox]:
        path = self._label_path(frame_index, ext=".json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        return [
            BoundingBox(
                class_id   = b["class_id"],
                class_name = b["class_name"],
                x_center   = b["x_center"],
                y_center   = b["y_center"],
                width      = b["width"],
                height     = b["height"],
                confidence = b.get("confidence", 1.0),
            )
            for b in data.get("boxes", [])
        ]

    # ── class map sidecar ─────────────────────────────────────────────────────
    def _classes_path(self) -> str:
        return os.path.join(self.base_dir, self.CLASSES_FILE)

    def _load_class_map(self) -> dict:
        path = self._classes_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                raw = json.load(f)
            return {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, OSError):
            return {}

    def _write_class_map(self):
        if not self._class_map:
            return
        with open(self._classes_path(), "w") as f:
            json.dump(
                {str(k): v for k, v in self._class_map.items()},
                f, indent=2, sort_keys=True,
            )

    # ── helpers ───────────────────────────────────────────────────────────────
    def _label_path(self, frame_index: int, ext: str = ".txt") -> str:
        return os.path.join(self.base_dir, f"frame_{frame_index:06d}{ext}")

    @staticmethod
    def _index_from_path(frame_path: str) -> int | None:
        """Extract 000042 → 42 from 'frame_000042.png'."""
        base = os.path.splitext(os.path.basename(frame_path))[0]
        try:
            return int(base.split("_")[-1])
        except (ValueError, IndexError):
            return None
