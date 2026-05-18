"""
tests/test_functional.py
─────────────────────────
Headless functional tests for all non-UI layers of tode.
No display, no camera, no real model weights required.

Run with:
    cd /home/user/tode && python -m pytest tests/test_functional.py -v
"""
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
import pytest

# ── Make sure project root is on sys.path ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ═══════════════════════════════════════════════════════════════════════════
# 1. MODELS — BoundingBox & FrameAnnotation
# ═══════════════════════════════════════════════════════════════════════════

from models.annotation_model import BoundingBox, FrameAnnotation  # noqa: E402


class TestBoundingBox:
    def _make(self, **kw):
        defaults = dict(
            class_id=0, class_name="car",
            x_center=0.5, y_center=0.5,
            width=0.4, height=0.3,
            confidence=0.9,
        )
        return BoundingBox(**{**defaults, **kw})

    def test_to_pixel_coords_center(self):
        box = self._make(x_center=0.5, y_center=0.5, width=0.4, height=0.2)
        x1, y1, x2, y2 = box.to_pixel_coords(100, 100)
        assert x1 == 30 and y1 == 40
        assert x2 == 70 and y2 == 60

    def test_to_pixel_coords_top_left(self):
        box = self._make(x_center=0.1, y_center=0.1, width=0.2, height=0.2)
        x1, y1, x2, y2 = box.to_pixel_coords(100, 100)
        assert x1 == 0 and y1 == 0
        assert x2 == 20 and y2 == 20

    def test_to_yolo_line_format(self):
        box = self._make(class_id=3, x_center=0.5, y_center=0.25, width=0.8, height=0.1)
        line = box.to_yolo_line()
        parts = line.strip().split()
        assert parts[0] == "3"
        assert float(parts[1]) == pytest.approx(0.5, rel=1e-4)
        assert float(parts[2]) == pytest.approx(0.25, rel=1e-4)
        assert float(parts[3]) == pytest.approx(0.8, rel=1e-4)
        assert float(parts[4]) == pytest.approx(0.1, rel=1e-4)

    def test_default_confidence_is_one(self):
        box = BoundingBox(0, "obj", 0.5, 0.5, 0.1, 0.1)
        assert box.confidence == 1.0

    def test_to_pixel_coords_rectangular_frame(self):
        box = self._make(x_center=0.5, y_center=0.5, width=1.0, height=1.0)
        x1, y1, x2, y2 = box.to_pixel_coords(1920, 1080)
        assert x1 == 0 and y1 == 0
        assert x2 == 1920 and y2 == 1080


class TestFrameAnnotation:
    def _box(self, name="car"):
        return BoundingBox(0, name, 0.5, 0.5, 0.2, 0.2, 0.9)

    def test_starts_not_annotated(self):
        ann = FrameAnnotation(0, "/tmp/frame.png")
        assert not ann.is_annotated
        assert ann.boxes == []

    def test_add_box_sets_annotated(self):
        ann = FrameAnnotation(0, "/tmp/frame.png")
        ann.add_box(self._box())
        assert ann.is_annotated
        assert len(ann.boxes) == 1

    def test_remove_box(self):
        ann = FrameAnnotation(0, "/tmp/frame.png")
        ann.add_box(self._box("cat"))
        ann.add_box(self._box("dog"))
        ann.remove_box(0)
        assert len(ann.boxes) == 1
        assert ann.boxes[0].class_name == "dog"

    def test_remove_last_box_clears_annotated(self):
        ann = FrameAnnotation(0, "/tmp/frame.png")
        ann.add_box(self._box())
        ann.remove_box(0)
        assert not ann.is_annotated

    def test_remove_out_of_range_is_safe(self):
        ann = FrameAnnotation(0, "/tmp/frame.png")
        ann.remove_box(99)  # must not raise

    def test_clear_boxes(self):
        ann = FrameAnnotation(0, "/tmp/frame.png")
        ann.add_box(self._box())
        ann.add_box(self._box())
        ann.clear_boxes()
        assert ann.boxes == []
        assert not ann.is_annotated


# ═══════════════════════════════════════════════════════════════════════════
# 2. STORAGE — LabelStorage
# ═══════════════════════════════════════════════════════════════════════════

from storage.label_storage import LabelStorage  # noqa: E402


class TestLabelStorageYOLO:
    """YOLO .txt round-trip."""

    @pytest.fixture(autouse=True)
    def tmp_labels(self, tmp_path, monkeypatch):
        import utils.config as cfg
        monkeypatch.setattr(cfg, "LABELS_DIR", str(tmp_path))
        import storage.label_storage as ls_mod
        monkeypatch.setattr(ls_mod, "LABELS_DIR", str(tmp_path))

    def _storage(self, tmp_path):
        from storage.label_storage import LabelStorage as LS
        import storage.label_storage as ls_mod
        ls_mod.LABELS_DIR = str(tmp_path)
        return LS.__new__(LS)

    def test_yolo_roundtrip(self, tmp_path):
        storage = LabelStorage.__new__(LabelStorage)
        storage.video_name = "test_vid"
        storage.fmt        = "yolo"
        storage.base_dir   = str(tmp_path / "test_vid")
        os.makedirs(storage.base_dir, exist_ok=True)
        storage._class_map = {}

        box1 = BoundingBox(0, "car",  0.5, 0.5, 0.4, 0.3, 0.9)
        box2 = BoundingBox(1, "person", 0.2, 0.3, 0.1, 0.2, 0.7)
        ann  = FrameAnnotation(42, str(tmp_path / "frame_000042.png"))
        ann.add_box(box1)
        ann.add_box(box2)

        path  = storage.save(ann)
        assert os.path.exists(path)

        boxes = storage.load(str(tmp_path / "frame_000042.png"))
        assert boxes is not None
        assert len(boxes) == 2
        assert boxes[0].class_name == "car"
        assert boxes[1].class_name == "person"
        assert boxes[0].x_center == pytest.approx(0.5, rel=1e-4)

    def test_yolo_empty_frame_loads_empty(self, tmp_path):
        storage = LabelStorage.__new__(LabelStorage)
        storage.video_name = "test_vid"
        storage.fmt        = "yolo"
        storage.base_dir   = str(tmp_path / "test_vid")
        os.makedirs(storage.base_dir, exist_ok=True)
        storage._class_map = {}

        boxes = storage.load(str(tmp_path / "frame_000007.png"))
        assert boxes == []

    def test_class_map_sidecar_written(self, tmp_path):
        storage = LabelStorage.__new__(LabelStorage)
        storage.video_name = "v"
        storage.fmt        = "yolo"
        storage.base_dir   = str(tmp_path / "v")
        os.makedirs(storage.base_dir, exist_ok=True)
        storage._class_map = {}

        ann = FrameAnnotation(1, str(tmp_path / "frame_000001.png"))
        ann.add_box(BoundingBox(0, "truck", 0.5, 0.5, 0.2, 0.2))
        storage.save(ann)

        classes_path = os.path.join(storage.base_dir, "classes.json")
        assert os.path.exists(classes_path)
        with open(classes_path) as f:
            data = json.load(f)
        assert data["0"] == "truck"

    def test_unknown_class_id_falls_back_to_string(self, tmp_path):
        storage = LabelStorage.__new__(LabelStorage)
        storage.video_name = "v"
        storage.fmt        = "yolo"
        storage.base_dir   = str(tmp_path / "v")
        os.makedirs(storage.base_dir, exist_ok=True)
        storage._class_map = {}

        # Write a label file manually with class 99 (no name in map)
        lbl = os.path.join(storage.base_dir, "frame_000001.txt")
        with open(lbl, "w") as f:
            f.write("99 0.5 0.5 0.2 0.2\n")

        boxes = storage.load(str(tmp_path / "frame_000001.png"))
        assert len(boxes) == 1
        assert boxes[0].class_name == "99"

    def test_index_from_path_parses_correctly(self):
        from storage.label_storage import LabelStorage as LS
        assert LS._index_from_path("/some/path/frame_000042.png") == 42
        assert LS._index_from_path("/some/path/frame_000001.png") == 1
        assert LS._index_from_path("/no_frame_here.png") is None

    def test_json_roundtrip(self, tmp_path):
        storage = LabelStorage.__new__(LabelStorage)
        storage.video_name = "v"
        storage.fmt        = "json"
        storage.base_dir   = str(tmp_path / "v")
        os.makedirs(storage.base_dir, exist_ok=True)
        storage._class_map = {}

        box = BoundingBox(2, "bike", 0.3, 0.4, 0.15, 0.1, 0.85)
        ann = FrameAnnotation(5, str(tmp_path / "frame_000005.png"))
        ann.add_box(box)
        storage.save(ann)

        boxes = storage.load(str(tmp_path / "frame_000005.png"))
        assert len(boxes) == 1
        assert boxes[0].class_name == "bike"
        assert boxes[0].confidence == pytest.approx(0.85, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# 3. STORAGE — FrameStorage
# ═══════════════════════════════════════════════════════════════════════════

from storage.frame_storage import FrameStorage  # noqa: E402


class TestFrameStorage:
    def _make(self, tmp_path):
        storage = FrameStorage.__new__(FrameStorage)
        storage.video_name = "vid"
        storage.base_dir   = str(tmp_path / "vid")
        os.makedirs(storage.base_dir, exist_ok=True)
        return storage

    def test_save_and_load(self, tmp_path):
        fs    = self._make(tmp_path)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[50, 50] = [0, 255, 0]
        path  = fs.save(0, frame)
        assert os.path.exists(path)
        loaded = fs.load(0)
        assert loaded is not None
        assert loaded.shape == (100, 100, 3)

    def test_load_missing_returns_none(self, tmp_path):
        fs = self._make(tmp_path)
        assert fs.load(999) is None

    def test_exists(self, tmp_path):
        fs    = self._make(tmp_path)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        assert not fs.exists(0)
        fs.save(0, frame)
        assert fs.exists(0)

    def test_list_saved_indices(self, tmp_path):
        fs    = self._make(tmp_path)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in [3, 7, 1]:
            fs.save(i, frame)
        assert fs.list_saved_indices() == [1, 3, 7]

    def test_frame_path_format(self, tmp_path):
        fs   = self._make(tmp_path)
        path = fs._frame_path(42)
        assert path.endswith("frame_000042.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. CORE — DatasetExporter (YOLO + COCO)
# ═══════════════════════════════════════════════════════════════════════════

from core.exporter import DatasetExporter  # noqa: E402


class TestDatasetExporter:
    @pytest.fixture()
    def setup(self, tmp_path):
        # Create fake source images on disk
        imgs = []
        for i in range(3):
            p = str(tmp_path / f"frame_{i:06d}.png")
            cv2.imwrite(p, np.zeros((100, 100, 3), dtype=np.uint8))
            imgs.append(p)

        class_names = {0: "car", 1: "person"}
        annotations = {}
        for i, img_path in enumerate(imgs):
            ann = FrameAnnotation(i, img_path)
            ann.add_box(BoundingBox(0, "car", 0.5, 0.5, 0.4, 0.3, 0.9))
            annotations[i] = ann

        out_dir = str(tmp_path / "export")
        return annotations, class_names, out_dir

    def test_yolo_export_creates_files(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        exp    = DatasetExporter(annotations, class_names, out_dir)
        result = exp.export("yolo")

        assert result["format"] == "yolo"
        assert result["images"] == 3
        assert os.path.isdir(os.path.join(out_dir, "images"))
        assert os.path.isdir(os.path.join(out_dir, "labels"))
        assert os.path.exists(os.path.join(out_dir, "classes.txt"))
        assert os.path.exists(os.path.join(out_dir, "data.yaml"))

    def test_yolo_label_content(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        DatasetExporter(annotations, class_names, out_dir).export("yolo")

        lbl = os.path.join(out_dir, "labels", "img_1.txt")
        assert os.path.exists(lbl)
        with open(lbl) as f:
            line = f.read().strip().split()
        assert line[0] == "0"                             # class id
        assert float(line[1]) == pytest.approx(0.5, rel=1e-4)  # cx

    def test_yolo_data_yaml_content(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        DatasetExporter(annotations, class_names, out_dir).export("yolo")

        with open(os.path.join(out_dir, "data.yaml")) as f:
            content = f.read()
        assert "nc: " in content
        assert "names:" in content

    def test_coco_export_creates_annotations_json(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        result = DatasetExporter(annotations, class_names, out_dir).export("coco")

        assert result["format"] == "coco"
        json_path = os.path.join(out_dir, "annotations.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            coco = json.load(f)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 3
        assert len(coco["annotations"]) == 3

    def test_coco_bbox_pixel_coords(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        DatasetExporter(annotations, class_names, out_dir).export("coco")

        with open(os.path.join(out_dir, "annotations.json")) as f:
            coco = json.load(f)
        bbox = coco["annotations"][0]["bbox"]
        # box: cx=0.5, cy=0.5, w=0.4, h=0.3 → image 100x100
        # x_tl = 0.5*100 - 0.4*100/2 = 30
        # y_tl = 0.5*100 - 0.3*100/2 = 35
        assert bbox[0] == pytest.approx(30.0, abs=0.1)
        assert bbox[1] == pytest.approx(35.0, abs=0.1)
        assert bbox[2] == pytest.approx(40.0, abs=0.1)
        assert bbox[3] == pytest.approx(30.0, abs=0.1)

    def test_unknown_format_raises(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        with pytest.raises(ValueError, match="Unknown export format"):
            DatasetExporter(annotations, class_names, out_dir).export("csv")

    def test_no_annotated_frames_raises(self, tmp_path):
        ann = {0: FrameAnnotation(0, "/fake.png")}  # is_annotated=False
        with pytest.raises(ValueError, match="Nothing to export"):
            DatasetExporter(ann, {}, str(tmp_path / "out")).export("yolo")

    def test_progress_callback_called(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        calls = []
        DatasetExporter(annotations, class_names, out_dir).export(
            "yolo", progress_callback=lambda cur, tot: calls.append((cur, tot))
        )
        assert len(calls) == 3
        assert calls[-1] == (3, 3)

    def test_class_id_map_preserves_model_order(self, tmp_path, setup):
        annotations, class_names, out_dir = setup
        exp = DatasetExporter(annotations, class_names, out_dir)
        id_map = exp._build_class_id_map(list(annotations.values()))
        assert id_map["car"] == 0
        assert id_map["person"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. CORE/DETECTORS — ONNXDetector preprocessing + postprocessing
# ═══════════════════════════════════════════════════════════════════════════

from core.detectors.onnx_detector import ONNXDetector  # noqa: E402


class TestONNXPreprocessing:
    """Validate letterbox preprocessing — no ONNX weights needed."""

    def test_output_tensor_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, scale, pad_top, pad_left = ONNXDetector._preprocess(frame)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_pixel_values_normalized(self):
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        tensor, *_ = ONNXDetector._preprocess(frame)
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

    def test_square_input_no_padding(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _, scale, pad_top, pad_left = ONNXDetector._preprocess(frame)
        assert scale == pytest.approx(1.0)
        assert pad_top == 0
        assert pad_left == 0

    def test_landscape_frame_pads_top_bottom(self):
        # 1280×480 wide frame — scale = 640/1280 = 0.5, new_h = 240, pad = 200
        frame = np.zeros((480, 1280, 3), dtype=np.uint8)
        _, scale, pad_top, pad_left = ONNXDetector._preprocess(frame)
        assert scale == pytest.approx(0.5)
        assert pad_top == 200
        assert pad_left == 0

    def test_portrait_frame_pads_sides(self):
        # 320×640 tall frame — scale = 1.0, new_w = 320, pad_left = 160
        frame = np.zeros((640, 320, 3), dtype=np.uint8)
        _, scale, pad_top, pad_left = ONNXDetector._preprocess(frame)
        assert scale == pytest.approx(1.0)
        assert pad_left == 160
        assert pad_top == 0

    def test_channel_order_is_rgb(self):
        # Create a purely red BGR frame: [0, 0, 255]
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :] = [0, 0, 255]  # BGR red
        tensor, *_ = ONNXDetector._preprocess(frame)
        # After BGR→RGB, channel 0 should be 1.0 (red), channels 1&2 should be 0
        assert tensor[0, 0].max() == pytest.approx(1.0)   # R channel
        assert tensor[0, 1].max() == pytest.approx(0.0)   # G channel
        assert tensor[0, 2].max() == pytest.approx(0.0)   # B channel


class TestONNXClassNames:
    def test_sidecar_json_loaded(self, tmp_path):
        onnx_path = str(tmp_path / "model.onnx")
        sidecar   = str(tmp_path / "model_classes.json")
        with open(sidecar, "w") as f:
            json.dump({"0": "helmet", "1": "vest"}, f)

        names = ONNXDetector._load_class_names(onnx_path)
        assert names[0] == "helmet"
        assert names[1] == "vest"

    def test_no_sidecar_falls_back_to_coco80(self, tmp_path):
        onnx_path = str(tmp_path / "noclass.onnx")
        names = ONNXDetector._load_class_names(onnx_path)
        assert names[0] == "person"
        assert names[2] == "car"
        assert len(names) == 80

    def test_is_not_loaded_by_default(self):
        det = ONNXDetector()
        assert not det.is_loaded()

    def test_detect_returns_empty_when_not_loaded(self):
        det   = ONNXDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert det.detect(frame) == []

    def test_backend_name(self):
        assert ONNXDetector().backend_name == "ONNX Runtime"


# ═══════════════════════════════════════════════════════════════════════════
# 6. CORE — YOLOAnnotator routing logic (no real weights)
# ═══════════════════════════════════════════════════════════════════════════

from core.yolo_annotator import YOLOAnnotator  # noqa: E402


class TestYOLOAnnotatorRouting:
    """Test detector selection logic without loading real model weights."""

    def test_onnx_path_creates_onnx_detector(self, tmp_path, monkeypatch):
        from core.detectors.onnx_detector import ONNXDetector as OD

        monkeypatch.setattr(OD, "_load_session", lambda self, p: None)
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert isinstance(ann._detector, OD)

    def test_starts_unloaded(self, tmp_path, monkeypatch):
        from core.detectors.onnx_detector import ONNXDetector as OD
        monkeypatch.setattr(OD, "_load_session", lambda self, p: None)
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert not ann.is_loaded()

    def test_model_path_stored(self, tmp_path, monkeypatch):
        from core.detectors.onnx_detector import ONNXDetector as OD
        monkeypatch.setattr(OD, "_load_session", lambda self, p: None)
        path = str(tmp_path / "model.onnx")
        ann  = YOLOAnnotator(model_path=path)
        assert ann.model_path == path

    def test_backend_name_onnx(self, tmp_path, monkeypatch):
        from core.detectors.onnx_detector import ONNXDetector as OD
        monkeypatch.setattr(OD, "_load_session", lambda self, p: None)
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert ann.backend_name == "ONNX Runtime"

    def test_annotate_frame_returns_empty_if_not_loaded(self, tmp_path, monkeypatch):
        from core.detectors.onnx_detector import ONNXDetector as OD
        monkeypatch.setattr(OD, "_load_session", lambda self, p: None)
        ann   = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # load() will call _load_session (patched to no-op), is_loaded still False
        assert ann.annotate_frame(frame) == []


# ═══════════════════════════════════════════════════════════════════════════
# 7. UTILS — image_utils
# ═══════════════════════════════════════════════════════════════════════════

from utils.image_utils import draw_boxes, hex_to_bgr, resize_frame  # noqa: E402


class TestHexToBgr:
    def test_white(self):
        assert hex_to_bgr("#ffffff") == (255, 255, 255)

    def test_black(self):
        assert hex_to_bgr("#000000") == (0, 0, 0)

    def test_pure_red(self):
        # HTML red #ff0000 → R=255, G=0, B=0 → BGR=(0, 0, 255)
        assert hex_to_bgr("#ff0000") == (0, 0, 255)

    def test_pure_green(self):
        assert hex_to_bgr("#00ff00") == (0, 255, 0)

    def test_pure_blue(self):
        assert hex_to_bgr("#0000ff") == (255, 0, 0)

    def test_without_hash(self):
        assert hex_to_bgr("00ff88") == hex_to_bgr("#00ff88")


class TestResizeFrame:
    def test_output_dimensions(self):
        frame   = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = resize_frame(frame, 640, 480)
        assert resized.shape == (480, 640, 3)

    def test_square_input_no_distortion(self):
        frame   = np.full((200, 200, 3), 128, dtype=np.uint8)
        resized = resize_frame(frame, 100, 100)
        assert resized.shape == (100, 100, 3)

    def test_letterbox_preserves_aspect_ratio(self):
        # 200×100 frame (2:1) into 100×100 canvas
        # Scale = min(100/200, 100/100) = 0.5 → new size 100×50
        # Vertical padding: (100-50)//2 = 25 top and bottom
        frame = np.full((100, 200, 3), 200, dtype=np.uint8)
        out   = resize_frame(frame, 100, 100)
        # The first 25 rows should be black (padding)
        assert np.all(out[:25, :] == 0)
        # The middle row should have content (not black)
        assert out[50, 50, 0] > 0

    def test_returns_copy_not_original(self):
        frame   = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = resize_frame(frame, 100, 100)
        frame[0, 0] = [255, 0, 0]
        assert resized[0, 0, 0] == 0  # resized is not affected


class TestDrawBoxes:
    def _box(self):
        return BoundingBox(0, "car", 0.5, 0.5, 0.4, 0.4, 0.9)

    def test_returns_copy_not_original(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out   = draw_boxes(frame, [self._box()])
        assert not np.array_equal(frame, out)

    def test_original_unchanged(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_boxes(frame, [self._box()])
        assert np.all(frame == 0)

    def test_no_boxes_returns_identical(self):
        frame = np.full((100, 100, 3), 42, dtype=np.uint8)
        out   = draw_boxes(frame, [])
        assert np.array_equal(frame, out)

    def test_box_drawn_modifies_pixels(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out   = draw_boxes(frame, [self._box()], color="#00ff00")
        assert out.sum() > 0  # something was drawn

    def test_multiple_boxes(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        boxes = [
            BoundingBox(0, "a", 0.25, 0.25, 0.2, 0.2, 0.9),
            BoundingBox(1, "b", 0.75, 0.75, 0.2, 0.2, 0.8),
        ]
        out = draw_boxes(frame, boxes)
        assert out.sum() > 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. UTILS — config
# ═══════════════════════════════════════════════════════════════════════════

from utils.config import (  # noqa: E402
    ACCENT,
    BG_DARK,
    BG_PANEL,
    FRAMES_DIR,
    LABEL_FORMAT,
    LABELS_DIR,
    OUTPUT_DIR,
    WEIGHTS_DIR,
    YOLO_CONFIDENCE,
    YOLO_IOU_THRESHOLD,
    YOLO_MODELS,
)


class TestConfig:
    def test_directories_exist(self):
        for d in (OUTPUT_DIR, FRAMES_DIR, LABELS_DIR, WEIGHTS_DIR):
            assert os.path.isdir(d), f"Expected directory: {d}"

    def test_yolo_models_is_list(self):
        assert isinstance(YOLO_MODELS, list)
        assert len(YOLO_MODELS) >= 1

    def test_inference_defaults_in_range(self):
        assert 0.0 < YOLO_CONFIDENCE < 1.0
        assert 0.0 < YOLO_IOU_THRESHOLD < 1.0

    def test_label_format_is_yolo(self):
        assert LABEL_FORMAT == "yolo"

    def test_color_strings_are_hex(self):
        for color in (BG_DARK, BG_PANEL, ACCENT):
            assert color.startswith("#")
            assert len(color) == 7


# ═══════════════════════════════════════════════════════════════════════════
# 9. CORE — ImageLoader basic behaviour (no real video file)
# ═══════════════════════════════════════════════════════════════════════════

from core.image_loader import ImageLoader  # noqa: E402


class TestImageLoader:
    def test_load_single_image(self, tmp_path):
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
        loader = ImageLoader(img_path)
        loader.open()
        assert loader.total_frames == 1
        assert loader.fps == 1.0

    def test_load_folder(self, tmp_path):
        for i in range(4):
            cv2.imwrite(
                str(tmp_path / f"img_{i:03d}.jpg"),
                np.zeros((50, 50, 3), dtype=np.uint8),
            )
        loader = ImageLoader(str(tmp_path))
        loader.open()
        assert loader.total_frames == 4

    def test_read_frame_returns_ndarray(self, tmp_path):
        img_path = str(tmp_path / "frame.png")
        cv2.imwrite(img_path, np.full((80, 60, 3), 128, dtype=np.uint8))
        loader = ImageLoader(img_path)
        loader.open()
        frame = loader.read_frame(0)
        assert frame is not None
        assert frame.shape == (80, 60, 3)

    def test_read_out_of_range_returns_none(self, tmp_path):
        img_path = str(tmp_path / "only.png")
        cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype=np.uint8))
        loader = ImageLoader(img_path)
        loader.open()
        assert loader.read_frame(99) is None

    def test_is_open_after_open(self, tmp_path):
        img_path = str(tmp_path / "x.png")
        cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype=np.uint8))
        loader = ImageLoader(img_path)
        assert not loader.is_open()
        loader.open()
        assert loader.is_open()


# ═══════════════════════════════════════════════════════════════════════════
# 10. CORE — FrameExtractor (actual sequential extraction)
# ═══════════════════════════════════════════════════════════════════════════

from core.image_frame_extractor import ImageFrameExtractor  # noqa: E402


class TestImageFrameExtractor:
    def test_extracts_all_images(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        for i in range(3):
            cv2.imwrite(
                str(src_dir / f"img_{i}.png"),
                np.zeros((10, 10, 3), dtype=np.uint8),
            )

        from core.image_loader import ImageLoader
        loader = ImageLoader(str(src_dir))
        loader.open()

        extractor = ImageFrameExtractor(loader, str(out_dir))
        results = list(extractor.extract())
        assert len(results) == 3
        for idx, frame, path in results:
            assert os.path.exists(path)
            assert frame is not None
