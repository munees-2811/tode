"""ONNXDetector preprocessing/postprocessing + YOLOAnnotator routing."""
import json

import numpy as np
import pytest

from core.detectors.onnx_detector import ONNXDetector
from core.yolo_annotator import YOLOAnnotator


class TestONNXPreprocessing:
    """Validate letterbox preprocessing — no ONNX weights needed."""

    def test_output_tensor_shape(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, _scale, _pad_top, _pad_left = ONNXDetector._preprocess(frame)
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


class TestYOLOAnnotatorRouting:
    """Test detector selection logic without loading real model weights."""

    @pytest.fixture(autouse=True)
    def _patch_load(self, monkeypatch):
        monkeypatch.setattr(ONNXDetector, "_load_session", lambda self, p: None)

    def test_onnx_path_creates_onnx_detector(self, tmp_path):
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert isinstance(ann._detector, ONNXDetector)

    def test_starts_unloaded(self, tmp_path):
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert not ann.is_loaded()

    def test_model_path_stored(self, tmp_path):
        path = str(tmp_path / "model.onnx")
        ann  = YOLOAnnotator(model_path=path)
        assert ann.model_path == path

    def test_backend_name_onnx(self, tmp_path):
        ann = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        assert ann.backend_name == "ONNX Runtime"

    def test_annotate_frame_returns_empty_if_not_loaded(self, tmp_path):
        ann   = YOLOAnnotator(model_path=str(tmp_path / "model.onnx"))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert ann.annotate_frame(frame) == []
