"""Sanity checks for utils.config defaults."""
import os

from utils.config import (
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
