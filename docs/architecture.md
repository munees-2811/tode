# Architecture

`tode` is organised as a thin Tk UI on top of a fully headless core. The
package layout mirrors the runtime layers — UI never reaches into another
UI module's internals, and the headless layers never import Tk.

```
┌─────────────────────────────────────────────────────────────────┐
│  ui/        Tk widgets, dialogs, main window                     │  ← display layer
├─────────────────────────────────────────────────────────────────┤
│  core/      annotation_manager, exporter, yolo_annotator,        │
│             detectors/{onnx,ultralytics}, loaders                │  ← headless app logic
├─────────────────────────────────────────────────────────────────┤
│  storage/   label_storage (YOLO/JSON), frame_storage (PNG)       │  ← persistence
├─────────────────────────────────────────────────────────────────┤
│  models/    BoundingBox, FrameAnnotation (pure dataclasses)      │  ← pure types
│  utils/     config, image_utils, logger                          │  ← shared helpers
└─────────────────────────────────────────────────────────────────┘
```

## Detector backends

`core/detectors/__init__.py` is intentionally empty (no eager imports)
so that the ONNX path never pulls in `torch` / `ultralytics`. Import the
concrete backend you need directly:

```python
from core.detectors.onnx_detector        import ONNXDetector        # MIT
from core.detectors.ultralytics_detector import UltralyticsDetector # AGPL-3.0
```

`core.yolo_annotator.YOLOAnnotator` picks one of the two based on the
weights file extension:

- `*.onnx` → `ONNXDetector` (no torch dependency at runtime)
- `*.pt`   → `UltralyticsDetector` (lazy-imports `ultralytics`)

## Why this matters

The split lets the CPU and `onnx` docker images skip `torch` and
`ultralytics` entirely, keeping image size under ~600 MB and avoiding
AGPL-3.0 entanglement for downstream products that only ever ship `.onnx`
weights.
