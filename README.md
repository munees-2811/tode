# Video Annotation Tool

A desktop application for annotating video frames and images with bounding boxes, powered by YOLO auto-annotation.

---

## Features

- **Auto-annotation** — one-click YOLO inference on a single frame or all frames at once
- **YOLO model selector** — choose any YOLO26 / YOLO11 / YOLOv8 variant from a dropdown, or browse for a local `.pt` file
- **Manual annotation** — click and drag to draw bounding boxes; assign any class name
- **Video & image support** — load MP4/AVI/MOV/MKV videos, single images, or entire image folders (recursive)
- **YouTube download** — paste a URL, preview metadata, choose quality, and download before annotating
- **YOLO-format labels** — annotations saved as standard `.txt` files (class cx cy w h, normalised) compatible with Ultralytics training pipelines
- **Class names persist** — a `classes.json` sidecar keeps label names across sessions
- **Log viewer** — live in-app log window for debugging

---

## Requirements

| Dependency | Version |
|---|---|
| Python | **3.11 or 3.12** recommended (3.14 has no prebuilt wheels for cv2/ultralytics) |
| ultralytics | ≥ 8.0.0 |
| opencv-python | ≥ 4.8.0 |
| Pillow | ≥ 10.0.0 |
| numpy | ≥ 1.24.0 |
| yt-dlp | ≥ 2024.1.0 (YouTube feature only) |

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/tedo001/Video_Annotaion.git
cd Video_Annotaion

# 2. Create a virtual environment (Python 3.11 or 3.12)
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

### PyCharm setup

1. `Settings` → `Project` → `Python Interpreter` → `Add Interpreter` → `Virtualenv`
2. Select **Python 3.11 or 3.12** as the base interpreter
3. Open the built-in terminal → `pip install -r requirements.txt`
4. Run `main.py`

---

## Loading a local video

**Quickest way — click the canvas:**
> When no source is loaded the canvas shows a `📂` prompt. Click it to open a file picker.

**Toolbar buttons:**

| Button | Action |
|---|---|
| `📂 Open` | Tabbed dialog — Video / Image / Image Folder / YouTube |
| `🎬 Video` | Direct file picker for video files |
| `🖼 Image` | Pick a single image or an image folder |
| `▶ YouTube` | Opens the YouTube download tab directly |

**Supported video formats:** MP4, AVI, MOV, MKV, WEBM, FLV, WMV

---

## Choosing a YOLO model

The **ANNOTATION PANEL → Auto (YOLO)** tab has a model dropdown at the top.

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `yolo26x` | XLarge | Slow | Best |
| `yolo26l` | Large | Medium | High |
| `yolo26m` | Medium | Fast | Good |
| `yolo26s` | Small | Faster | OK |
| `yolo26n` | Nano | Fastest | Basic |

Models are **auto-downloaded** from the Ultralytics hub on first use.

**Use a local `.pt` file:** click the `📂` button next to the dropdown and browse to your weights file.

---

## Annotating

### Auto (YOLO)

1. Load a video or image folder
2. Adjust the **Confidence Threshold** slider (default 0.45)
3. Optionally type class names in **Filter Classes** (comma-separated) to keep only those classes
4. Click **⚡ YOLO This Frame** for the current frame, or **🔁 YOLO All Frames** to process everything

### Manual

1. Switch to the **✏ Manual** tab in the annotation panel
2. Select a class from the dropdown (populated from the loaded YOLO model), or type a custom class name
3. Click **✏ Draw Box** in the mode bar above the canvas
4. Click and drag on the frame to draw a box
5. Repeat across frames, then click **💾 Save Annotations**

---

## Output structure

```
output/
  frames/
    <video_name>/
      frame_000000.png
      frame_000001.png
      ...
  labels/
    <video_name>/
      frame_000000.txt    ← YOLO format label
      frame_000001.txt
      classes.json        ← class id → name mapping
```

Label format (`.txt`):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalised to `[0, 1]`. One line per bounding box.

---

## Project structure

```
Video_Annotaion/
├── main.py                     # entry point
├── requirements.txt
├── core/
│   ├── video_loader.py         # OpenCV video I/O
│   ├── frame_extractor.py      # frame-stepping from video
│   ├── image_loader.py         # single image / folder loader (recursive)
│   ├── image_frame_extractor.py
│   ├── yolo_annotator.py       # YOLO inference wrapper (thread-safe)
│   ├── annotation_manager.py   # orchestrates the full pipeline
│   └── youtube_downloader.py   # yt-dlp wrapper
├── models/
│   └── annotation_model.py     # BoundingBox, FrameAnnotation dataclasses
├── storage/
│   ├── frame_storage.py        # saves extracted frames to disk
│   └── label_storage.py        # reads/writes YOLO .txt + classes.json
├── ui/
│   ├── main_window.py          # root frame, toolbar, event wiring
│   ├── video_player.py         # canvas + navigation controls
│   ├── annotation_panel.py     # right panel (YOLO settings, box list)
│   ├── source_dialog.py        # tabbed open-source dialog
│   ├── label_editor.py         # label rename dialog
│   └── log_viewer.py           # live log window
└── utils/
    ├── config.py               # paths, constants, model catalogue
    ├── image_utils.py          # draw_boxes, resize, BGR→PhotoImage
    └── logger.py               # rotating file + coloured console + GUI queue
```

---

## License

MIT
