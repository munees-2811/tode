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
- **Dataset export** — one-click export to **YOLO** (images/ + labels/ + data.yaml) or **COCO** (single annotations.json) format. Non-annotated frames are skipped automatically
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
git clone https://github.com/tedo001/tode_anotation.v1.1.1.git
cd tode_anotation.v1.1.1

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

### Instant video open

Long videos open **instantly** — the frame index is built in memory before any decoding happens, so you can start annotating right away. Frame PNGs are extracted in a background thread; the status bar shows progress (`Extracting frames in background… 240/3000`). Any frame you navigate to before the worker reaches it is decoded on-demand and cached.

---

## Box editing

After drawing or YOLO-detecting a box you can fix it without redrawing:

1. **Select** — click anywhere inside a box. It highlights in orange with 8 resize handles (4 corners, 4 edges).
2. **Resize** — drag any corner or edge handle.
3. **Move** — drag the body of a selected box.
4. **Deselect** — click an empty area, or switch to View mode.
5. **From the list** — clicking a row in the **DETECTED BOXES** list also selects that box on the canvas. The selection stays in sync both ways.

Live preview while dragging; release commits the change.

---

## Keyboard shortcuts (labelImg-style)

| Key | Action |
|---|---|
| `A` / `←` | Previous frame |
| `D` / `→` | Next frame |
| `Home` / `End` | Jump to first / last frame |
| `W` | Switch to **Draw Box** mode |
| `V` / `Esc` | Switch back to **View** mode |
| `Y` | Run YOLO on the current frame |
| `Ctrl+S` | Save annotations |
| `Ctrl+E` | Export dataset |
| `Ctrl+O` | Open source dialog |
| `Delete` | Clear all boxes on the current frame |

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

## Exporting (YOLO or COCO)

Click **📤 Export** in the toolbar to package your annotations as a training-ready dataset.

A dialog asks for:
- **Format** — YOLO or COCO
- **Output folder** — defaults to `output/exports/`

**Only annotated frames are exported.** Frames with no boxes are skipped entirely, so image and label files always match 1-to-1 (`img_000003.png` ↔ `img_000003.txt`).

**Default location:** `~/Documents/labeled_img/<source_name>/` — easy to find. You can browse to any other folder in the dialog.

### YOLO export layout

```
~/Documents/labeled_img/<source_name>/
├── images/
│   ├── img_1.png
│   ├── img_2.png
│   └── img_3.png              ← sequential 1-based naming
├── labels/                    ← labels match images 1-to-1
│   ├── img_1.txt
│   ├── img_2.txt
│   └── img_3.txt
├── classes.txt                ← one class name per line
└── data.yaml                  ← Ultralytics dataset config
```

Non-annotated frames are skipped entirely, then the remaining annotated frames are renumbered `1..N` in original order.

`data.yaml` is ready for training:
```bash
yolo train data=export_dir/data.yaml model=yolo26x.pt epochs=100
```

### COCO export layout

```
export_dir/
├── images/
│   ├── img_000000.png
│   └── img_000002.png
└── annotations.json           ← COCO JSON (images + annotations + categories)
```

The JSON contains the full COCO schema:
- `images` — file_name, width, height, id
- `annotations` — bbox in `[x_top_left, y_top_left, width, height]` pixels, area, category_id, image_id
- `categories` — id (1-based), name

---

## Output structure (raw working directory)

```
output/
├── frames/
│   └── <video_name>/
│       ├── frame_000000.png
│       └── frame_000001.png
├── labels/
│   └── <video_name>/
│       ├── frame_000000.txt    ← YOLO format label (working copy)
│       ├── frame_000001.txt
│       └── classes.json        ← class id → name mapping
└── exports/                    ← created by the Export button
    └── <your_export_name>/
```

Label format (`.txt`):
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalised to `[0, 1]`. One line per bounding box.

---

## Project structure

```
tode_anotation.v1.1.1/
├── main.py                     # entry point
├── requirements.txt
├── core/
│   ├── video_loader.py         # OpenCV video I/O
│   ├── frame_extractor.py      # frame-stepping from video
│   ├── image_loader.py         # single image / folder loader (recursive)
│   ├── image_frame_extractor.py
│   ├── yolo_annotator.py       # YOLO inference wrapper (thread-safe)
│   ├── annotation_manager.py   # orchestrates the full pipeline
│   ├── exporter.py             # YOLO / COCO dataset export
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
│   ├── export_dialog.py        # YOLO / COCO export dialog
│   ├── label_editor.py         # label rename dialog
│   └── log_viewer.py           # live log window
└── utils/
    ├── config.py               # paths, constants, model catalogue
    ├── image_utils.py          # draw_boxes, resize, BGR→PhotoImage
    └── logger.py               # rotating file + coloured console + GUI queue
```

---

## License

This project is licensed under **GNU AGPL-3.0** — see [`LICENSE`](LICENSE).

> **Why AGPL?** Ultralytics YOLO is itself licensed under AGPL-3.0, which is *viral copyleft*. Any project that links against `ultralytics` and is redistributed (including over a network) must be released under AGPL-3.0 too. If you need a permissive licence for closed-source / commercial use, you must either buy the [Ultralytics Enterprise Licence](https://www.ultralytics.com/license) or swap `ultralytics` for a permissively-licensed detector.

See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the full dependency licence table.

### What you can / cannot do

| Action | Allowed? |
|---|---|
| Use the app locally, privately | ✅ Yes |
| Modify the source code | ✅ Yes |
| Share modifications with the community | ✅ Yes — must include AGPL-3.0 source |
| Run as a public web service | ✅ Yes — must publish your modifications under AGPL-3.0 |
| Sell a closed-source / SaaS fork without releasing source | ❌ No — needs Ultralytics Enterprise Licence |
| Train models on **your own** data and use the weights | ✅ Yes — your weights, your data |
| Redistribute YouTube downloads | ❌ No — YouTube ToS, not our licence |

### YouTube content

`yt-dlp` only downloads — it does not grant you any rights over the content. Only use the YouTube tab with videos you own, videos under permissive licences (CC, public domain), or your own uploads. The maintainers accept no responsibility for ToS violations by end-users.

---

## Running with Docker

`Dockerfile` and `docker-compose.yml` give you a reproducible Linux environment with **all dependencies (Torch, Ultralytics, OpenCV, Tk, FFmpeg) pre-installed** — no `pip install`, no Python version issues. Use this if local install fails.

### Build the image

```bash
docker build -t tode-anotation .
```

First build takes ~5–15 min and produces a ~2 GB image (Torch + Ultralytics dominate).

### Run

```bash
# 1. Headless smoke test — confirms everything imports
docker run --rm tode-anotation python -c "from core import YOLOAnnotator; print('OK')"

# 2. Full GUI on Linux (X11 forwarding)
xhost +local:docker
docker compose up
```

Outputs persist on the host:
- `./output/` — frame cache + working labels
- `~/Documents/labeled_img/` — exported datasets

### GPU (NVIDIA)

Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host, then uncomment the `deploy.resources` block in `docker-compose.yml`.

```bash
docker compose up                                  # uses the GPU automatically
# or for direct docker run:
docker run --gpus all --rm tode-anotation python main.py
```

### Platform notes

| Platform | GUI in Docker? | Recommended use |
|---|---|---|
| Linux | ✅ Yes (X11 forwarding) | Full app or headless |
| Windows | ❌ Awkward (needs WSL2 + VcXsrv) | Headless inference only |
| macOS | ❌ No native X11 | Headless inference only |

On Windows / macOS, just run `python main.py` in a venv for the GUI.

---

## Publishing to Docker Hub (manual)

Lets community members run the app with one command instead of cloning + installing:

```bash
docker pull tedo001/tode-anotation:latest
docker run --rm tedo001/tode-anotation python main.py
```

### One-time setup

1. Create a Docker Hub account at <https://hub.docker.com>
2. Generate an access token at <https://hub.docker.com/settings/security>
   (Permissions: **Read & Write** — copy it, it's shown only once)
3. On the host where you'll build, log in once:
   ```bash
   docker login -u tedo001
   # paste the access token when prompted (NOT your password)
   ```

### Publish a new version

```bash
# Build with both a version tag and 'latest'
docker build \
    -t tedo001/tode-anotation:0.1.0 \
    -t tedo001/tode-anotation:latest .

# Push both tags
docker push tedo001/tode-anotation:0.1.0
docker push tedo001/tode-anotation:latest
```

Convention:
- `:latest` — always the newest build (mutable)
- `:0.1.0`, `:0.2.0` — immutable snapshots users can pin to

### After the first push

On <https://hub.docker.com/r/tedo001/tode-anotation> add:
- **Description** — one-liner about the app
- **Source repository link** — `https://github.com/tedo001/tode_anotation.v1.1.1`
  (satisfies the AGPL source-disclosure obligation automatically)

### Image storage cost

Public images are free, unlimited. Anonymous pulls are rate-limited to 100/6h per IP; authenticated pulls (`docker login`) are unlimited.

---

## Packaging as a downloadable desktop app

For end-users who don't have Python installed, build a single-file binary with **PyInstaller**:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed \
    --add-data "utils:utils" --add-data "models:models" \
    --icon=icon.ico \
    main.py
# Output: dist/main(.exe) — copy alongside any required YOLO .pt weights
```

Each platform (Windows / macOS / Linux) needs its own build host. Sign and notarise on macOS, sign on Windows for SmartScreen.

For an updateable release, push tagged versions to GitHub releases (`git tag v0.1.0 && git push --tags`); the README and `LICENSE` are bundled automatically.
