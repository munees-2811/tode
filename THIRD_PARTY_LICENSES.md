# Third-Party Licenses

This project bundles and depends on the following third-party software. Each is
the property of its respective owners and licensed under the terms below.

## Direct Python dependencies

| Package | Version | License | Notes |
|---|---|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | ≥ 8.0.0 | **AGPL-3.0** | **Viral copyleft** — forces this project to AGPL-3.0. Commercial use requires an [Ultralytics Enterprise Licence](https://www.ultralytics.com/license). |
| [opencv-python](https://github.com/opencv/opencv-python) | ≥ 4.8.0 | Apache 2.0 | Permissive. |
| [Pillow](https://github.com/python-pillow/Pillow) | ≥ 10.0.0 | MIT-CMU (HPND) | Permissive. |
| [numpy](https://github.com/numpy/numpy) | ≥ 1.24.0 | BSD-3-Clause | Permissive. |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | ≥ 2024.1.0 | Unlicense (public domain) | No constraints. |
| [requests](https://github.com/psf/requests) | ≥ 2.28.0 | Apache 2.0 | Permissive. |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | ≥ 0.4.9 | BSD-2-Clause | Wrapper only. Bundles an FFmpeg binary — see below. |

## Bundled binaries

| Software | Source | License | Notes |
|---|---|---|---|
| FFmpeg | shipped inside `imageio-ffmpeg` | LGPL-2.1+ (some builds GPL-2.1+) | The build that `imageio-ffmpeg` ships is LGPL-compatible. We use it as a separate binary invoked by yt-dlp — no static linking — so LGPL "dynamic use" applies. |
| Pre-trained YOLO weights (`yolo*.pt`) | downloaded from [Ultralytics hub](https://docs.ultralytics.com/models/) at runtime | AGPL-3.0 (weights inherit the framework licence) | If you redistribute the app with bundled weights, those weights are also covered by AGPL. |

## Model accuracy / dataset attribution

The pre-trained YOLO weights are trained on **COCO** ([cocodataset.org](https://cocodataset.org)) — Creative Commons Attribution 4.0. Models inherit dataset attribution requirements when used to publish dataset-derived predictions.

## User-supplied content

- **YouTube downloads** — yt-dlp downloads videos but **does not grant you rights to them**. YouTube's Terms of Service prohibit redistributing copyrighted content. Use only with videos you own or that are licensed (CC, public-domain, your own uploads). The maintainers of this project accept no responsibility for ToS violations by end-users.
- **Annotated images** — the user retains all rights to images and labels they create with this tool. Exported datasets are the user's property.

## Why AGPL-3.0 for this project

Ultralytics YOLO is AGPL-3.0. Section 13 of the AGPL ("Remote Network Interaction") means that **any project linking against AGPL code must itself be AGPL-licensed when distributed**, including over a network. Since this app links directly against `ultralytics`, AGPL-3.0 is the only legally compatible licence for redistribution unless you purchase an [Ultralytics Enterprise Licence](https://www.ultralytics.com/license).

If you want a permissive licence for commercial closed-source use, you would need to:
1. Buy the Ultralytics Enterprise Licence, **or**
2. Replace `ultralytics` with a permissively-licensed detector (e.g. plain PyTorch with custom weights, MMDetection under Apache 2.0, etc.).
