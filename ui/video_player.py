"""
Canvas widget — displays frames + overlays.
Supports two modes:
  VIEW   : navigate frames only
  DRAW   : click-drag to draw a new bounding box
"""
import os
import tkinter as tk
from typing import List, Callable, Optional

import cv2
from models.annotation_model import BoundingBox
from utils.image_utils import draw_boxes, bgr_to_photoimage, resize_frame
from utils.config import BG_DARK, BG_PANEL, ACCENT, TEXT_LIGHT, BOX_COLOR


class VideoPlayer(tk.Frame):
    MODE_VIEW = "view"
    MODE_DRAW = "draw"

    def __init__(self, master, on_frame_change: Callable = None,
                 on_box_drawn: Callable = None,
                 on_open_request: Callable = None,
                 on_box_edited: Callable = None,
                 on_box_selected: Callable = None):
        """
        Parameters
        ----------
        on_frame_change  : callable(frame_index, bgr_frame)
        on_box_drawn     : callable(x1_norm, y1_norm, x2_norm, y2_norm)
        on_box_edited    : callable(box_index, x1_norm, y1_norm, x2_norm, y2_norm)
        on_box_selected  : callable(box_index_or_None)
        """
        super().__init__(master, bg=BG_DARK)
        self._on_change       = on_frame_change
        self._on_box_drawn    = on_box_drawn
        self._on_open_request = on_open_request
        self._on_box_edited   = on_box_edited
        self._on_box_selected = on_box_selected
        self._frame_path_provider: Optional[Callable[[int], str]] = None

        self._loader      = None
        self._indices: List[int] = []
        self._pos         = 0
        self._boxes: List[BoundingBox] = []
        self._photo       = None
        self._current_frame = None

        # ── draw-mode state ───────────────────────────────────────────────────
        self._mode        = self.MODE_VIEW
        self._draw_start: Optional[tuple] = None   # (canvas_x, canvas_y)
        self._draw_rect   = None                   # canvas rect id

        # ── edit-mode state ───────────────────────────────────────────────────
        # selected box index (into self._boxes), edit handle being dragged,
        # and the original box pixel rect at drag-start (so we can update
        # relative to the press point).
        self._selected_idx: Optional[int] = None
        self._edit_handle: Optional[str]  = None      # nw|n|ne|w|e|sw|s|se|move|None
        self._edit_drag_start: Optional[tuple] = None # (mouse_x, mouse_y, px_box)

        # frame→canvas offset (for aspect-ratio letterboxing)
        self._frame_offset_x = 0
        self._frame_offset_y = 0
        self._frame_scale    = 1.0
        self._frame_w        = 1
        self._frame_h        = 1

        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        # ── mode toggle bar ───────────────────────────────────────────────────
        mode_bar = tk.Frame(self, bg=BG_PANEL, height=32)
        mode_bar.pack(fill=tk.X)
        mode_bar.pack_propagate(False)

        tk.Label(
            mode_bar, text="  MODE:", bg=BG_PANEL, fg=TEXT_LIGHT,
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(8, 4), pady=6)

        self._view_btn = tk.Button(
            mode_bar, text="👁 View", command=self.set_view_mode,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=10, font=("Consolas", 9, "bold"), cursor="hand2",
        )
        self._view_btn.pack(side=tk.LEFT, padx=2, pady=4)

        self._draw_btn = tk.Button(
            mode_bar, text="✏ Draw Box", command=self.set_draw_mode,
            bg=BG_DARK, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=10, font=("Consolas", 9, "bold"), cursor="hand2",
        )
        self._draw_btn.pack(side=tk.LEFT, padx=2, pady=4)

        self._mode_label = tk.Label(
            mode_bar, text="  VIEW MODE — navigate freely",
            bg=BG_PANEL, fg="#aaaacc", font=("Consolas", 8, "italic"),
        )
        self._mode_label.pack(side=tk.LEFT, padx=12)

        # ── canvas ────────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(self, bg="#0d0d1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._on_canvas_resize())

        # mouse events (always bound — gated by mode inside handlers)
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

        self._draw_empty_hint()

        # ── controls ──────────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG_PANEL)
        ctrl.pack(fill=tk.X, pady=(2, 0))

        self.slider = tk.Scale(
            ctrl, from_=0, to=0, orient=tk.HORIZONTAL,
            command=self._on_slider,
            bg=BG_PANEL, fg=TEXT_LIGHT, troughcolor=BG_DARK,
            highlightthickness=0, sliderrelief=tk.FLAT,
        )
        self.slider.pack(fill=tk.X, padx=8)

        btns = tk.Frame(ctrl, bg=BG_PANEL)
        btns.pack(pady=4)

        for text, cmd in [
            ("⏮", self._go_first), ("◀", self._prev),
            ("▶", self._next),     ("⏭", self._go_last),
        ]:
            tk.Button(
                btns, text=text, command=cmd,
                bg=ACCENT, fg="white", relief=tk.FLAT,
                width=4, font=("Consolas", 11), cursor="hand2",
            ).pack(side=tk.LEFT, padx=3)

        self.idx_label = tk.Label(
            ctrl, text="Frame —",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 9),
        )
        self.idx_label.pack(pady=(0, 4))

    # ── public API ────────────────────────────────────────────────────────────
    def load(self, loader, indices: List[int],
             frame_path_provider: Optional[Callable[[int], str]] = None):
        self._loader  = loader
        self._indices = indices
        self._pos     = 0
        self._frame_path_provider = frame_path_provider
        self._clear_hint()
        if indices:
            self.slider.config(to=len(indices) - 1)
            self._show_current()

    def set_overlay_boxes(self, boxes: List[BoundingBox]):
        self._boxes = list(boxes)
        # If selection index is out of range, clear it.
        if (self._selected_idx is not None
                and self._selected_idx >= len(self._boxes)):
            self._selected_idx = None
        self._redraw()

    def set_selected_box(self, idx: Optional[int]):
        """Public — used by AnnotationPanel to sync selection from list."""
        if idx is not None and (idx < 0 or idx >= len(self._boxes)):
            idx = None
        self._select_box(idx)

    @property
    def current_frame_index(self) -> int:
        if not self._indices:
            return 0
        return self._indices[self._pos]

    # ── mode switching ────────────────────────────────────────────────────────
    def set_view_mode(self):
        self._mode = self.MODE_VIEW
        self.canvas.config(cursor="arrow")
        self._view_btn.config(bg=ACCENT,   fg="white")
        self._draw_btn.config(bg=BG_DARK,  fg=TEXT_LIGHT)
        self._mode_label.config(text="  VIEW MODE — navigate freely")
        self._cancel_draw()

    def set_draw_mode(self):
        self._mode = self.MODE_DRAW
        self.canvas.config(cursor="crosshair")
        self._draw_btn.config(bg="#e05c5c", fg="white")
        self._view_btn.config(bg=BG_DARK,  fg=TEXT_LIGHT)
        self._mode_label.config(
            text="  DRAW MODE — click & drag to annotate"
        )

    def is_draw_mode(self) -> bool:
        return self._mode == self.MODE_DRAW

    # ── mouse handlers ────────────────────────────────────────────────────────
    def _on_mouse_press(self, event):
        if self._mode == self.MODE_DRAW:
            self._draw_start = (event.x, event.y)
            if self._draw_rect:
                self.canvas.delete(self._draw_rect)
            self._draw_rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#ff4444", width=2, dash=(4, 4),
            )
            return

        # View mode → box editing
        # If a box is already selected and we pressed near a handle, start
        # resizing. Otherwise, see if the press hit any box → select it.
        if self._selected_idx is not None:
            handle = self._handle_at(event.x, event.y, self._selected_idx)
            if handle is not None:
                self._begin_edit(handle, event.x, event.y)
                return

        hit = self._box_at(event.x, event.y)
        if hit is not None:
            self._select_box(hit)
            self._begin_edit("move", event.x, event.y)
        else:
            # Click on empty area → deselect
            self._select_box(None)

    def _on_mouse_drag(self, event):
        if self._mode == self.MODE_DRAW and self._draw_start is not None:
            x0, y0 = self._draw_start
            self.canvas.coords(self._draw_rect, x0, y0, event.x, event.y)
            return

        # Edit drag
        if self._edit_handle is not None and self._selected_idx is not None:
            self._apply_edit_drag(event.x, event.y, commit=False)

    def _on_mouse_release(self, event):
        if self._mode == self.MODE_DRAW and self._draw_start is not None:
            x0, y0 = self._draw_start
            x1, y1 = event.x, event.y
            if self._draw_rect:
                self.canvas.delete(self._draw_rect)
                self._draw_rect = None
            self._draw_start = None
            if abs(x1 - x0) < 8 or abs(y1 - y0) < 8:
                return
            norm = self._canvas_to_norm(x0, y0, x1, y1)
            if norm and self._on_box_drawn:
                self._on_box_drawn(*norm)
            return

        # Edit release → commit final coords to caller
        if self._edit_handle is not None and self._selected_idx is not None:
            self._apply_edit_drag(event.x, event.y, commit=True)
            self._edit_handle = None
            self._edit_drag_start = None

    def _cancel_draw(self):
        if self._draw_rect:
            self.canvas.delete(self._draw_rect)
            self._draw_rect = None
        self._draw_start = None

    # ── box editing helpers ───────────────────────────────────────────────────
    HANDLE_SIZE = 6   # pixels — square half-side for resize hit-test

    def _box_pixel_rect(self, box):
        """Return (x1, y1, x2, y2) in canvas (not frame) coordinates."""
        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc     = self._frame_scale
        cx = box.x_center * self._frame_w
        cy = box.y_center * self._frame_h
        w  = box.width    * self._frame_w
        h  = box.height   * self._frame_h
        x1 = ox + (cx - w/2) * sc
        y1 = oy + (cy - h/2) * sc
        x2 = ox + (cx + w/2) * sc
        y2 = oy + (cy + h/2) * sc
        return x1, y1, x2, y2

    def _box_at(self, cx, cy) -> Optional[int]:
        """Return the index of the topmost box containing canvas point (cx,cy),
        or None. Smallest-area first so a tiny box inside a big one wins."""
        candidates = []
        for i, b in enumerate(self._boxes):
            x1, y1, x2, y2 = self._box_pixel_rect(b)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                candidates.append((i, (x2 - x1) * (y2 - y1)))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[1])
        return candidates[0][0]

    def _handle_at(self, cx, cy, box_idx) -> Optional[str]:
        """Return the handle name (corner/edge/move) if (cx,cy) is on a
        handle of self._boxes[box_idx], else None."""
        if box_idx is None or box_idx >= len(self._boxes):
            return None
        x1, y1, x2, y2 = self._box_pixel_rect(self._boxes[box_idx])
        s = self.HANDLE_SIZE
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        handles = {
            "nw": (x1, y1), "n": (mx, y1), "ne": (x2, y1),
            "w":  (x1, my),                "e":  (x2, my),
            "sw": (x1, y2), "s": (mx, y2), "se": (x2, y2),
        }
        for name, (hx, hy) in handles.items():
            if abs(cx - hx) <= s and abs(cy - hy) <= s:
                return name
        return None

    def _begin_edit(self, handle, mx, my):
        if self._selected_idx is None:
            return
        box = self._boxes[self._selected_idx]
        self._edit_handle = handle
        self._edit_drag_start = (mx, my, self._box_pixel_rect(box))

    def _apply_edit_drag(self, mx, my, commit: bool):
        if self._edit_drag_start is None or self._selected_idx is None:
            return
        sx, sy, (x1, y1, x2, y2) = self._edit_drag_start
        dx, dy = mx - sx, my - sy
        h = self._edit_handle

        if h == "move":
            x1 += dx; y1 += dy; x2 += dx; y2 += dy
        else:
            if "n" in h: y1 += dy
            if "s" in h: y2 += dy
            if "w" in h: x1 += dx
            if "e" in h: x2 += dx

        # Clamp to frame area
        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc     = self._frame_scale
        fx2    = ox + self._frame_w * sc
        fy2    = oy + self._frame_h * sc
        x1, x2 = sorted((max(ox, min(x1, fx2)), max(ox, min(x2, fx2))))
        y1, y2 = sorted((max(oy, min(y1, fy2)), max(oy, min(y2, fy2))))

        norm = self._canvas_to_norm(x1, y1, x2, y2)
        if norm is None:
            return
        nx1, ny1, nx2, ny2 = norm
        if nx2 - nx1 < 0.005 or ny2 - ny1 < 0.005:
            return  # ignore degenerate

        # Live-preview by updating local copy + redraw
        box = self._boxes[self._selected_idx]
        box.x_center = (nx1 + nx2) / 2
        box.y_center = (ny1 + ny2) / 2
        box.width    = nx2 - nx1
        box.height   = ny2 - ny1
        self._redraw()

        if commit and self._on_box_edited:
            self._on_box_edited(self._selected_idx, nx1, ny1, nx2, ny2)

    def _select_box(self, idx: Optional[int]):
        if idx == self._selected_idx:
            return
        self._selected_idx = idx
        self._redraw()
        if self._on_box_selected:
            self._on_box_selected(idx)

    # ── coord conversion ──────────────────────────────────────────────────────
    def _canvas_to_norm(self, cx0, cy0, cx1, cy1):
        """
        Convert canvas pixel rect → (x1_n, y1_n, x2_n, y2_n) normalised to
        the *original frame* dimensions. Returns None if frame not loaded.
        """
        if self._frame_scale == 0:
            return None

        ox, oy = self._frame_offset_x, self._frame_offset_y
        sc     = self._frame_scale

        # Clamp to frame area
        fw = self._frame_w * sc
        fh = self._frame_h * sc

        ix0 = max(0.0, min((cx0 - ox) / sc, self._frame_w))
        iy0 = max(0.0, min((cy0 - oy) / sc, self._frame_h))
        ix1 = max(0.0, min((cx1 - ox) / sc, self._frame_w))
        iy1 = max(0.0, min((cy1 - oy) / sc, self._frame_h))

        # Normalise to [0, 1]
        nx0 = ix0 / self._frame_w
        ny0 = iy0 / self._frame_h
        nx1 = ix1 / self._frame_w
        ny1 = iy1 / self._frame_h

        # Ensure top-left < bottom-right
        return (min(nx0, nx1), min(ny0, ny1),
                max(nx0, nx1), max(ny0, ny1))

    # ── navigation ────────────────────────────────────────────────────────────
    def _go_first(self): self._goto(0)
    def _go_last(self):  self._goto(len(self._indices) - 1)
    def _prev(self):     self._goto(self._pos - 1)
    def _next(self):     self._goto(self._pos + 1)

    def _on_slider(self, val):
        self._goto(int(val), update_slider=False)

    def _goto(self, pos: int, update_slider: bool = True):
        if not self._indices:
            return
        self._pos = max(0, min(pos, len(self._indices) - 1))
        if update_slider:
            self.slider.set(self._pos)
        self._show_current()

    def _show_current(self):
        if not self._loader or not self._indices:
            return
        idx   = self._indices[self._pos]
        frame = self._read_frame_reliable(idx)
        if frame is None:
            return
        self._current_frame = frame
        self._frame_h, self._frame_w = frame.shape[:2]
        self.idx_label.config(
            text=f"Frame {idx}  ({self._pos + 1} / {len(self._indices)})"
        )
        if self._on_change:
            self._on_change(idx, frame)
        self._redraw()

    def _on_canvas_resize(self):
        if self._current_frame is None:
            self._draw_empty_hint()
        else:
            self._redraw()

    def _read_frame_reliable(self, idx: int):
        """
        Prefer the saved PNG over re-seeking the video — cv2 seek is
        unreliable on H.264/variable-rate videos and can return wrong
        frames or None mid-navigation.
        """
        if self._frame_path_provider:
            try:
                path = self._frame_path_provider(idx)
            except Exception:
                path = None
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    return img
        return self._loader.read_frame(idx)

    # ── empty-state hint ──────────────────────────────────────────────────────
    def _draw_empty_hint(self):
        self.canvas.delete("hint")
        cw = self.canvas.winfo_width()  or 640
        ch = self.canvas.winfo_height() or 480
        cx, cy = cw // 2, ch // 2
        self.canvas.create_text(
            cx, cy - 28, text="📂",
            font=("Consolas", 36), fill="#3a3a5e", tags="hint",
        )
        self.canvas.create_text(
            cx, cy + 20,
            text="Click to open a video or image",
            font=("Consolas", 13), fill="#4a4a6e", tags="hint",
        )
        self.canvas.create_text(
            cx, cy + 44,
            text="— or use the toolbar buttons above —",
            font=("Consolas", 9), fill="#333355", tags="hint",
        )
        self.canvas.tag_bind("hint", "<Button-1>", self._on_hint_click)
        self.canvas.config(cursor="hand2")

    def _clear_hint(self):
        self.canvas.delete("hint")
        self.canvas.config(cursor="arrow")

    def _on_hint_click(self, _event):
        if self._on_open_request:
            self._on_open_request()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _redraw(self):
        if self._current_frame is None:
            return

        cw = self.canvas.winfo_width()  or 640
        ch = self.canvas.winfo_height() or 480
        fh, fw = self._current_frame.shape[:2]

        # Compute letterbox scale + offsets
        scale  = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        ox     = (cw - nw) // 2
        oy     = (ch - nh) // 2

        self._frame_scale    = scale
        self._frame_offset_x = ox
        self._frame_offset_y = oy
        self._frame_w        = fw
        self._frame_h        = fh

        frame = self._current_frame
        if self._boxes:
            from utils.image_utils import draw_boxes
            frame = draw_boxes(frame, self._boxes)

        from utils.image_utils import bgr_to_photoimage
        self._photo = bgr_to_photoimage(frame, cw, ch)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2,
                                 image=self._photo, anchor=tk.CENTER)

        # Highlight selected box + draw resize handles on top
        if (self._selected_idx is not None
                and self._selected_idx < len(self._boxes)):
            x1, y1, x2, y2 = self._box_pixel_rect(self._boxes[self._selected_idx])
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="#ffaa00", width=2, dash=(2, 2), tags="selection",
            )
            s = self.HANDLE_SIZE
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            for hx, hy in [
                (x1, y1), (mx, y1), (x2, y1),
                (x1, my),           (x2, my),
                (x1, y2), (mx, y2), (x2, y2),
            ]:
                self.canvas.create_rectangle(
                    hx - s, hy - s, hx + s, hy + s,
                    fill="#ffaa00", outline="white", width=1, tags="handle",
                )