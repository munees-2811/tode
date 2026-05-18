"""
ui/source_dialog.py
────────────────────
Modal dialog — click a source card to browse and load a local file or folder.
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT
from utils.logger import get_logger

log = get_logger("ui.SourceDialog")


class SourceDialog(tk.Toplevel):
    """
    result dict after closing:
        {"type": "video"|"image"|"image_folder", "path": "..."}
    or None if cancelled.
    """

    def __init__(self, master):
        super().__init__(master)
        self.title("Open Annotation Source")
        self.resizable(False, False)
        self.configure(bg=BG_DARK)
        self.geometry("520x360")
        self.grab_set()
        self.focus_set()

        self.result: dict | None = None
        self._selected_path = tk.StringVar(value="")
        self._selected_type = tk.StringVar(value="")

        self._build()
        self.wait_window()

    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK)
        hdr.pack(fill=tk.X, padx=24, pady=(20, 0))

        tk.Label(hdr, text="Choose Annotation Source",
                 bg=BG_DARK, fg=ACCENT,
                 font=("Consolas", 14, "bold")).pack(anchor=tk.W)
        tk.Label(hdr, text="Click a card to browse, then press Open",
                 bg=BG_DARK, fg="#777799",
                 font=("Consolas", 9)).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=20, pady=12)

        # Source cards
        cards_frame = tk.Frame(self, bg=BG_DARK)
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=16)

        cards = [
            ("video",        "🎬", "Video File",
             "MP4 · AVI · MOV · MKV · WEBM · FLV · WMV", "#4a3a8a"),
            ("image",        "🖼", "Single Image",
             "JPG · PNG · BMP · TIFF · WEBP", "#3a6a4a"),
            ("image_folder", "📂", "Image Folder",
             "Load every image inside a folder (recursive)", "#3a4a7a"),
        ]
        for src_type, icon, title, sub, hover_bg in cards:
            self._make_card(cards_frame, src_type, icon, title, sub, hover_bg)

        # Path preview bar
        preview = tk.Frame(self, bg=BG_PANEL, height=28)
        preview.pack(fill=tk.X, padx=16, pady=(8, 0))
        preview.pack_propagate(False)
        tk.Label(preview, text=" Selected: ",
                 bg=BG_PANEL, fg="#666688",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Label(preview, textvariable=self._selected_path,
                 bg=BG_PANEL, fg=ACCENT,
                 font=("Consolas", 8),
                 anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=16, pady=8)

        # Bottom buttons
        btns = tk.Frame(self, bg=BG_DARK)
        btns.pack(pady=(0, 14), padx=16, fill=tk.X)

        tk.Button(btns, text="✖  Cancel",
                  command=self.destroy,
                  bg="#444455", fg="white", relief=tk.FLAT,
                  padx=18, pady=8,
                  font=("Consolas", 10, "bold"),
                  cursor="hand2").pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(btns, text="Open  →",
                  command=self._confirm,
                  bg=ACCENT, fg="white", relief=tk.FLAT,
                  padx=18, pady=8,
                  font=("Consolas", 10, "bold"),
                  cursor="hand2").pack(side=tk.LEFT)

        self._status_lbl = tk.Label(
            btns, text="",
            bg=BG_DARK, fg="#aaaacc",
            font=("Consolas", 8))
        self._status_lbl.pack(side=tk.RIGHT, padx=8)

    # ── Source cards ──────────────────────────────────────────────────────────
    def _make_card(self, parent, src_type, icon, title, subtitle, hover_bg):
        card = tk.Frame(parent, bg=BG_PANEL, cursor="hand2",
                        relief=tk.FLAT, bd=0)
        card.pack(fill=tk.X, padx=0, pady=5, ipady=4)

        strip = tk.Frame(card, bg=hover_bg, width=5)
        strip.pack(side=tk.LEFT, fill=tk.Y)

        body = tk.Frame(card, bg=BG_PANEL)
        body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=8)

        rb = tk.Radiobutton(body,
                            variable=self._selected_type,
                            value=src_type,
                            bg=BG_PANEL,
                            activebackground=BG_PANEL,
                            selectcolor=hover_bg)
        rb.pack(side=tk.LEFT)

        text_frame = tk.Frame(body, bg=BG_PANEL)
        text_frame.pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(text_frame,
                 text=f"{icon}  {title}",
                 bg=BG_PANEL, fg=TEXT_LIGHT,
                 font=("Consolas", 10, "bold")).pack(anchor=tk.W)
        tk.Label(text_frame,
                 text=subtitle,
                 bg=BG_PANEL, fg="#777799",
                 font=("Consolas", 8)).pack(anchor=tk.W)

        tk.Button(body,
                  text="Browse…",
                  command=lambda t=src_type: self._browse(t),
                  bg=hover_bg, fg="white", relief=tk.FLAT,
                  padx=10, pady=4,
                  font=("Consolas", 8, "bold"),
                  cursor="hand2").pack(side=tk.RIGHT, padx=4)

        for widget in (card, body, strip, text_frame):
            widget.bind("<Button-1>",
                        lambda _e, t=src_type: self._browse(t))

        def _enter(_e):
            card.config(bg=hover_bg)
            body.config(bg=hover_bg)
            text_frame.config(bg=hover_bg)
            strip.config(bg=ACCENT)

        def _leave(_e):
            card.config(bg=BG_PANEL)
            body.config(bg=BG_PANEL)
            text_frame.config(bg=BG_PANEL)
            strip.config(bg=hover_bg)

        for widget in (card, body, strip, text_frame):
            widget.bind("<Enter>", _enter)
            widget.bind("<Leave>", _leave)

    def _browse(self, src_type: str):
        self._selected_type.set(src_type)

        if src_type == "image_folder":
            path = filedialog.askdirectory(
                title="Select Image Folder", parent=self)
        elif src_type == "image":
            path = filedialog.askopenfilename(
                title="Select Image", parent=self,
                filetypes=[
                    ("Image files",
                     "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                    ("All files", "*.*"),
                ],
            )
        else:  # video
            path = filedialog.askopenfilename(
                title="Select Video File", parent=self,
                filetypes=[
                    ("Video files",
                     "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv"),
                    ("All files", "*.*"),
                ],
            )

        if path:
            self._selected_path.set(path)
            self._status_lbl.config(
                text=f"✔  {os.path.basename(path)}", fg="#55cc77")
            log.info(f"User browsed — type={src_type}, path={path}")
        else:
            if not self._selected_path.get():
                self._selected_type.set("")

    # ── Confirm ───────────────────────────────────────────────────────────────
    def _confirm(self):
        path = self._selected_path.get().strip()
        src  = self._selected_type.get().strip()

        if not path:
            messagebox.showwarning(
                "Nothing selected",
                "Please click a card and browse for a file or folder.",
                parent=self,
            )
            return

        if not os.path.exists(path):
            messagebox.showerror(
                "Path not found",
                f"The selected path does not exist:\n{path}",
                parent=self,
            )
            return

        if src == "image_folder" and not os.path.isdir(path):
            messagebox.showerror("Wrong type",
                                 "Selected path is not a folder.",
                                 parent=self)
            return
        if src in ("video", "image") and not os.path.isfile(path):
            messagebox.showerror("Wrong type",
                                 "Selected path is not a file.",
                                 parent=self)
            return

        self.result = {"type": src, "path": path}
        log.info(f"Source confirmed — type={src}, path={path}")
        self.destroy()
