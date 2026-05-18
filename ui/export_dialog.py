"""
ui/export_dialog.py
────────────────────
Modal dialog — pick format (YOLO / COCO) and output folder, then export.
Non-annotated frames are skipped automatically.
"""
import os
import tkinter as tk
from tkinter import filedialog, ttk

from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT


class ExportDialog(tk.Toplevel):
    """
    result dict after closing:
        {"format": "yolo"|"coco", "output_dir": "..."}
    or None if cancelled.
    """

    def __init__(self, master, default_dir: str = ""):
        super().__init__(master)
        self.title("Export Annotations")
        self.resizable(False, False)
        self.configure(bg=BG_DARK)
        self.geometry("520x360")
        self.grab_set()
        self.focus_set()

        self.result: dict | None = None

        self.format_var = tk.StringVar(value="yolo")
        self.dir_var    = tk.StringVar(value=default_dir)

        self._build()
        self.wait_window()

    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        # Header
        tk.Label(
            self, text="Export Annotations",
            bg=BG_DARK, fg=ACCENT,
            font=("Consolas", 14, "bold"),
        ).pack(pady=(18, 0), padx=24, anchor=tk.W)

        tk.Label(
            self, text="Only annotated frames are exported. Empty frames are skipped.",
            bg=BG_DARK, fg="#888899", font=("Consolas", 9),
        ).pack(padx=24, anchor=tk.W, pady=(2, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=12)

        # Format section
        tk.Label(
            self, text="Format",
            bg=BG_DARK, fg=TEXT_LIGHT, font=("Consolas", 10, "bold"),
        ).pack(padx=24, anchor=tk.W)

        fmt_row = tk.Frame(self, bg=BG_DARK)
        fmt_row.pack(padx=24, pady=(4, 6), fill=tk.X)

        for value, label, sub in [
            ("yolo", "YOLO",
             "images/ + labels/ + classes.txt + data.yaml"),
            ("coco", "COCO",
             "images/ + annotations.json (single file, image dims included)"),
        ]:
            row = tk.Frame(fmt_row, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=2)

            tk.Radiobutton(
                row, text="", variable=self.format_var, value=value,
                bg=BG_PANEL, activebackground=BG_PANEL, selectcolor=ACCENT,
            ).pack(side=tk.LEFT, padx=(8, 2))

            txt = tk.Frame(row, bg=BG_PANEL)
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=4)
            tk.Label(
                txt, text=label, bg=BG_PANEL, fg=TEXT_LIGHT,
                font=("Consolas", 10, "bold"),
            ).pack(anchor=tk.W)
            tk.Label(
                txt, text=sub, bg=BG_PANEL, fg="#888899",
                font=("Consolas", 8),
            ).pack(anchor=tk.W)

            # click anywhere on row selects format
            for w in (row, txt):
                w.bind("<Button-1>",
                       lambda _e, v=value: self.format_var.set(v))

        # Output folder
        tk.Label(
            self, text="Output Folder",
            bg=BG_DARK, fg=TEXT_LIGHT, font=("Consolas", 10, "bold"),
        ).pack(padx=24, anchor=tk.W, pady=(12, 4))

        dir_row = tk.Frame(self, bg=BG_DARK)
        dir_row.pack(fill=tk.X, padx=24)

        tk.Entry(
            dir_row, textvariable=self.dir_var,
            bg=BG_PANEL, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
            relief=tk.FLAT, font=("Consolas", 9),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 6))

        tk.Button(
            dir_row, text="Browse…", command=self._browse,
            bg=BG_PANEL, fg=TEXT_LIGHT, relief=tk.FLAT,
            padx=12, font=("Consolas", 9), cursor="hand2",
        ).pack(side=tk.LEFT, ipady=2)

        # Bottom buttons
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=20, pady=(16, 10))

        btns = tk.Frame(self, bg=BG_DARK)
        btns.pack(padx=24, pady=(0, 16), fill=tk.X)

        tk.Button(
            btns, text="✖  Cancel", command=self.destroy,
            bg="#444455", fg="white", relief=tk.FLAT,
            padx=16, pady=8, font=("Consolas", 10, "bold"), cursor="hand2",
        ).pack(side=tk.LEFT)

        tk.Button(
            btns, text="📤  Export", command=self._confirm,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=18, pady=8, font=("Consolas", 10, "bold"), cursor="hand2",
        ).pack(side=tk.RIGHT)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askdirectory(
            title="Choose Export Folder", parent=self,
            initialdir=self.dir_var.get() or os.getcwd(),
        )
        if path:
            self.dir_var.set(path)

    def _confirm(self):
        from tkinter import messagebox
        out = self.dir_var.get().strip()
        if not out:
            messagebox.showwarning(
                "No folder",
                "Please choose an output folder.", parent=self,
            )
            return
        self.result = {
            "format":     self.format_var.get(),
            "output_dir": out,
        }
        self.destroy()
