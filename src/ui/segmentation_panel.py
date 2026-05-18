"""
ui/segmentation_panel.py
──────────────────────────
Right-hand panel for polygon / segmentation annotation management.
"""
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from models.annotation_model import PolygonAnnotation
from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT


class SegmentationPanel(tk.Frame):
    """
    Panel displayed when annotation type is 'Segmentation'.

    Provides:
    - Active class selector (shared list from detector or custom)
    - Polygon list with delete
    - Save / Clear actions
    """

    def __init__(
        self,
        master,
        on_save_click:   Callable,
        on_clear_click:  Callable,
        on_delete_poly:  Callable = None,  # callable(poly_index)
        on_poly_select:  Callable = None,  # callable(index_or_None)
    ) -> None:
        super().__init__(master, bg=BG_PANEL, width=280)
        self.pack_propagate(False)

        self._on_save        = on_save_click
        self._on_clear       = on_clear_click
        self._on_delete_poly = on_delete_poly
        self._on_poly_select = on_poly_select

        self.selected_class_var = tk.StringVar(value="object")
        self.custom_class_var   = tk.StringVar(value="")

        self._build()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        tk.Label(
            self, text="SEGMENTATION PANEL",
            bg=BG_PANEL, fg=ACCENT,
            font=("Consolas", 10, "bold"),
        ).pack(pady=(10, 4))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        # Instructions
        tk.Label(
            self,
            text=(
                "\n  HOW TO ANNOTATE:\n"
                "  1. Click '⬠ Polygon' mode button\n"
                "  2. Click canvas to add vertices\n"
                "  3. Double-click to close polygon\n"
                "  4. Press Esc to cancel in-progress\n"
            ),
            bg=BG_PANEL, fg="#8888aa",
            font=("Consolas", 8), justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # Class selector
        tk.Label(
            self, text="Class for New Polygon",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 9),
        ).pack(pady=(4, 2), padx=10, anchor=tk.W)

        self.class_combo = ttk.Combobox(
            self, textvariable=self.selected_class_var,
            font=("Consolas", 9), state="readonly",
        )
        self.class_combo.pack(fill=tk.X, padx=10, ipady=3)

        tk.Label(
            self, text="  ─── or type custom class ───",
            bg=BG_PANEL, fg="#888899", font=("Consolas", 8),
        ).pack(pady=(6, 2))

        tk.Entry(
            self, textvariable=self.custom_class_var,
            bg=BG_DARK, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT,
            relief=tk.FLAT, font=("Consolas", 9),
        ).pack(fill=tk.X, padx=10, ipady=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=(8, 4))

        # Polygon list
        tk.Label(
            self, text="POLYGONS ON THIS FRAME",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(pady=(0, 2))

        list_frame = tk.Frame(self, bg=BG_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            bg=BG_DARK, fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 8), relief=tk.FLAT, bd=0, height=8,
        )
        self._listbox.pack(fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        scrollbar.config(command=self._listbox.yview)

        self._stats_var = tk.StringVar(value="0 polygons")
        stats_row = tk.Frame(self, bg=BG_PANEL)
        stats_row.pack(fill=tk.X, padx=8, pady=(2, 0))

        tk.Label(
            stats_row, textvariable=self._stats_var,
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8),
        ).pack(side=tk.LEFT)

        tk.Button(
            stats_row, text="🗑 Delete Selected",
            command=self._delete_selected,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=6, pady=2, font=("Consolas", 8), cursor="hand2",
        ).pack(side=tk.RIGHT)

        # Bottom buttons
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        tk.Button(
            self, text="💾  Save Annotations",
            command=self._on_save,
            bg="#2d8a4e", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(fill=tk.X, padx=8, pady=2)

        tk.Button(
            self, text="🗑  Clear Frame Polygons",
            command=self._on_clear,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

    # ── event handlers ────────────────────────────────────────────────────────

    def _delete_selected(self) -> None:
        sel = self._listbox.curselection()
        if sel and self._on_delete_poly:
            self._on_delete_poly(sel[0])

    def _on_listbox_select(self, _event) -> None:
        sel = self._listbox.curselection()
        idx = sel[0] if sel else None
        if self._on_poly_select:
            self._on_poly_select(idx)

    # ── public API ────────────────────────────────────────────────────────────

    def update_polygons(self, polygons: list[PolygonAnnotation], class_names: list[str]) -> None:
        if class_names:
            self.class_combo["values"] = sorted(set(class_names))
            if self.selected_class_var.get() not in class_names:
                self.selected_class_var.set(class_names[0])

        self._listbox.delete(0, tk.END)
        for i, poly in enumerate(polygons):
            conf = f"{poly.confidence:.2f}" if poly.confidence < 1.0 else "  — "
            self._listbox.insert(
                tk.END,
                f"  [{i:02d}] {poly.class_name:<14} {len(poly.points):2d}pts  {conf}",
            )
        n = len(polygons)
        self._stats_var.set(f"{n} polygon{'s' if n != 1 else ''}")

    def get_selected_class(self) -> str:
        custom = self.custom_class_var.get().strip()
        return custom if custom else self.selected_class_var.get()

    def set_class_names(self, names: list[str]) -> None:
        self.class_combo["values"] = sorted(set(names))
        if names and self.selected_class_var.get() not in names:
            self.selected_class_var.set(names[0])
