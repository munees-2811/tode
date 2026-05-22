"""
ui/segmentation_panel.py
──────────────────────────────────────────────────────────────────────────────
Semantic Segmentation panel — right-hand panel shown when annotation type
is 'Segmentation'.

Features added (on top of the original polygon list):
  • Colour-coded class list with swatches
  • Add / rename / delete semantic class
  • Opacity slider for filled mask overlays
  • Per-frame polygon list (class name + vertex count)
  • Delete selected polygon
  • Save / Clear frame actions
"""

import tkinter as tk
from collections.abc import Callable
from tkinter import colorchooser, messagebox, simpledialog, ttk

from models.annotation_model import PolygonAnnotation
from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT

# ── default colour palette ─────────────────────────────────────────────────
_PALETTE = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#E91E63", "#F1C40F", "#00BCD4",
]

ROW_H = 18   # approximate Listbox row height in pixels


class SegmentationPanel(tk.Frame):
    """
    Semantic segmentation right-panel.

    Callbacks (all optional):
        on_save_click()
        on_clear_click()
        on_delete_poly(index: int)
        on_poly_select(index: int | None)
        on_class_changed(class_name: str, color: str)
        on_opacity_change(value: float)   # 0.0 – 1.0
    """

    def __init__(
        self,
        master,
        on_save_click:    Callable = None,
        on_clear_click:   Callable = None,
        on_delete_poly:   Callable = None,
        on_poly_select:   Callable = None,
        on_class_changed: Callable = None,
        on_opacity_change: Callable = None,
    ) -> None:
        super().__init__(master, bg=BG_PANEL, width=280)
        self.pack_propagate(False)

        self._on_save          = on_save_click
        self._on_clear         = on_clear_click
        self._on_delete_poly   = on_delete_poly
        self._on_poly_select   = on_poly_select
        self._on_class_changed = on_class_changed
        self._on_opacity       = on_opacity_change

        # semantic class list: [{name, color}, …]
        self._classes: list[dict] = []
        self._selected_class_idx: int | None = None

        self._build()
        # seed one default class
        self._add_class(name="object", color=_PALETTE[0], notify=False)

    # ── build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 3}

        # title
        tk.Label(
            self, text="SEMANTIC SEGMENTATION",
            bg=BG_PANEL, fg=ACCENT,
            font=("Consolas", 10, "bold"),
        ).pack(pady=(10, 2))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        # how-to hint
        tk.Label(
            self,
            text=(
                "\n  HOW TO ANNOTATE:\n"
                "  1. Pick a semantic class below\n"
                "  2. Click '⬠ Polygon' mode button\n"
                "  3. Click canvas to place vertices\n"
                "  4. Double-click to close polygon\n"
                "  5. Press Esc to cancel in-progress\n"
            ),
            bg=BG_PANEL, fg="#8888aa",
            font=("Consolas", 8), justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        # ── opacity slider ─────────────────────────────────────────────────
        op_row = tk.Frame(self, bg=BG_PANEL)
        op_row.pack(fill=tk.X, **pad)
        tk.Label(op_row, text="Mask opacity:", bg=BG_PANEL, fg=TEXT_LIGHT,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._opacity_var = tk.DoubleVar(value=0.40)
        tk.Scale(
            op_row,
            variable=self._opacity_var,
            from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, length=110,
            bg=BG_PANEL, fg=TEXT_LIGHT, troughcolor=BG_DARK,
            highlightthickness=0, bd=0,
            command=lambda v: self._on_opacity and self._on_opacity(float(v)),
        ).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=2)

        # ── semantic class list ────────────────────────────────────────────
        tk.Label(
            self, text="SEMANTIC CLASSES",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(pady=(4, 2))

        cls_outer = tk.Frame(self, bg=BG_PANEL)
        cls_outer.pack(fill=tk.X, padx=8)

        # colour swatch strip (left)
        self._swatch_canvas = tk.Canvas(
            cls_outer, width=16, bg=BG_DARK,
            bd=0, highlightthickness=0, height=90,
        )
        self._swatch_canvas.pack(side=tk.LEFT, fill=tk.Y)

        cls_sb = tk.Scrollbar(cls_outer, orient=tk.VERTICAL, bg=BG_DARK)
        cls_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._cls_listbox = tk.Listbox(
            cls_outer,
            yscrollcommand=cls_sb.set,
            bg=BG_DARK, fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 9), relief=tk.FLAT, bd=0, height=5,
            activestyle="none",
        )
        self._cls_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cls_sb.config(command=self._cls_listbox.yview)
        self._cls_listbox.bind("<<ListboxSelect>>", self._on_cls_listbox_select)

        # class action buttons
        cls_btn_row = tk.Frame(self, bg=BG_PANEL)
        cls_btn_row.pack(fill=tk.X, padx=8, pady=(4, 2))

        _cb = dict(relief=tk.FLAT, padx=5, pady=2,
                   font=("Consolas", 8), cursor="hand2",
                   bg=ACCENT, fg="white",
                   activebackground="#9d8fff", activeforeground="white")

        tk.Button(cls_btn_row, text="+ Add",
                  command=self._prompt_add_class, **_cb).pack(side=tk.LEFT, padx=2)
        tk.Button(cls_btn_row, text="✎ Rename",
                  command=self._rename_class, **_cb).pack(side=tk.LEFT, padx=2)
        tk.Button(cls_btn_row, text="🎨 Color",
                  command=self._pick_color, **_cb).pack(side=tk.LEFT, padx=2)
        tk.Button(cls_btn_row, text="✕ Del",
                  command=self._delete_class,
                  bg="#7a3333", fg="white", relief=tk.FLAT,
                  padx=5, pady=2, font=("Consolas", 8), cursor="hand2",
                  activebackground="#a04040",
                  ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        # ── polygon list ───────────────────────────────────────────────────
        tk.Label(
            self, text="POLYGONS ON THIS FRAME",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(pady=(0, 2))

        poly_frame = tk.Frame(self, bg=BG_PANEL)
        poly_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        poly_sb = tk.Scrollbar(poly_frame)
        poly_sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._poly_listbox = tk.Listbox(
            poly_frame,
            yscrollcommand=poly_sb.set,
            bg=BG_DARK, fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Consolas", 8), relief=tk.FLAT, bd=0, height=7,
        )
        self._poly_listbox.pack(fill=tk.BOTH, expand=True)
        self._poly_listbox.bind("<<ListboxSelect>>", self._on_poly_listbox_select)
        poly_sb.config(command=self._poly_listbox.yview)

        stats_row = tk.Frame(self, bg=BG_PANEL)
        stats_row.pack(fill=tk.X, padx=8, pady=(2, 0))

        self._stats_var = tk.StringVar(value="0 polygons")
        tk.Label(
            stats_row, textvariable=self._stats_var,
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8),
        ).pack(side=tk.LEFT)

        tk.Button(
            stats_row, text="🗑 Delete Selected",
            command=self._delete_selected_poly,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=6, pady=2, font=("Consolas", 8), cursor="hand2",
        ).pack(side=tk.RIGHT)

        # ── bottom buttons ─────────────────────────────────────────────────
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        tk.Button(
            self, text="💾  Save Annotations",
            command=lambda: self._on_save and self._on_save(),
            bg="#2d8a4e", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(fill=tk.X, padx=8, pady=2)

        tk.Button(
            self, text="🗑  Clear Frame Polygons",
            command=lambda: self._on_clear and self._on_clear(),
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

    # ── public API ─────────────────────────────────────────────────────────

    def update_polygons(
        self,
        polygons: list[PolygonAnnotation],
        class_names: list[str],
    ) -> None:
        """Refresh polygon list. Also syncs class combo values."""
        # merge any unseen class names into our class list
        existing = {c["name"] for c in self._classes}
        for name in class_names:
            if name not in existing:
                color = _PALETTE[len(self._classes) % len(_PALETTE)]
                self._add_class(name=name, color=color, notify=False)
                existing.add(name)

        self._poly_listbox.delete(0, tk.END)
        for i, poly in enumerate(polygons):
            color = self._color_for(poly.class_name)
            conf  = f"{poly.confidence:.2f}" if poly.confidence < 1.0 else "manual"
            self._poly_listbox.insert(
                tk.END,
                f"  [{i:02d}] {poly.class_name:<14} {len(poly.points):2d}pts  {conf}",
            )
            # tint the row with the class colour
            self._poly_listbox.itemconfig(
                i, fg=color, selectforeground="white",
            )

        n = len(polygons)
        self._stats_var.set(f"{n} polygon{'s' if n != 1 else ''}")

    def get_selected_class(self) -> str:
        """Return the currently highlighted class name."""
        if self._selected_class_idx is not None and self._classes:
            idx = self._selected_class_idx
            if 0 <= idx < len(self._classes):
                return self._classes[idx]["name"]
        return self._classes[0]["name"] if self._classes else "object"

    def get_selected_color(self) -> str:
        """Return the hex colour for the active class."""
        if self._selected_class_idx is not None and self._classes:
            idx = self._selected_class_idx
            if 0 <= idx < len(self._classes):
                return self._classes[idx]["color"]
        return _PALETTE[0]

    def get_opacity(self) -> float:
        return self._opacity_var.get()

    def set_class_names(self, names: list[str]) -> None:
        """Bulk-load class names (e.g. from YOLO model)."""
        existing = {c["name"] for c in self._classes}
        for name in names:
            if name not in existing:
                color = _PALETTE[len(self._classes) % len(_PALETTE)]
                self._add_class(name=name, color=color, notify=False)
                existing.add(name)

    # ── class management ───────────────────────────────────────────────────

    def _prompt_add_class(self) -> None:
        name = simpledialog.askstring("New class", "Class name:", parent=self)
        if name and name.strip():
            self._add_class(name=name.strip())

    def _add_class(
        self,
        name: str = "object",
        color: str | None = None,
        notify: bool = True,
    ) -> None:
        if color is None:
            color = _PALETTE[len(self._classes) % len(_PALETTE)]
        self._classes.append({"name": name, "color": color})
        self._cls_listbox.insert(tk.END, f"  {name}")
        self._refresh_swatches()
        # auto-select if first
        if len(self._classes) == 1:
            self._cls_listbox.selection_set(0)
            self._selected_class_idx = 0
        if notify:
            self._fire_class_changed()

    def _rename_class(self) -> None:
        idx = self._selected_class_idx
        if idx is None:
            return
        new = simpledialog.askstring(
            "Rename class", "New name:",
            initialvalue=self._classes[idx]["name"],
            parent=self,
        )
        if new and new.strip():
            self._classes[idx]["name"] = new.strip()
            self._cls_listbox.delete(idx)
            self._cls_listbox.insert(idx, f"  {new.strip()}")
            self._cls_listbox.selection_set(idx)
            self._fire_class_changed()

    def _pick_color(self) -> None:
        idx = self._selected_class_idx
        if idx is None:
            return
        result = colorchooser.askcolor(
            color=self._classes[idx]["color"],
            title="Pick class colour",
            parent=self,
        )
        if result and result[1]:
            self._classes[idx]["color"] = result[1]
            self._refresh_swatches()
            self._fire_class_changed()

    def _delete_class(self) -> None:
        idx = self._selected_class_idx
        if idx is None or not self._classes:
            return
        if len(self._classes) == 1:
            messagebox.showinfo("Cannot delete", "At least one class is required.")
            return
        name = self._classes[idx]["name"]
        if not messagebox.askyesno(
            "Delete class",
            f"Delete class '{name}'?\n"
            "Existing polygons tagged with this class keep their label.",
            parent=self,
        ):
            return
        self._classes.pop(idx)
        self._cls_listbox.delete(idx)
        self._refresh_swatches()
        new_idx = min(idx, len(self._classes) - 1)
        self._cls_listbox.selection_set(new_idx)
        self._selected_class_idx = new_idx
        self._fire_class_changed()

    def _on_cls_listbox_select(self, _event=None) -> None:
        sel = self._cls_listbox.curselection()
        if sel:
            self._selected_class_idx = sel[0]
            self._fire_class_changed()

    def _fire_class_changed(self) -> None:
        if self._on_class_changed and self._selected_class_idx is not None:
            idx = self._selected_class_idx
            if 0 <= idx < len(self._classes):
                c = self._classes[idx]
                self._on_class_changed(c["name"], c["color"])

    def _refresh_swatches(self) -> None:
        self._swatch_canvas.delete("all")
        for i, cls in enumerate(self._classes):
            y = i * ROW_H + 4
            self._swatch_canvas.create_rectangle(
                2, y, 13, y + 11,
                fill=cls["color"], outline="",
            )

    def _color_for(self, class_name: str) -> str:
        for c in self._classes:
            if c["name"] == class_name:
                return c["color"]
        return TEXT_LIGHT

    # ── polygon list events ────────────────────────────────────────────────

    def _delete_selected_poly(self) -> None:
        sel = self._poly_listbox.curselection()
        if sel and self._on_delete_poly:
            self._on_delete_poly(sel[0])

    def _on_poly_listbox_select(self, _event) -> None:
        sel  = self._poly_listbox.curselection()
        idx  = sel[0] if sel else None
        if self._on_poly_select:
            self._on_poly_select(idx)
