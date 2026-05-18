"""
ui/classification_panel.py
────────────────────────────
Right-hand panel for image-level classification labelling.
"""
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from models.annotation_model import ImageClassification
from utils.config import ACCENT, BG_DARK, BG_PANEL, TEXT_LIGHT


class ClassificationPanel(tk.Frame):
    """
    Panel displayed when annotation type is 'Classification'.

    Shows one label selector per frame. Fires *on_classify(class_name)*
    when the user confirms a label, and *on_clear_click* to remove it.
    """

    def __init__(
        self,
        master,
        on_classify:  Callable,        # callable(class_name: str)
        on_save_click:  Callable,
        on_clear_click: Callable,
    ) -> None:
        super().__init__(master, bg=BG_PANEL, width=280)
        self.pack_propagate(False)

        self._on_classify  = on_classify
        self._on_save      = on_save_click
        self._on_clear     = on_clear_click

        self.selected_class_var = tk.StringVar(value="")
        self._current_cls: ImageClassification | None = None

        self._build()

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        tk.Label(
            self, text="CLASSIFICATION PANEL",
            bg=BG_PANEL, fg=ACCENT,
            font=("Consolas", 10, "bold"),
        ).pack(pady=(10, 4))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8)

        tk.Label(
            self,
            text=(
                "\n  CLASSIFY THIS FRAME:\n"
                "  Select or type a label then\n"
                "  click 'Apply Label'.\n"
                "  Only one label per frame.\n"
            ),
            bg=BG_PANEL, fg="#8888aa",
            font=("Consolas", 8), justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)

        tk.Label(
            self, text="Label",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 9),
        ).pack(pady=(4, 2), padx=10, anchor=tk.W)

        self._combo = ttk.Combobox(
            self, textvariable=self.selected_class_var,
            font=("Consolas", 9),
        )
        self._combo.pack(fill=tk.X, padx=10, ipady=3)

        tk.Button(
            self, text="✅  Apply Label",
            command=self._apply,
            bg=ACCENT, fg="white", relief=tk.FLAT,
            padx=8, pady=5, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(fill=tk.X, padx=10, pady=(8, 2))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=8)

        # Current label display
        tk.Label(
            self, text="CURRENT LABEL",
            bg=BG_PANEL, fg=TEXT_LIGHT, font=("Consolas", 8, "bold"),
        ).pack(padx=10, anchor=tk.W)

        self._label_var = tk.StringVar(value="(none)")
        tk.Label(
            self, textvariable=self._label_var,
            bg=BG_DARK, fg=ACCENT,
            font=("Consolas", 11, "bold"),
            padx=8, pady=6,
        ).pack(fill=tk.X, padx=10, pady=4)

        tk.Button(
            self, text="🗑  Remove Label",
            command=self._on_clear,
            bg="#7a3333", fg="white", relief=tk.FLAT,
            padx=8, pady=4, font=("Consolas", 9), cursor="hand2",
        ).pack(fill=tk.X, padx=10, pady=2)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=6)

        tk.Button(
            self, text="💾  Save Annotations",
            command=self._on_save,
            bg="#2d8a4e", fg="white", relief=tk.FLAT,
            padx=8, pady=6, font=("Consolas", 9, "bold"), cursor="hand2",
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

    # ── event handlers ────────────────────────────────────────────────────────

    def _apply(self) -> None:
        name = self.selected_class_var.get().strip()
        if name and self._on_classify:
            self._on_classify(name)

    # ── public API ────────────────────────────────────────────────────────────

    def set_class_names(self, names: list[str]) -> None:
        self._combo["values"] = sorted(set(names))

    def update_classification(self, cls: ImageClassification | None) -> None:
        self._current_cls = cls
        if cls:
            self._label_var.set(f"  {cls.class_name}")
            self.selected_class_var.set(cls.class_name)
        else:
            self._label_var.set("(none)")
