"""
ui/annotation_type_selector.py
────────────────────────────────
Horizontal tab bar that switches between annotation type panels.
Inspired by the Label Studio task-type switcher.
"""
import tkinter as tk

from utils.config import ACCENT, BG_DARK, BG_PANEL

# (icon, label, key)
_TYPES = [
    ("🔲", "Bounding Box",  "bbox"),
    ("⬠",  "Segmentation",  "seg"),
    ("🏷",  "Classification","cls"),
]


class AnnotationTypeSelector(tk.Frame):
    """
    Horizontal switcher shown above the annotation panels.

    Calls *on_type_change(type_key)* when the user clicks a different type.
    Type keys: ``"bbox"``, ``"seg"``, ``"cls"``.
    """

    def __init__(self, master: tk.Misc, on_type_change=None, **kwargs) -> None:
        super().__init__(master, bg=BG_PANEL, height=36, **kwargs)
        self.pack_propagate(False)

        self._callback  = on_type_change
        self._current   = "bbox"
        self._btns: dict[str, tk.Button] = {}

        for icon, label, key in _TYPES:
            btn = tk.Button(
                self,
                text=f"{icon}  {label}",
                command=lambda k=key: self._select(k),
                bg=ACCENT if key == "bbox" else BG_DARK,
                fg="white",
                relief=tk.FLAT,
                padx=14,
                font=("Consolas", 9, "bold"),
                cursor="hand2",
                bd=0,
            )
            btn.pack(side=tk.LEFT, padx=2, pady=4)
            self._btns[key] = btn

    def _select(self, key: str) -> None:
        if key == self._current:
            return
        self._btns[self._current].config(bg=BG_DARK)
        self._current = key
        self._btns[key].config(bg=ACCENT)
        if self._callback:
            self._callback(key)

    def get_type(self) -> str:
        return self._current
