"""
tode — Entry Point
Run: python main.py
"""
# ── OpenCV / FFmpeg thread safety ────────────────────────────────────────────
# Must be set BEFORE the first `import cv2` anywhere in the process,
# otherwise FFmpeg's async decoder may assert (libavcodec/pthread_frame.c:173)
# when VideoCapture is touched from multiple threads.
import os
import sys

# src/ layout: put the package root on the path so `from core...` works.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")

import cv2

cv2.setNumThreads(0)

import tkinter as tk

from ui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.title("tode")
    root.geometry("1280x800")
    root.minsize(1024, 700)

    app = MainWindow(root)
    app.pack(fill=tk.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
