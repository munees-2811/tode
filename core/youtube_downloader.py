"""
core/youtube_downloader.py
───────────────────────────
Downloads a YouTube video (or any yt-dlp-supported URL) to disk,
reporting progress via a callback.

Requires:  pip install yt-dlp imageio-ffmpeg
"""
import os
import shutil
import threading
from typing import Callable, Optional
from utils.config import OUTPUT_DIR
from utils.logger import get_logger

log = get_logger("core.YouTubeDownloader")

DOWNLOAD_DIR = os.path.join(OUTPUT_DIR, "youtube_downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def resolve_ffmpeg() -> Optional[str]:
    """
    Return a path to an ffmpeg executable, or None if not available.

    Resolution order:
      1. System ffmpeg on PATH (lets users override the bundled one)
      2. imageio-ffmpeg bundled binary (works cross-platform, no install)
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        log.warning(f"imageio-ffmpeg not available: {exc}")
        return None


class YouTubeDownloader:
    """
    Downloads a YouTube video using yt-dlp.

    Parameters
    ----------
    url               : str
    progress_callback : callable(percent: float, speed: str, eta: str)
    done_callback     : callable(file_path: str)
    error_callback    : callable(error_msg: str)
    quality           : str   — '720', '480', '360', 'best'
    """

    def __init__(
        self,
        url:               str,
        progress_callback: Optional[Callable] = None,
        done_callback:     Optional[Callable] = None,
        error_callback:    Optional[Callable] = None,
        quality:           str = "720",
    ):
        self.url               = url
        self.progress_callback = progress_callback
        self.done_callback     = done_callback
        self.error_callback    = error_callback
        self.quality           = quality
        self._output_path: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        log.info(f"YouTubeDownloader created — url={url}, quality={quality}")

    # ── public ────────────────────────────────────────────────────────────────
    def start(self):
        """Begin download in a daemon thread."""
        self._thread = threading.Thread(target=self._download, daemon=True)
        self._thread.start()
        log.info("Download thread started")

    def get_video_info(self) -> dict:
        """Return title/duration/channel without downloading."""
        try:
            import yt_dlp
            opts = {"quiet": True, "no_warnings": True, "skip_download": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                return {
                    "title":    info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                    "channel":  info.get("uploader", "Unknown"),
                    "formats":  [
                        f"{f.get('height', '?')}p"
                        for f in info.get("formats", [])
                        if f.get("height")
                    ],
                }
        except Exception as exc:
            log.error(f"Failed to fetch video info: {exc}")
            return {}

    # ── internal ──────────────────────────────────────────────────────────────
    def _download(self):
        try:
            import yt_dlp
        except ImportError:
            msg = "yt-dlp not installed. Run: pip install yt-dlp"
            log.error(msg)
            if self.error_callback:
                self.error_callback(msg)
            return

        # Format selection: prefer mp4 at requested height
        if self.quality == "best":
            fmt = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
        else:
            fmt = (
                f"bestvideo[height<={self.quality}][ext=mp4]"
                f"+bestaudio[ext=m4a]/"
                f"best[height<={self.quality}][ext=mp4]/"
                f"best[height<={self.quality}]"
            )

        ydl_opts = {
            "format":   fmt,
            "outtmpl":  os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"),
            "quiet":    True,
            "no_warnings": True,
            "progress_hooks": [self._progress_hook],
            "merge_output_format": "mp4",
        }

        # Resolve FFmpeg — yt-dlp needs it to merge video+audio streams.
        ffmpeg_path = resolve_ffmpeg()
        if ffmpeg_path:
            ydl_opts["ffmpeg_location"] = ffmpeg_path
            log.info(f"Using ffmpeg: {ffmpeg_path}")
        else:
            log.warning(
                "FFmpeg not found and imageio-ffmpeg unavailable — "
                "falling back to single-stream download (lower quality)."
            )
            # Force single-stream download so no merge is attempted.
            ydl_opts["format"] = (
                "best[ext=mp4]/best"
                if self.quality == "best"
                else f"best[height<={self.quality}][ext=mp4]/best[height<={self.quality}]"
            )

        log.info(f"Starting yt-dlp download — quality={self.quality}")
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=True)
                # Resolve final filename
                filename = ydl.prepare_filename(info)
                # Ensure .mp4 extension
                base, _ = os.path.splitext(filename)
                for ext in (".mp4", ".mkv", ".webm"):
                    candidate = base + ext
                    if os.path.exists(candidate):
                        self._output_path = candidate
                        break
                if not self._output_path:
                    self._output_path = filename

            log.info(f"Download complete → {self._output_path}")
            if self.done_callback:
                self.done_callback(self._output_path)

        except Exception as exc:
            log.error(f"yt-dlp error: {exc}", exc_info=True)
            if self.error_callback:
                self.error_callback(str(exc))

    def _progress_hook(self, d: dict):
        if d.get("status") == "downloading":
            pct   = d.get("_percent_str", "?").strip()
            speed = d.get("_speed_str", "?").strip()
            eta   = d.get("_eta_str", "?").strip()
            try:
                pct_f = float(pct.replace("%", ""))
            except ValueError:
                pct_f = 0.0
            log.debug(f"Download progress: {pct} @ {speed} ETA {eta}")
            if self.progress_callback:
                self.progress_callback(pct_f, speed, eta)
        elif d.get("status") == "finished":
            log.info("Download finished, post-processing…")
            if self.progress_callback:
                self.progress_callback(100.0, "—", "0s")