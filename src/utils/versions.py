"""Model, library and environment version capture for provenance."""

from __future__ import annotations

import platform
import sys
from typing import Any


def _safe_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def library_versions() -> dict[str, str]:
    """Versions of the libraries that influence output."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _safe_version("numpy"),
        "opencv": _safe_version("cv2"),
        "torch": _safe_version("torch"),
        "ultralytics": _safe_version("ultralytics"),
        "pydantic": _safe_version("pydantic"),
        "streamlit": _safe_version("streamlit"),
    }


def compute_device() -> str:
    """Report the inference device that will be used."""
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def ffmpeg_version(ffprobe_path: str = "ffprobe") -> str:
    """ffprobe version string, or ``unavailable``."""
    import shutil
    import subprocess

    if shutil.which(ffprobe_path) is None:
        return "unavailable"
    try:
        result = subprocess.run(
            [ffprobe_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return "unavailable"
    first_line = (result.stdout or "").strip().splitlines()
    return first_line[0] if first_line else "unavailable"


def environment_summary() -> dict[str, Any]:
    """Combined provenance block for the run manifest."""
    return {
        "libraries": library_versions(),
        "device": compute_device(),
        "ffprobe": ffmpeg_version(),
    }
