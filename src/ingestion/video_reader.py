"""Video container probing and reliable sequential frame iteration."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from src.utils.hashing import sha256_file
from src.utils.run_id import timestamp_ms_for_frame

MAX_CONSECUTIVE_READ_FAILURES = 10


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or has no readable frames."""


@dataclass
class FramePacket:
    """One decoded frame, or one recorded decode failure."""

    frame_id: int
    timestamp_ms: int
    image: np.ndarray | None
    read_ok: bool
    error: str | None = None

    @property
    def height(self) -> int:
        """Frame height in pixels, or 0 when the frame failed."""
        if self.image is None:
            return 0
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Frame width in pixels, or 0 when the frame failed."""
        if self.image is None:
            return 0
        return int(self.image.shape[1])


@dataclass
class VideoMetadata:
    """Everything known about the input video before analysis."""

    path: str
    filename: str
    sha256: str
    size_bytes: int
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float | None
    codec: str
    bitrate: int | None = None
    container: str | None = None
    frame_count_source: str = "opencv"
    width_source: str = "opencv"
    height_source: str = "opencv"
    fps_source: str = "opencv"
    probe_error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> str:
        """``WIDTHxHEIGHT`` string."""
        return f"{self.width}x{self.height}"

    @property
    def duration_from_frames_s(self) -> float | None:
        """Duration implied by frame count and FPS."""
        if self.fps and self.fps > 0 and self.frame_count > 0:
            return round(self.frame_count / self.fps, 3)
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable metadata record."""
        return {
            "path": self.path,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "fps": round(float(self.fps), 6),
            "frame_count": int(self.frame_count),
            "width": int(self.width),
            "height": int(self.height),
            "resolution": self.resolution,
            "duration_s": self.duration_s,
            "duration_from_frames_s": self.duration_from_frames_s,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "container": self.container,
            "provenance": {
                "frame_count_source": self.frame_count_source,
                "width_source": self.width_source,
                "height_source": self.height_source,
                "fps_source": self.fps_source,
            },
            "probe_error": self.probe_error,
            "extras": self.extras,
        }


def _ffprobe_stream(path: Path, timeout_s: int = 20) -> tuple[dict[str, Any] | None, str | None]:
    """Probe the first video stream with ffprobe."""
    if shutil.which("ffprobe") is None:
        return None, "ffprobe not available"
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,bit_rate,duration",
        "-show_entries",
        "format=format_name,duration,bit_rate,size",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return None, f"ffprobe failed: {exc}"
    if completed.returncode != 0:
        return None, f"ffprobe exit {completed.returncode}"
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return None, f"ffprobe json error: {exc}"
    return payload, None


def _parse_fraction(value: str | None) -> float | None:
    """Parse ffprobe rational frame rates such as ``30000/1001``."""
    if not value:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_metadata(path: str | Path, ffprobe_timeout_s: int = 20) -> VideoMetadata:
    """Probe container metadata, falling back to OpenCV where ffprobe is silent."""
    video_path = Path(path)
    if not video_path.exists():
        raise VideoReadError(f"video not found: {video_path}")
    if not video_path.is_file():
        raise VideoReadError(f"not a file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"OpenCV could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if frame_count <= 0:
        frame_count = _count_frames_by_iteration(capture)
        if frame_count > 0:
            frame_count_source = "opencv_iteration"
        else:
            frame_count_source = "unknown"
    else:
        frame_count_source = "opencv"

    payload, probe_error = _ffprobe_stream(video_path, ffprobe_timeout_s)
    codec = "unknown"
    duration_s: float | None = None
    bitrate: int | None = None
    container: str | None = None
    extras: dict[str, Any] = {}

    if payload:
        streams = payload.get("streams") or []
        stream = streams[0] if streams else {}
        fmt = payload.get("format") or {}
        codec = str(stream.get("codec_name") or "unknown")

        reported_width = int(stream.get("width") or 0)
        reported_height = int(stream.get("height") or 0)
        if reported_width > 0:
            width = reported_width
        if reported_height > 0:
            height = reported_height

        probe_fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(
            stream.get("r_frame_rate")
        )
        if probe_fps and probe_fps > 0:
            fps = probe_fps
            fps_source = "ffprobe"
        else:
            fps_source = "opencv"

        probe_frames = stream.get("nb_frames")
        if probe_frames and str(probe_frames).isdigit() and int(probe_frames) > 0:
            frame_count = int(probe_frames)
            frame_count_source = "ffprobe"

        duration_value = stream.get("duration") or fmt.get("duration")
        if duration_value is not None:
            try:
                duration_s = round(float(duration_value), 3)
            except (TypeError, ValueError):
                duration_s = None

        bitrate_value = stream.get("bit_rate") or fmt.get("bit_rate")
        if bitrate_value is not None:
            try:
                bitrate = int(float(bitrate_value))
            except (TypeError, ValueError):
                bitrate = None

        container = str(fmt.get("format_name") or "unknown") or "unknown"
        extras["ffprobe_stream"] = {
            key: stream.get(key)
            for key in ("codec_name", "profile", "pix_fmt", "color_space", "avg_frame_rate")
            if stream.get(key) is not None
        }
    else:
        fps_source = "opencv"

    capture.release()

    if duration_s is None and fps > 0 and frame_count > 0:
        duration_s = round(frame_count / fps, 3)

    return VideoMetadata(
        path=str(video_path.resolve()),
        filename=video_path.name,
        sha256=sha256_file(video_path),
        size_bytes=int(video_path.stat().st_size),
        fps=float(fps),
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        duration_s=duration_s,
        codec=codec,
        bitrate=bitrate,
        container=container,
        frame_count_source=frame_count_source,
        width_source="ffprobe" if payload and width else "opencv",
        height_source="ffprobe" if payload and height else "opencv",
        fps_source=fps_source,
        probe_error=probe_error,
        extras=extras,
    )


def _count_frames_by_iteration(capture: cv2.VideoCapture) -> int:
    """Count frames by sequential reads, then rewind. Used when the container lies."""
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


class VideoReader:
    """Context-managed sequential frame source that never silently drops frames."""

    def __init__(self, path: str | Path, metadata: VideoMetadata | None = None) -> None:
        self.path = Path(path)
        self.metadata = metadata if metadata is not None else probe_metadata(self.path)
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> "VideoReader":
        """Open the capture and seek to the start."""
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise VideoReadError(f"OpenCV could not open video: {self.path}")
        self._capture = capture
        return self

    def close(self) -> None:
        """Release the capture."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def iter_frames(self) -> Iterator[FramePacket]:
        """Yield one :class:`FramePacket` per expected frame index.

        A failed read yields a packet with ``read_ok=False`` instead of being
        skipped. When the capture stops advancing, the remaining expected frames
        are emitted as failures so downstream record counts stay exact.
        """
        if self._capture is None:
            self.open()
        assert self._capture is not None

        fps = self.metadata.fps
        expected = int(self.metadata.frame_count)
        consecutive_failures = 0
        frame_id = 0
        exhausted = False

        while True:
            if expected > 0 and frame_id >= expected:
                break
            if exhausted and expected <= 0:
                break

            ok, image = self._capture.read()
            timestamp = timestamp_ms_for_frame(frame_id, fps)

            if ok and image is not None:
                consecutive_failures = 0
                yield FramePacket(
                    frame_id=frame_id,
                    timestamp_ms=timestamp,
                    image=image,
                    read_ok=True,
                )
                frame_id += 1
                continue

            consecutive_failures += 1
            yield FramePacket(
                frame_id=frame_id,
                timestamp_ms=timestamp,
                image=None,
                read_ok=False,
                error="frame_decode_failure",
            )
            frame_id += 1

            if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                exhausted = True
                if expected <= 0:
                    break
                while frame_id < expected:
                    yield FramePacket(
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms_for_frame(frame_id, fps),
                        image=None,
                        read_ok=False,
                        error="frame_decode_failure",
                    )
                    frame_id += 1
                break

    def read_frame(self, frame_id: int) -> FramePacket:
        """Random-access single frame read by index."""
        if self._capture is None:
            self.open()
        assert self._capture is not None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, image = self._capture.read()
        timestamp = timestamp_ms_for_frame(frame_id, self.metadata.fps)
        if ok and image is not None:
            return FramePacket(frame_id=frame_id, timestamp_ms=timestamp, image=image, read_ok=True)
        return FramePacket(
            frame_id=frame_id,
            timestamp_ms=timestamp,
            image=None,
            read_ok=False,
            error="frame_decode_failure",
        )
