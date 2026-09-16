"""Stage 1: video ingestion, metadata probing and frame iteration."""

from __future__ import annotations

from src.ingestion.video_reader import (
    FramePacket,
    VideoMetadata,
    VideoReader,
    VideoReadError,
    probe_metadata,
)

__all__ = [
    "FramePacket",
    "VideoMetadata",
    "VideoReader",
    "VideoReadError",
    "probe_metadata",
]
