"""Frame extraction that accounts for every expected frame.

The extractor is usable standalone (:meth:`FrameExtractor.run`) or per frame
(:meth:`FrameExtractor.process_frame`) when the pipeline wants a single decode
pass shared with detection and tracking. Both paths produce identical
:class:`FrameRecord` values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.ingestion.video_reader import FramePacket, VideoMetadata, VideoReader
from src.utils.context import RunContext
from src.utils.image import blur_score, dhash, hamming_distance, mean_absolute_difference, resize_to_width

SCORE_WIDTH = 320
SAVE_WIDTH = 0


@dataclass
class FrameRecord:
    """Index entry for a single frame attempt."""

    frame_id: int
    timestamp_ms: int
    width: int
    height: int
    read_ok: bool
    blur_score: float | None = None
    blurred: bool = False
    duplicate_of: int | None = None
    hash_distance: int | None = None
    mean_abs_diff: float | None = None
    saved_path: str | None = None
    error: str | None = None
    decode_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        payload = asdict(self)
        if payload["blur_score"] is not None:
            payload["blur_score"] = round(float(payload["blur_score"]), 4)
        if payload["mean_abs_diff"] is not None:
            payload["mean_abs_diff"] = round(float(payload["mean_abs_diff"]), 4)
        if payload["decode_ms"] is not None:
            payload["decode_ms"] = round(float(payload["decode_ms"]), 4)
        return payload


@dataclass
class FrameIndex:
    """Result of extraction over a whole video."""

    metadata: VideoMetadata
    records: list[FrameRecord] = field(default_factory=list)
    expected_frames: int = 0

    @property
    def extracted(self) -> int:
        """Frames that decoded successfully."""
        return sum(1 for record in self.records if record.read_ok)

    @property
    def failed(self) -> int:
        """Frames that failed to decode."""
        return sum(1 for record in self.records if not record.read_ok)

    @property
    def blurred(self) -> int:
        """Frames below the blur threshold."""
        return sum(1 for record in self.records if record.blurred)

    @property
    def duplicates(self) -> int:
        """Frames flagged as duplicates of a previous frame."""
        return sum(1 for record in self.records if record.duplicate_of is not None)

    @property
    def records_emitted(self) -> int:
        """Number of records produced, which must equal expected frames."""
        return len(self.records)

    @property
    def complete(self) -> bool:
        """True when exactly one record exists per expected frame."""
        return self.expected_frames <= 0 or self.records_emitted == self.expected_frames

    def summary(self) -> dict[str, Any]:
        """Aggregate extraction metrics."""
        blur_values = [record.blur_score for record in self.records if record.blur_score is not None]
        return {
            "expected_frames": self.expected_frames,
            "records_emitted": self.records_emitted,
            "extracted_frames": self.extracted,
            "failed_frames": self.failed,
            "blurred_frames": self.blurred,
            "duplicate_frames": self.duplicates,
            "mean_blur_score": round(float(np.mean(blur_values)), 4) if blur_values else None,
            "complete": self.complete,
        }


class FrameExtractor:
    """Computes per-frame quality signals and persists the frame index."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.blur_threshold = float(ctx.get("frames.blur_threshold", 60.0))
        self.duplicate_mad_threshold = float(ctx.get("frames.duplicate_mad_threshold", 1.5))
        self.duplicate_hash_distance = int(ctx.get("frames.duplicate_hash_distance", 2))
        self.save_raw = ctx.flag("output.save_raw_frames", False)
        self.annotated_every_n = max(1, int(ctx.get("output.annotated_every_n_frames", 1)))
        self._previous_gray: np.ndarray | None = None
        self._previous_hash: int | None = None
        self._previous_frame_id: int | None = None
        self.records: list[FrameRecord] = []
        self.metrics = ctx.metrics

    def reset(self) -> None:
        """Clear per-run state so the extractor can be reused."""
        self._previous_gray = None
        self._previous_hash = None
        self._previous_frame_id = None
        self.records = []

    def process_frame(self, packet: FramePacket) -> FrameRecord:
        """Score one frame and return its index record."""
        if not packet.read_ok or packet.image is None:
            record = FrameRecord(
                frame_id=packet.frame_id,
                timestamp_ms=packet.timestamp_ms,
                width=0,
                height=0,
                read_ok=False,
                error=packet.error or "frame_decode_failure",
            )
            self.records.append(record)
            self.metrics.increment("extraction", "failed_frames")
            return record

        image = packet.image
        height, width = int(image.shape[0]), int(image.shape[1])
        score_image = resize_to_width(image, SCORE_WIDTH)
        gray = cv2.cvtColor(score_image, cv2.COLOR_BGR2GRAY)
        score = float(blur_score(gray))
        frame_hash = dhash(gray)

        mean_diff: float | None = None
        hash_distance: int | None = None
        duplicate_of: int | None = None
        if self._previous_gray is not None:
            mean_diff = float(mean_absolute_difference(self._previous_gray, gray))
            hash_distance = int(hamming_distance(self._previous_hash or 0, frame_hash))
            if (
                mean_diff <= self.duplicate_mad_threshold
                and hash_distance <= self.duplicate_hash_distance
            ):
                duplicate_of = self._previous_frame_id

        record = FrameRecord(
            frame_id=packet.frame_id,
            timestamp_ms=packet.timestamp_ms,
            width=width,
            height=height,
            read_ok=True,
            blur_score=score,
            blurred=score < self.blur_threshold,
            duplicate_of=duplicate_of,
            hash_distance=hash_distance,
            mean_abs_diff=mean_diff,
        )

        if self.save_raw:
            record.saved_path = self._save_frame(image, packet.frame_id)

        self._previous_gray = gray
        self._previous_hash = frame_hash
        self._previous_frame_id = packet.frame_id

        self.records.append(record)
        self.metrics.increment("extraction", "extracted_frames")
        self.metrics.observe("extraction", "blur_score", score)
        if record.blurred:
            self.metrics.increment("extraction", "blurred_frames")
        if record.duplicate_of is not None:
            self.metrics.increment("extraction", "duplicate_frames")
        return record

    def _save_frame(self, image: np.ndarray, frame_id: int) -> str:
        """Persist a raw frame under ``frames/`` when configured."""
        target = self.ctx.store.path("frames", f"frame_{frame_id:06d}.jpg")
        if SAVE_WIDTH and image.shape[1] > SAVE_WIDTH:
            scale = SAVE_WIDTH / float(image.shape[1])
            image = cv2.resize(
                image,
                (SAVE_WIDTH, int(round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return str(target)

    def finalize(self, metadata: VideoMetadata) -> FrameIndex:
        """Persist the frame index and metrics, returning the aggregate index."""
        index = FrameIndex(
            metadata=metadata,
            records=list(self.records),
            expected_frames=int(metadata.frame_count),
        )
        writer = self.ctx.store.writer("frames", "frame_index.jsonl")
        for record in self.records:
            writer.write(record.to_dict())
        writer.close()

        summary = index.summary()
        self.metrics.record("extraction", summary)
        self.metrics.set("extraction", "fps", metadata.fps)
        self.metrics.set("extraction", "codec", metadata.codec)
        self.metrics.set("extraction", "resolution", metadata.resolution)
        self.metrics.set("extraction", "expected_frames", index.expected_frames)
        self.ctx.store.write_json("frames", "frame_index_summary.json", summary)
        return index

    def run(self, video_path: str | Path, metadata: VideoMetadata | None = None) -> FrameIndex:
        """Standalone extraction over a full video."""
        reader = VideoReader(video_path, metadata=metadata)
        self.reset()
        with reader:
            info = reader.metadata
            for packet in reader.iter_frames():
                self.process_frame(packet)
        return self.finalize(info)
