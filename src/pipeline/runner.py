"""Reusable lifecycle for CLI and Streamlit video processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from src.ingestion.video_reader import FramePacket, VideoMetadata, VideoReadError, VideoReader, probe_metadata
from src.metrics.collector import MetricsCollector
from src.output.annotator import TimelinePlotter
from src.output.schemas import (
    BBox,
    BBoxNormalized,
    Decision,
    EgoVehicle,
    Environment,
    ENVIRONMENT_COUNT_FIELDS,
    EnvironmentObject,
    Event,
    FrameAnalysis,
    FrameInfo,
    LaneBoundary,
    LaneEstimate,
    Point,
    Road,
    Severity,
    Validation,
    UNKNOWN,
)
from src.output.writer import RunStore, iter_jsonl
from src.pipeline.stage_bundle import build_bundle
from src.temporal.forward_pass import FrameOutcome, FramePass, ForwardPass, make_event
from src.utils.config import Config
from src.utils.context import ProgressReporter, RunContext
from src.utils.logging_utils import setup_logger
from src.utils.seeding import set_global_seed
from src.utils.run_id import utc_now_iso
from src.validation.validator import validate_frame

ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class PipelineResult:
    """Stable result returned by :class:`PipelineRunner.run`."""

    run_id: str
    run_dir: Path
    metadata: dict[str, Any]
    frame_count: int
    successful_frames: int
    failed_frames: int
    status: str
    manifest: Path
    final_jsonl: Path
    final_json: Path
    zip_path: Path | None = None


class PipelineRunner:
    """Own one complete run, including failure finalisation."""

    def __init__(
        self,
        output_root: str | Path = "outputs",
        config_path: str | Path | None = None,
        config: Config | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.config = config or Config.load(config_path)
        self.progress_callback = progress_callback

    def run(self, video_path: str | Path) -> PipelineResult:
        """Process every expected frame and persist a complete run."""
        run_config_hash = self.config.hash()
        store = RunStore.create(self.output_root, config_hash=run_config_hash)
        logger = setup_logger(
            f"deepsense.{store.run_id}",
            store.path("logs", "run.log"),
            self.config.get("logging.level", "INFO"),
        )
        metrics = MetricsCollector(store=store, logger=logger)
        context = RunContext(
            config=self.config,
            store=store,
            metrics=metrics,
            logger=logger,
            progress=ProgressReporter(self.progress_callback, logger),
        )
        started = utc_now_iso()
        try:
            metadata = probe_metadata(video_path, int(self.config.get("ingestion.ffprobe_timeout_s", 20)))
            store.copy_input(video_path)
            store.write_json("metrics", "video_metadata.json", metadata.to_dict())
            seed = int(self.config.get("pipeline.seed", 1337))
            set_global_seed(seed)
            context.notes["input_copy"] = str(store.path("input", "source" + Path(video_path).suffix))
            context.notes["metadata"] = metadata.to_dict()
            context.notes["started_iso"] = started
            result = self._process(metadata, context)
            return result
        except Exception as exc:
            logger.exception("pipeline failed")
            metrics.note("runner", f"{type(exc).__name__}: {exc}")
            metrics.set("runner", "status", "failed")
            metrics.persist_overall({"status": "failed"})
            store.close()
            manifest = store.write_manifest(
                self._manifest_payload(context, status="failed", metadata=context.notes.get("metadata"), error=str(exc))
            )
            return PipelineResult(
                run_id=store.run_id,
                run_dir=store.run_dir,
                metadata=context.notes.get("metadata") or {},
                frame_count=0,
                successful_frames=0,
                failed_frames=0,
                status="failed",
                manifest=manifest,
                final_jsonl=store.path("final", "frame_analysis.jsonl"),
                final_json=store.path("final", "frame_analysis.json"),
            )

    def _process(self, metadata: VideoMetadata, context: RunContext) -> PipelineResult:
        """Run forward analysis, reverse validation and final materialisation."""
        expected = max(0, int(metadata.frame_count))
        bundle = build_bundle(context, metadata.fps)
        context.notes["components"] = bundle.describe()
        forward = ForwardPass(context, bundle)
        context.progress.report("forward", 0, expected, "starting")
        forward_records = context.store.writer("temporal", "forward_records.jsonl")
        frame_count = 0
        successful = 0
        failed = 0
        try:
            for packet in forward.iter_frames(str(metadata.path), metadata):
                try:
                    outcome = forward.process(packet)
                except Exception as exc:
                    outcome = self._failed_outcome(packet, exc)
                    context.metrics.note("forward", f"frame {packet.frame_id}: {type(exc).__name__}: {exc}")
                frame_count += 1
                successful += int(outcome.read_ok)
                failed += int(not outcome.read_ok)
                forward_records.write({"record": self._base_record(outcome, context.run_id), "summary": self._summary(outcome)})
                context.progress.report("forward", frame_count, expected or frame_count, f"frame {packet.frame_id}")
        finally:
            forward_records.close()
        forward_stats = forward.finalize()
        context.metrics.record("forward", forward_stats)
        frame_index = bundle.extractor.finalize(metadata)
        context.progress.report("forward", frame_count, expected or frame_count, "complete")

        backward_by_frame = self._backward_pass(metadata, context, frame_count)
        self._write_validation_and_final(context, backward_by_frame)
        context.metrics.set("runner", "expected_frames", expected)
        context.metrics.set("runner", "records_emitted", frame_count)
        context.metrics.set("runner", "failed_frames", failed)
        context.metrics.set("runner", "successful_frames", successful)
        context.metrics.set("runner", "frame_count_invariant", frame_count == expected if expected else True)
        context.metrics.set("runner", "detector_engine", (context.notes.get("components") or {}).get("detector", {}).get("engine", "unknown"))
        context.metrics.persist_overall(
            {
                "status": "completed",
                "expected_frames": expected,
                "records_emitted": frame_count,
                "failed_frames": failed,
                "successful_frames": successful,
                "frame_count_invariant": frame_count == expected if expected else True,
            }
        )
        context.store.close()
        manifest = context.store.write_manifest(self._manifest_payload(context, "completed", metadata.to_dict()))
        zip_path = None
        if context.flag("output.zip_after_run", False):
            zip_path = context.store.package_zip()
        return PipelineResult(
            run_id=context.run_id,
            run_dir=context.run_dir,
            metadata=metadata.to_dict(),
            frame_count=frame_count,
            successful_frames=successful,
            failed_frames=failed,
            status="completed",
            manifest=manifest,
            final_jsonl=context.store.path("final", "frame_analysis.jsonl"),
            final_json=context.store.path("final", "frame_analysis.json"),
            zip_path=zip_path,
        )

    def _backward_pass(self, metadata: VideoMetadata, context: RunContext, frame_count: int) -> dict[int, dict[str, Any]]:
        """Run an independent reverse pass, retaining only compact summaries."""
        context.progress.report("backward", 0, frame_count, "starting")
        bundle = build_bundle(context, metadata.fps)
        backward = FramePass(context, bundle, direction="backward", persist_artifacts=False, frames_output=None)
        summaries: dict[int, dict[str, Any]] = {}
        writer = context.store.writer("temporal", "backward_frames.jsonl")
        try:
            with VideoReader(metadata.path, metadata=metadata) as reader:
                for done, frame_id in enumerate(range(frame_count - 1, -1, -1), 1):
                    packet = reader.read_frame(frame_id)
                    try:
                        outcome = backward.process(packet)
                    except Exception as exc:
                        outcome = self._failed_outcome(packet, exc)
                    summary = self._summary(outcome)
                    summaries[frame_id] = summary
                    writer.write(summary)
                    context.progress.report("backward", done, frame_count, f"frame {frame_id}")
        finally:
            writer.close()
        backward.finalize()
        context.metrics.set("validation", "backward_frames", len(summaries))
        context.progress.report("backward", frame_count, frame_count, "complete")
        return summaries

    def _write_validation_and_final(self, context: RunContext, backward: dict[int, dict[str, Any]]) -> None:
        """Join forward records and reverse decisions into final JSONL and JSON."""
        validation_writer = context.store.writer("validation", "validation.jsonl")
        final_writer = context.store.writer("final", "frame_analysis.jsonl")
        validated = 0
        disagreed = 0
        compact: list[dict[str, Any]] = []
        validation_events: list[dict[str, Any]] = []
        for item in iter_jsonl(context.store.path("temporal", "forward_records.jsonl")):
            base = item["record"]
            summary = item["summary"]
            decision = validate_frame(summary, backward.get(int(summary["frame_id"])))
            validated += 1
            disagreed += int(decision.consistent is False)
            validation_payload = decision.to_dict()
            validation_writer.write(validation_payload)
            if len(compact) < 5000:
                compact.append({"frame_id": int(summary["frame_id"]), "lane_count": summary.get("primary", {}).get("lane_count"), "ego_lane": summary.get("primary", {}).get("ego_lane"), "object_count": len(summary.get("objects", []))})
            if decision.anomalies and len(validation_events) < 5000:
                validation_events.append(validation_payload)
            payload = self._final_payload(base, decision, context.run_id)
            final_writer.write(payload)
        validation_writer.close()
        final_writer.close()
        context.store.write_json_array("final", "frame_analysis.json", iter_jsonl(context.store.path("final", "frame_analysis.jsonl")))
        context.metrics.record("validation", {"frames_validated": validated, "frames_disagreed": disagreed})
        context.metrics.set("validation", "status", "completed")
        TimelinePlotter(context).plot(compact, validation_events)

    def _base_record(self, outcome: FrameOutcome, run_id: str) -> dict[str, Any]:
        """Convert stage output to a schema-valid record before validation."""
        frame_record = outcome.frame_record
        frame = FrameInfo(
            frame_id=outcome.frame_id,
            timestamp_ms=outcome.timestamp_ms,
            width=outcome.width,
            height=outcome.height,
            read_ok=outcome.read_ok,
            blur_score=None if frame_record is None else frame_record.blur_score,
            duplicate_of=None if frame_record is None else frame_record.duplicate_of,
            error=None if frame_record is None else frame_record.error,
        )
        road = self._road(outcome)
        environment = self._environment(outcome)
        events = list(outcome.events)
        for error in outcome.stage_errors:
            events.append(make_event("stage_error", outcome.frame_id, outcome.timestamp_ms, error, Severity.ERROR))
        validation_status = "failed" if not outcome.read_ok else "anomalous" if outcome.anomalies else "valid"
        validation = Validation(
            status=validation_status,
            anomalies=list(outcome.anomalies),
            anomaly_details=events,
            forward_backward_consistent=UNKNOWN,
            unresolved=bool(outcome.stage_errors),
        )
        return FrameAnalysis(
            run_id=run_id,
            frame=frame,
            ego_vehicle=EgoVehicle(lane=UNKNOWN if road.ego_lane == UNKNOWN else road.ego_lane),
            road=road,
            objects=outcome.objects,
            environment=environment,
            events=events,
            validation=validation,
            decision=Decision(primary_result=outcome.primary_summary(), validation_result={"stage_errors": list(outcome.stage_errors)}),
        ).model_dump(mode="json")

    @staticmethod
    def _road(outcome: FrameOutcome) -> Road:
        observation = outcome.lane_observation
        source = outcome.road
        if source is None:
            return Road()
        width, height = max(1, outcome.width), max(1, outcome.height)
        boundaries: list[LaneBoundary] = []
        if observation is not None:
            for index, line in enumerate(observation.boundaries, 1):
                boundaries.append(LaneBoundary(lane_id=index, side=line.side, source=line.source, points=[Point(x=int(x), y=int(y)) for x, y in line.points], points_normalized=[[round(x / width, 6), round(y / height, 6)] for x, y in line.points], polynomial=line.polynomial, confidence=line.confidence))
        lanes: list[LaneEstimate] = []
        if observation is not None:
            for lane in observation.lanes:
                left, right = int(lane.get("left_x", 0)), int(lane.get("right_x", 0))
                polygon = [Point(x=left, y=height - 1), Point(x=right, y=height - 1), Point(x=right, y=int(0.55 * height)), Point(x=left, y=int(0.55 * height))]
                lanes.append(LaneEstimate(lane_id=int(lane.get("lane_id", len(lanes) + 1)), polygon=polygon, polygon_normalized=[[round(point.x / width, 6), round(point.y / height, 6)] for point in polygon], center_offset_x=int((left + right) / 2 - width / 2), confidence=observation.confidence))
        return Road(lane_count=source.lane_count, ego_lane=source.ego_lane, road_type=source.road_type, confidence=source.confidence, curvature=source.curvature, vanishing_point=None if source.vanishing_point is None else Point(x=int(source.vanishing_point[0]), y=int(source.vanishing_point[1])), lanes=lanes, boundaries=boundaries, features=dict(source.features))

    @staticmethod
    def _environment(outcome: FrameOutcome) -> Environment:
        observation = outcome.environment
        if observation is None:
            return Environment(unknown_counts=list(ENVIRONMENT_COUNT_FIELDS))
        width, height = max(1, outcome.width), max(1, outcome.height)
        objects = []
        for item in observation.items:
            bbox = BBox(**item.bbox)
            objects.append(EnvironmentObject(type=item.type, bbox=bbox, bbox_normalized=BBoxNormalized(x=bbox.x / width, y=bbox.y / height, width=bbox.width / width, height=bbox.height / height), confidence=item.confidence, source=item.source, track_id=item.track_id, lane=item.lane))
        counts = dict(observation.counts)
        return Environment(**counts, vegetation_ratio=observation.vegetation_ratio, horizon_y=observation.horizon_y, confidence=observation.confidence, environment_objects=objects, features=dict(observation.features))

    @staticmethod
    def _summary(outcome: FrameOutcome) -> dict[str, Any]:
        return {"frame_id": outcome.frame_id, "timestamp_ms": outcome.timestamp_ms, "read_ok": outcome.read_ok, "primary": outcome.primary_summary(), "objects": outcome.object_summaries(), "anomalies": list(outcome.anomalies), "stage_errors": list(outcome.stage_errors)}

    @staticmethod
    def _final_payload(base: dict[str, Any], decision: Any, run_id: str) -> dict[str, Any]:
        """Apply validation with immutable dictionary copies."""
        payload = dict(base)
        payload["run_id"] = run_id
        existing_validation = dict(payload.get("validation", {}))
        anomalies = list(dict.fromkeys(list(existing_validation.get("anomalies", [])) + list(decision.anomalies)))
        details = list(existing_validation.get("anomaly_details", []))
        for detail in decision.details:
            details.append(Event(type="forward_backward_disagreement", frame_id=decision.frame_id, timestamp_ms=int(payload["frame"]["timestamp_ms"]), severity=Severity.WARNING, description=str(detail)).model_dump(mode="json"))
        existing_validation.update({"status": "failed" if existing_validation.get("status") == "failed" else "anomalous" if anomalies else "valid", "anomalies": anomalies, "anomaly_details": details, "forward_backward_consistent": decision.consistent, "unresolved": bool(existing_validation.get("unresolved", False) or decision.consistent is False)})
        payload["validation"] = existing_validation
        existing_decision = dict(payload.get("decision", {}))
        validation_result = dict(existing_decision.get("validation_result", {}))
        validation_result.update(decision.to_dict())
        existing_decision.update({"validation_result": validation_result, "final_result": "primary", "decision_reason": "forward_backward_consistent" if decision.consistent is True else "forward_backward_disagreement_preserved" if decision.consistent is False else "validation_insufficient"})
        payload["decision"] = existing_decision
        return FrameAnalysis.model_validate(payload).model_dump(mode="json")

    @staticmethod
    def _failed_outcome(packet: FramePacket, exc: Exception) -> FrameOutcome:
        outcome = FrameOutcome(frame_id=packet.frame_id, timestamp_ms=packet.timestamp_ms, read_ok=False, width=packet.width, height=packet.height)
        outcome.stage_errors.append(f"frame: {type(exc).__name__}: {exc}")
        outcome.add_anomaly("frame_decode_failure", f"frame {packet.frame_id} processing failed", Severity.ERROR)
        return outcome

    @staticmethod
    def _manifest_payload(context: RunContext, status: str, metadata: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status, "created_iso": context.notes.get("started_iso"), "pipeline": context.config.as_dict(), "pipeline_version": context.get("pipeline.version", "unknown"), "components": context.notes.get("components", {}), "metadata": metadata, "input_copy": context.notes.get("input_copy"), "config_hash": context.config.hash()}
        if error:
            payload["error"] = error
        return payload


def run_pipeline(video_path: str | Path, output_root: str | Path = "outputs", config_path: str | Path | None = None, progress_callback: ProgressCallback | None = None) -> PipelineResult:
    """Convenience API shared by scripts and UI."""
    return PipelineRunner(output_root, config_path, progress_callback=progress_callback).run(video_path)
