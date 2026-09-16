# DeepSense Video Scene Analyzer — Design

Date: 2026-09-16
Status: Approved for implementation (V1 MVP)
Source requirement: DeepSense Video Scene Analyzer Implementation Plan (user-provided)

## 1. Goal

A Python 3.11 + Streamlit application that accepts a road/dashcam video and emits
**exactly one structured JSON record per decoded frame**. Per-frame scene
understanding is produced by deterministic Computer Vision / ML modules. An LLM
or vision-language model is **never** the primary per-frame analyzer; at most it
may later validate ambiguous frames behind an interface.

Hard requirement: when the pipeline lacks confidence in a value, the emitted
value is the literal string `unknown` (or `null` where the field is numeric).
Nothing is silently guessed.

## 2. Non-goals (V1)

- Cloud/queue/GPU-farm scaling (V3). Interfaces must remain compatible.
- Custom-trained models, segmentation, monocular depth, BEV, calibration (V2).
- Semantic lane-instance segmentation. V1 uses classical geometric lane
  detection plus heuristics.
- Detecting `tree`/`building`/`street_light` from COCO detectors. V1 derives
  these from deterministic color/edge heuristics, each carrying a confidence and
  the `source` that produced it, so a future model can replace the heuristic
  without touching the schema.

## 3. Architecture

```
Video
  -> Ingestion (metadata + hash)
  -> Frame Extraction (every frame, no silent drops)
  -> Detection  ---------------+
  -> Tracking                  |  forward pass
  -> Lane / Road Analysis      |
  -> Motion + Direction        |
  -> Environment               |
  -> Temporal State  <---------+
  -> Backward Pass (validation_result)
  -> Consistency Checks
  -> Validation (final_result + decision_reason)
  -> Per-Frame JSON (one record per frame)
  -> Metrics + Artifacts (annotated frames, timeline)
  -> Streamlit UI
```

Each stage is an independent module with a narrow interface, a Pydantic/dataclass
DTO at the boundary, and its own metrics record. Stages never import each other's
internals; they are wired by `src/pipeline/processor.py` and selected by
`config/config.yaml`.

## 4. Module contracts

| Module | Responsibility | Key interface |
| --- | --- | --- |
| `src/ingestion/video_reader.py` | Open container, probe metadata, hash, iterate frames | `VideoReader.open()`, `VideoMetadata`, `iter_frames() -> Iterator[FramePacket]` |
| `src/frames/frame_extractor.py` | Every-frame extraction, blur/duplicate scoring, optional raw persistence | `FrameExtractor.run() -> FrameIndex` |
| `src/detection/object_detector.py` | Detections for one frame | `ObjectDetector.detect(frame) -> list[Detection]`, `build_detector(cfg)` |
| `src/tracking/object_tracker.py` | Identity across frames, track records, track anomalies | `ObjectTracker.update(dets, frame) -> list[TrackObservation]`, `finalize() -> list[Track]` |
| `src/road/lane_detector.py` | Per-frame lane geometry, lane count, ego lane | `LaneDetector.detect(frame) -> LaneObservation` |
| `src/road/road_analyzer.py` | Road type, curve, intersection/merge/split, lane-state change frames | `RoadAnalyzer.analyze(obs, frame) -> RoadObservation`, `flushes` change events |
| `src/motion/trajectory.py` | Trajectory buffer, smoothing, displacement, velocity | `Trajectory`, `TrajectoryStore` |
| `src/motion/movement.py` | Ego-motion estimation + moving/stationary/unknown | `EgoMotionEstimator.estimate(prev, cur)`, `MovementClassifier.classify()` |
| `src/motion/direction.py` | Image-space and relative direction (same/opposite) | `DirectionClassifier.classify()` |
| `src/environment/environment_analyzer.py` | Vegetation, buildings, street lights, sidewalks, barriers, sky/horizon | `EnvironmentAnalyzer.analyze(frame) -> EnvironmentObservation` |
| `src/temporal/state_manager.py` | Rolling history, change-preserving smoothing | `TemporalStateManager` |
| `src/temporal/forward_pass.py` | Frame 1..N orchestration, `primary_result` | `ForwardPass` |
| `src/temporal/backward_pass.py` | Frame N..1 orchestration, `validation_result` | `BackwardPass` |
| `src/temporal/consistency.py` | Forward vs backward comparison + anomaly detection | `ConsistencyChecker` |
| `src/validation/validator.py` | `final_result`, `decision_reason`, status, optional AI hook | `FrameValidator` |
| `src/output/schemas.py` | The frozen output contract | Pydantic v2 models |
| `src/output/writer.py` | Run directory, streaming JSONL, manifest, ZIP | `RunStore` |
| `src/output/annotator.py` | Annotated frames + timeline plot | `Annotator`, `TimelinePlotter` |
| `src/metrics/collector.py` | Per-stage + overall metrics, persisted after each stage | `MetricsCollector` |
| `src/pipeline/processor.py` | Wires everything, error-tolerant, progress callbacks | `PipelineProcessor.run()` |

## 5. Output contract (frozen, `schema_version = "1.0"`)

One JSON object per frame, streamed to `final/frame_analysis.jsonl` and also
written as a JSON array to `final/frame_analysis.json`.

Top-level keys: `schema_version`, `run_id`, `frame`, `ego_vehicle`, `road`,
`objects`, `environment`, `events`, `validation`, `decision`.

- `frame`: `frame_id`, `timestamp_ms`, `width`, `height`, plus `read_ok`,
  `blur_score`, `duplicate_of`.
- `ego_vehicle`: `heading`, `lane`.
- `road`: `lane_count`, `ego_lane`, `road_type`, `confidence`, plus `lanes`
  (pixel + normalized coordinates), `lane_change_frame`, `features`.
- `objects[]`: `track_id`, `type`, `bbox`, `bbox_normalized`, `center`, `lane`,
  `movement {state, direction, image_direction, longitudinal, lateral,
  speed_px_s}`, `confidence {detection, tracking, movement}`.
- `environment`: required counts `pedestrians`, `trees`, `buildings`,
  `traffic_lights`, `traffic_signs`, `street_lights`; extended counts
  `vehicles`, `barriers`, `sidewalks`; plus `environment_objects[]` with
  individual detections (`type`, `bbox`, `confidence`, `source`, `track_id`).
  Counts are derivable from the individual records; both are emitted because the
  requirement fixes the counts keys.
- `events[]`: `{type, frame_id, timestamp_ms, severity, description,
  confidence}` — anomalies and notable transitions.
- `validation`: `status` (`valid|anomalous|failed`), `anomalies[]` (type
  strings), `forward_backward_consistent` (bool|`unknown`).
- `decision`: `primary_result`, `validation_result`, `final_result`,
  `decision_reason` — the section-17 preservation requirement. Compact
  summaries only; full per-frame forward/backward records live in
  `temporal/forward_frames.jsonl` and `temporal/backward_frames.jsonl`.

A frame that fails to decode still produces a record with
`validation.status = "failed"`, `anomalies = ["frame_decode_failure"]`, and an
event carrying the error text. **The count of JSON records always equals the
count of expected frames.**

## 6. Anomaly taxonomy

`lane_count_jump`, `track_identity_switch`, `object_teleportation`,
`object_disappearance`, `reappearance`, `impossible_motion`,
`low_detection_confidence`, `low_tracking_confidence`, `blurred_frame`,
`duplicate_frame`, `frame_decode_failure`, `forward_backward_disagreement`,
`unstable_lane_estimate`.

## 7. Run storage

```
outputs/RUN_<UTCtimestamp>_<uuid8>/
  input/       copied/probed input metadata (+ optional input copy)
  frames/      frame_index.jsonl, optional raw frames
  detection/   detections.jsonl
  tracking/    tracks.jsonl, tracks_per_frame.jsonl
  motion/      motion.jsonl
  temporal/    forward_frames.jsonl, backward_frames.jsonl, lane_events.jsonl
  validation/  validation.jsonl
  final/       frame_analysis.jsonl, frame_analysis.json
  metrics/     <stage>.json, overall.json
  annotated/   annotated frames + timeline.png
  logs/        run.log
  run_manifest.json
```

`save_raw_frames: false` by default. `run_manifest.json` records input hash,
start/end timestamps, resolved configuration, config hash, model versions,
library versions, random seed, pipeline version and the output inventory.

## 8. Determinism and reliability

- Fixed default seed (`random`, `numpy`, `torch`) recorded in the manifest.
- Tracker assignment is deterministic (sorted greedy IoU matching with velocity
  prediction), so no run-to-run drift from unordered dicts.
- Detector inference uses `torch.inference_mode`; CPU by default, CUDA if
  available.
- Primary/validation/final results and `decision_reason` are all preserved;
  disagreements are never overwritten.
- Metrics are persisted after every stage so a crash still leaves diagnosable
  state.

## 9. Error handling

- Per-frame `try/except` in the pass loops: any stage failure degrades that
  frame's fields to `unknown`, records the exception type in the frame's events,
  and sets `validation.status` to `anomalous` or `failed`.
- Detector unavailable / weights undownloadable: the registry falls back to the
  classical motion detector, and the manifest records which detector ran.
- ffprobe missing: metadata falls back to OpenCV values and `codec = unknown`.

## 10. Testing strategy

Unit: timestamp math, bbox normalization, trajectory/velocity, movement
classification, direction classification, lane transition detection and
change-frame exactness, temporal smoothing that preserves confirmed changes,
JSON schema round-trip, run id/manifest.

Integration (`tests/test_integration_pipeline.py`): a deterministically rendered
synthetic road clip is processed end to end and asserted on: one JSON record per
frame, presence of tracked objects, metrics for every stage, directory layout,
and manifest completeness. An optional real-clip smoke run is supported via
`scripts/download_sample_video.py` when the network allows.

Accuracy evaluation against ground truth is out of scope for V1 beyond the
harness described in the plan (section 18); V1 ships the metric surfaces needed
to compute detection/tracking/lane/movement/direction scores.

## 11. Phasing

V1 (this work) delivers the full MVP chain and the Streamlit explorer.
V2 adds backward-pass-informed lane/depth/BEV models, segmentation, and selective
AI validation behind `FrameValidator.ai_validator`.
V3 keeps the same stage interfaces and swaps `PipelineProcessor` for a queue
producer plus GPU workers writing to object storage and a results DB.

## 12. Additions to the requested structure (justified)

- `src/environment/environment_analyzer.py` — section 10 requires an environment
  stage that the requested tree did not enumerate.
- `src/output/annotator.py` — annotated frames are required but no module owned
  them.
- `src/utils/` — logging, hashing, run-id, geometry, image helpers, seeding used
  by every stage.
- `scripts/` — CLI entry points and the synthetic-clip generator used by tests.
- `src/timeline/` — folded into `src/output/annotator.py` (`TimelinePlotter`) to
  avoid a module that exists only to draw one figure.
