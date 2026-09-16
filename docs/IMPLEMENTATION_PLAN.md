# DeepSense Video Scene Analyzer — Implementation Plan

Source: user-provided implementation plan, decomposed into build tasks.
Companion design doc: `docs/superpowers/specs/2026-09-16-deepsense-video-scene-analyzer-design.md`

## Deliverable

`video_scene_analyzer` application in this repository: upload a road video to
Streamlit, process **every frame**, get individually tracked objects plus
lane/motion/road/environment information, exactly **one JSON record per frame**,
plus metrics, validation data, logs, annotated frames and run metadata in a
unique `outputs/RUN_<timestamp>_<uuid>/` directory.

## Build order (V1)

| # | Task | Output | Status |
| --- | --- | --- | --- |
| 0 | Design spec, plan, README | `docs/`, `README.md` | done |
| 1 | Deps + config + utils | `config/config.yaml`, `src/utils/*` | done |
| 2 | Ingestion + frame extraction | `src/ingestion/`, `src/frames/` | done |
| 3 | Detection (YOLO + fallback + registry) | `src/detection/` | done |
| 4 | Tracking (deterministic IoU + velocity) | `src/tracking/` | done |
| 5 | Lane + road analysis + change frames | `src/road/` | done |
| 6 | Motion: trajectory, ego-motion, movement, direction | `src/motion/` | done |
| 7 | Environment heuristics | `src/environment/` | done |
| 8 | Temporal forward/backward + consistency | `src/temporal/` | done |
| 9 | Validation + schemas + writer + metrics | `src/validation/`, `src/output/`, `src/metrics/` | done |
| 10 | Pipeline orchestration + annotation | `src/pipeline/`, `src/output/annotator.py` | done |
| 11 | Streamlit UI | `app.py`, `.streamlit/config.toml` | done |
| 12 | Unit + integration tests | `tests/` | done |
| 13 | End-to-end verification on real/synthetic video | `outputs/RUN_*` | done |

## Stage details

### Ingestion

- `run_id = RUN_<YYYYmmddTHHMMSSZ>_<uuid4[:8]>`.
- Record filename, sha256 hash, size, fps, resolution, frame count, duration,
  codec (ffprobe, `unknown` if unavailable).
- Never drop frames: `CAP_PROP_POS_FRAMES` is not relied on for iteration;
  sequential `read()` is used and each read result is recorded.

### Frame extraction

- Every frame yields a `FramePacket(frame_id, timestamp_ms, image, read_ok,
  error)`; `timestamp_ms = round(frame_id * 1000 / fps)`.
- Blur score = variance of Laplacian; duplicate detection = mean absolute
  difference against the previous frame and an 8x8 dHash distance.
- `frames/frame_index.jsonl` records one line per frame.
- Raw frames persisted only when `output.save_raw_frames: true`.

### Detection

- Primary: Ultralytics YOLO (road-scene COCO pretrained), classes mapped to
  `car, truck, bus, motorcycle, bicycle, person, traffic_light, traffic_sign`.
- Fallback: classical MOG2 background subtraction + contour shape heuristics,
  used automatically when torch/weights are unavailable.
- Every detection stores pixel and normalized bbox, center, class, raw class,
  confidence, frame id and timestamp.
- `detection/detections.jsonl`.

### Tracking

- Deterministic IoU + constant-velocity matching, Hungarian-equivalent greedy
  assignment, `max_age` and `min_hits` gating.
- IDs: `car_001`, `truck_002`, `pedestrian_007`, ...
- Per track: trajectory, first/last frame, confidence, lane(s), displacement.
- Anomalies: id switch candidate, fragmentation, sudden disappearance,
  reappearance, impossible movement, teleportation.
- `tracking/tracks.jsonl`, `tracking/tracks_per_frame.jsonl`.

### Lane / road

- Classical lane detection (ROI, CLAHE, Canny, Hough, slope clustering,
  polynomial fit, vanishing point).
- Lane count from bottom-scanline peak sampling fused with lane-model spacing;
  `unknown` when confidence is low.
- Ego lane = lane polygon containing the bottom-center point (or lane index by
  lateral offset of image center against lane boundaries).
- Road type: `straight|curve_left|curve_right|intersection|merge|split|
  roundabout|unknown` from curvature, lane-line orientation spread and temporal
  lane-count behaviour.
- Change frames are exact: a change is emitted on the first frame it is
  confirmed for `temporal.lane_change_confirmation_frames`, and never smoothed
  away.

### Motion / direction

- Trajectory store with moving-average smoothing over `history_frames`.
- Ego-motion estimated by phase correlation between consecutive grayscale
  frames; per-object residual motion = object image motion minus local global
  motion.
- `state`: `moving|stationary|unknown` from residual displacement against
  `stationary_threshold` (scaled by object size/depth proxy).
- `image_direction`: `forward|backward|left|right|stationary|unknown` in image
  space, using the vanishing point as the forward reference.
- `direction` (relative to ego): `same_direction|opposite_direction|crossing|
  unknown` using vanishing-point convergence plus lane assignment.

### Environment

- Horizon/sky estimation, ExG vegetation mask + components, vertical-edge
  density clusters for buildings, bright-blob + pole heuristics for street
  lights, broad low-texture flanking regions for sidewalks, repeated horizontal
  structures for barriers.
- Vehicles/pedestrians/traffic lights/signs come from detections/tracks.
- Individual `environment_objects[]` plus derived counts.

### Temporal

- Forward pass: frame 1..N producing `primary_result`.
- Backward pass: frame N..1 producing `validation_result`; used to validate,
  never to blindly overwrite.
- Consistency: per-frame agreement on lane count, ego lane, road type and
  object identity (IoU-matched track ids); disagreements preserved.
- Anomaly detection covers the full taxonomy in the design doc.

### Validation / output

- `final_result` + `decision_reason` + `status`.
- Streaming JSONL write per frame (constant memory), JSON array assembled at the
  end, manifest always written.
- Metrics persisted after each stage; `overall.json` aggregates runtime,
  throughput, peak RSS, model versions, config hash and pipeline version.

### UI

Upload, metadata, run control with live progress + per-stage metrics, overall
metrics, frame selector by frame id or timestamp, original frame, annotated
frame, JSON for the selected frame, object table, anomaly list, lane/object
timeline, and downloads for JSON, JSONL, metrics, manifest and ZIP.

## Configuration (defaults)

```yaml
detection:
  confidence_threshold: 0.40
  iou_threshold: 0.50
tracking:
  max_age: 30
  min_hits: 3
motion:
  history_frames: 10
  stationary_threshold: 5
temporal:
  lane_change_confirmation_frames: 3
  object_disappearance_tolerance: 2
output:
  save_raw_frames: false
  save_annotated_frames: true
```

Plus engine selection (`detector: auto|yolo|motion|none`), device, model path,
seed, and per-stage toggles. Full file in `config/config.yaml`.

## Verification gates

1. `pytest tests/unit -q` green.
2. `pytest tests/integration -q` green, including `records == expected_frames`.
3. `python scripts/run_pipeline.py --video <clip> --output-root outputs` produces
   every stage directory, `run_manifest.json`, `final/frame_analysis.jsonl`.
4. `streamlit run app.py` serves the UI and renders a completed run.
5. Metrics files exist for extraction, detection, tracking, motion, lane,
   temporal, validation, environment, overall.
