# DeepSense Video Scene Analyzer

Upload a road or dashcam video and get **exactly one structured JSON record per
video frame** describing lanes, road state, vehicles, pedestrians, tracking IDs,
movement, direction, traffic controls and the surrounding environment.

Per-frame analysis is deterministic Computer Vision / ML. No LLM is used as the
per-frame analyzer.

When the pipeline cannot support a value with confidence, the value is
`unknown`. Nothing is silently invented or silently corrected.

- Design spec: `docs/superpowers/specs/2026-09-16-deepsense-video-scene-analyzer-design.md`
- Implementation plan: `docs/IMPLEMENTATION_PLAN.md`

## Pipeline

```
Video
  -> Ingestion (metadata + sha256)
  -> Frame Extraction (every frame, no silent drops)
  -> Detection
  -> Tracking
  -> Lane / Road Analysis
  -> Motion + Direction
  -> Environment Analysis
  -> Temporal State
  -> Backward Pass (validation)
  -> Consistency Checks
  -> Validation
  -> Per-Frame JSON
  -> Metrics + Artifacts
  -> Streamlit UI
```

Every stage is an independent module with a narrow interface, so detectors,
trackers and lane models can be replaced without touching the schema.

## Requirements

- Python 3.11+
- ffmpeg / ffprobe (metadata, codec probing)
- See `requirements.txt`. PyTorch CPU wheels are sufficient; CUDA or Apple
  Metal acceleration is used automatically when available.

## Installation

```bash
# System dependency
apt-get install -y ffmpeg

# Install all Python dependencies, including the semantic YOLO detector
python3 -m pip install --break-system-packages -r requirements.txt

# On Linux, you may install CPU-only PyTorch wheels explicitly if preferred:
# python3 -m pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The default detector is the semantic YOLO backend. If `ultralytics` or its
weights are unavailable, the pipeline falls back to a classical motion detector
so a run still completes; that fallback is much less reliable for vehicle
classification. The detector that actually ran is recorded in `run_manifest.json`
and in the metrics. Check that `components.detector.engine` is `yolo` before
trusting vehicle classifications.

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

The UI provides: upload, video metadata, start analysis with live stage
progress, per-stage and overall metrics, frame selector by frame ID or
timestamp, original and annotated frames, the JSON record for the selected
frame, the object table, the anomaly list, a timeline of lane/object changes,
and downloads for JSON, JSONL, metrics, manifest and a full ZIP.

### Command line

```bash
# Process a video
python scripts/run_pipeline.py --video path/to/road.mp4

# Use an explicit config
python scripts/run_pipeline.py --video path/to/road.mp4 --config config/config.yaml

# Point at a different output root
python scripts/run_pipeline.py --video path/to/road.mp4 --output-root outputs

# Make a deterministic synthetic road clip (used by the tests)
python scripts/make_test_video.py --out outputs/test_data/synthetic_road.mp4 --frames 60

# Optional: download a small public road clip for a realistic smoke run
python scripts/download_sample_video.py --out outputs/test_data/sample_road.mp4
```

## Configuration

Defaults live in `config/config.yaml`.

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

Additional keys select the detection engine (`auto`, `yolo`, `motion`, `none`),
device, model weights, random seed, and per-stage switches. Every resolved value
is written to `run_manifest.json` together with a config hash.

## Output

Each run gets its own directory:

```
outputs/RUN_<timestamp>_<uuid>/
  input/        input metadata
  frames/       frame_index.jsonl, optional raw frames
  detection/    detections.jsonl
  tracking/     tracks.jsonl, tracks_per_frame.jsonl
  motion/       motion.jsonl
  temporal/     forward_frames.jsonl, backward_frames.jsonl, lane_events.jsonl
  validation/   validation.jsonl
  final/        frame_analysis.jsonl, frame_analysis.json
  metrics/      <stage>.json, overall.json
  annotated/    annotated frames, timeline.png
  logs/         run.log
  run_manifest.json
```

`final/frame_analysis.jsonl` contains **one line per frame**, in frame order.
`final/frame_analysis.json` is the same data as a JSON array.

### Record shape

```json
{
  "schema_version": "1.0",
  "frame": {
    "frame_id": 235,
    "timestamp_ms": 7833,
    "width": 1920,
    "height": 1080
  },
  "ego_vehicle": {"heading": "forward", "lane": 2},
  "road": {"lane_count": 4, "ego_lane": 2, "road_type": "straight", "confidence": 0.94},
  "objects": [
    {
      "track_id": "car_001",
      "type": "car",
      "bbox": {"x": 421, "y": 285, "width": 82, "height": 67},
      "bbox_normalized": {"x": 0.219, "y": 0.264, "width": 0.043, "height": 0.062},
      "center": {"x": 462, "y": 318},
      "lane": 2,
      "movement": {"state": "moving", "direction": "same_direction"},
      "confidence": {"detection": 0.97, "tracking": 0.94, "movement": 0.91}
    }
  ],
  "environment": {
    "pedestrians": 2,
    "trees": 8,
    "buildings": 4,
    "traffic_lights": 1,
    "traffic_signs": 2,
    "street_lights": 5
  },
  "events": [],
  "validation": {
    "status": "valid",
    "anomalies": [],
    "forward_backward_consistent": true
  }
}
```

Records also include `run_id` and a `decision` block that preserves
`primary_result`, `validation_result`, `final_result` and `decision_reason`, as
required when forward and backward passes disagree. Details are in the design
spec.

### Anomalies

`lane_count_jump`, `track_identity_switch`, `object_teleportation`,
`object_disappearance`, `reappearance`, `impossible_motion`,
`low_detection_confidence`, `low_tracking_confidence`, `blurred_frame`,
`duplicate_frame`, `frame_decode_failure`, `forward_backward_disagreement`,
`unstable_lane_estimate`.

A frame that fails to decode still produces a record with
`validation.status = "failed"` and a `frame_decode_failure` anomaly, so the
number of JSON records always equals the number of expected frames.

## Tests

```bash
python3 -m pytest tests/unit -q
python3 -m pytest tests/integration -q
python3 -m pytest -q
```

The reusable runner is `src.pipeline.runner.run_pipeline`. It creates the run
and input copy before probing the video, so an invalid input still receives a
failure manifest rather than an ambiguous partial-success result. Forward and
reverse summaries are compared in `validation/validation.jsonl`; disagreements
are preserved in each final record's `validation` and `decision` blocks.

Integration tests render a deterministic synthetic road clip, run the full
pipeline, and assert one JSON record per frame, tracked objects, per-stage
metrics, the output directory layout, and a complete manifest.

## Determinism

A fixed seed is applied to `random`, `numpy` and `torch`. Tracking uses
deterministic ordered assignment. The manifest records seed, model versions,
library versions and a hash of the resolved configuration, so the same video
plus the same model and config reproduces the same output as far as inference
permits.

## Limitations (V1)

- Lane detection is classical and geometric; it degrades on worn markings, rain
  and night footage and reports `unknown` rather than guessing.
- Trees, buildings, street lights, sidewalks and barriers come from
  deterministic heuristics, not a trained model. Each carries a confidence and
  its `source`, and can be replaced by a model later without schema changes.
- `traffic_sign` detection is limited to the sign classes the underlying
  detector knows; unknown sign types are not fabricated.
- No monocular depth, BEV or camera calibration yet; direction and speed are
  image-space derived with an ego-motion correction, and are marked `unknown`
  when the evidence is weak.
