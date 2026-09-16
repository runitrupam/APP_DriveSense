This looks **architectural**, because you are defining a new end-to-end video-processing system, its data contract, UI, processing pipeline, metrics, persistence, and validation strategy.

I’ll structure the plan so you can paste it directly into Copilot. I’m treating the JSON schema as **part of the plan and proposing a concrete schema**, since there is no existing output contract. The primary design is **Python + Streamlit + computer vision + tracking + deterministic temporal validation**, with optional AI validation only for uncertain cases. This follows the required design-first workflow.

## Proposed design

### Core principle

```text
Video
  ↓
Frame extraction
  ↓
Object detection
  ↓
Object tracking
  ↓
Lane / road analysis
  ↓
Motion + trajectory analysis
  ↓
Temporal consistency
  ↓
Validation
  ↓
Per-frame JSON
  ↓
Metrics + artifacts
  ↓
Streamlit UI
```

Every frame must produce a JSON record. The system must **never silently drop a frame**. If a frame cannot be analyzed reliably, its JSON should explicitly record the failure/uncertainty.

### Recommended V1

Use pretrained computer-vision models first rather than training custom models immediately. The architecture should make model implementations replaceable, so later you can fine-tune/custom-train detection, segmentation, lane, or depth models without rewriting the pipeline.

The LLM should **not** be in the critical path for every frame.

Use it only later as an optional validator for anomalous/ambiguous frames.

---

# Implementation Plan — Video-to-Frame Scene JSON Analyzer

## 1. Project objective

Build a local Python application with a Streamlit UI that allows a user to:

1. Upload a video.
2. Inspect video metadata.
3. Process every frame.
4. Detect and track objects.
5. Analyze road/lane information.
6. Analyze object motion and direction.
7. Maintain temporal consistency between frames.
8. Validate the generated scene state.
9. Generate one structured JSON record for every frame.
10. Display the processing result and important metrics in Streamlit.
11. Store all important outputs and metrics under a unique output directory for every run.
12. Allow the generated JSON and metrics/artifacts to be downloaded/viewed.

The system should support videos of arbitrary practical duration, such as:

```text
1 minute
3 minutes
5 minutes
10 minutes+
```

and should not hard-code assumptions about video duration or frame count.

---

# 2. High-level architecture

```text
                         Streamlit UI
                              │
                              ▼
                       Upload Video
                              │
                              ▼
                       Run Manager
                              │
                              ▼
                    Video Ingestion Layer
                              │
                              ▼
                     Frame Extraction
                              │
                              ▼
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
         Object Detection              Lane/Road Analysis
                │                            │
                ▼                            │
         Object Tracking                    │
                │                            │
                └─────────────┬──────────────┘
                              ▼
                       Motion Analysis
                              │
                              ▼
                    Temporal State Engine
                              │
                       ┌──────┴──────┐
                       ▼             ▼
                   Forward        Backward
                   Analysis       Validation
                       │             │
                       └──────┬──────┘
                              ▼
                       Consistency Engine
                              │
                              ▼
                       Validation Layer
                              │
                              ▼
                     Frame JSON Generator
                              │
                              ▼
                  ┌───────────┴───────────┐
                  ▼                       ▼
             final.jsonl             metrics.json
                  │                       │
                  └───────────┬───────────┘
                              ▼
                         Output Folder
                              │
                              ▼
                         Streamlit UI
```

---

# 3. Technology stack

Use Python.

Recommended initial stack:

```text
Python 3.11+
Streamlit
OpenCV
NumPy
Pydantic
PyTorch
Ultralytics / equivalent detector
ByteTrack / BoT-SORT / equivalent tracker
FFmpeg where useful
Pandas
Matplotlib
```

Do not tightly couple the application to one specific model.

Create interfaces such as:

```python
ObjectDetector
ObjectTracker
LaneAnalyzer
MotionAnalyzer
TemporalValidator
```

so models can later be replaced.

---

# 4. Repository structure

Create:

```text
video_scene_analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│   └── config.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── video_reader.py
│   │
│   ├── frames/
│   │   ├── __init__.py
│   │   └── frame_extractor.py
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── object_detector.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── object_tracker.py
│   │
│   ├── road/
│   │   ├── __init__.py
│   │   ├── lane_detector.py
│   │   └── road_analyzer.py
│   │
│   ├── motion/
│   │   ├── __init__.py
│   │   ├── trajectory.py
│   │   ├── movement.py
│   │   └── direction.py
│   │
│   ├── temporal/
│   │   ├── __init__.py
│   │   ├── state_manager.py
│   │   ├── forward_pass.py
│   │   ├── backward_pass.py
│   │   └── consistency.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validator.py
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── writer.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── collector.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       └── processor.py
│
├── tests/
│   ├── test_video_reader.py
│   ├── test_frame_extractor.py
│   ├── test_motion.py
│   ├── test_temporal.py
│   ├── test_schema.py
│   └── test_pipeline.py
│
└── outputs/
```

---

# 5. Run management

Every processing request must receive a unique `run_id`.

Example:

```text
RUN_20260916_233100_a81f
```

Create:

```text
outputs/
└── RUN_20260916_233100_a81f/
    ├── input/
    ├── frames/
    ├── detection/
    ├── tracking/
    ├── motion/
    ├── temporal/
    ├── validation/
    ├── final/
    ├── metrics/
    └── logs/
```

The input video can either be copied into `input/` or referenced safely, depending on configuration.

---

# 6. Video ingestion

When the user uploads a video through Streamlit:

```text
Upload Video
     ↓
Validate file
     ↓
Generate run_id
     ↓
Create output directories
     ↓
Read metadata
```

Capture:

```json
{
  "filename": "test_drive_001.mp4",
  "fps": 30.0,
  "frame_count": 1800,
  "duration_seconds": 60.0,
  "width": 1920,
  "height": 1080,
  "codec": "...",
  "file_size_bytes": 123456789
}
```

Store:

```text
outputs/<run_id>/metrics/video_metadata.json
```

---

# 7. Frame extraction

Extract frames sequentially.

Each frame must have:

```text
video_id
frame_id
timestamp_ms
width
height
```

For a 30 FPS video:

```text
frame 0 → 0 ms
frame 1 → 33.33 ms
frame 2 → 66.67 ms
...
```

Do not describe timestamps as "millisecond-level truth" beyond the actual source FPS.

The JSON should contain both:

```text
frame_id
timestamp_ms
```

---

# 8. Frame storage strategy

Do not permanently store thousands of JPEG/PNG files unless required.

For V1:

* process frames sequentially
* optionally save sampled/debug frames
* save annotated frames when requested
* keep raw frames in memory where possible

Provide a configuration option:

```yaml
save_raw_frames: false
save_annotated_frames: true
annotated_frame_interval: 30
```

This avoids enormous disk consumption for thousands of videos.

---

# 9. Object detection

Implement an `ObjectDetector` abstraction.

Example interface:

```python
class ObjectDetector:
    def detect(self, frame) -> DetectionResult:
        ...
```

Initial detector should be a pretrained modern object detector.

Start with classes relevant to road scenes:

```text
car
truck
bus
motorcycle
bicycle
pedestrian
traffic_light
traffic_sign
```

Potential additional classes:

```text
animal
construction_vehicle
emergency_vehicle
```

Do not add dozens of categories until there is a downstream requirement.

---

# 10. Detection output

For each detected object:

```json
{
  "class": "car",
  "confidence": 0.97,
  "bbox": {
    "x": 421,
    "y": 285,
    "width": 82,
    "height": 67
  }
}
```

Store intermediate detection output:

```text
detection/detections.jsonl
```

One line per frame.

---

# 11. Detection metrics

After detection, record:

```text
total_frames_processed
frames_with_detections
frames_without_detections
total_detections
detections_by_class
average_confidence
minimum_confidence
maximum_confidence
processing_time_seconds
average_frame_processing_ms
frames_per_second
```

Example:

```json
{
  "stage": "detection",
  "total_frames": 1800,
  "frames_processed": 1800,
  "total_detections": 6321,
  "detections_by_class": {
    "car": 4310,
    "truck": 431,
    "bus": 123,
    "motorcycle": 784,
    "pedestrian": 673
  },
  "average_confidence": 0.91,
  "processing_time_seconds": 83.4,
  "throughput_fps": 21.58
}
```

---

# 12. Object tracking

Pass detections into the tracker.

The tracker must assign stable IDs:

```text
car_001
car_002
car_003
```

The same physical object should retain its ID across frames whenever tracking is successful.

Example:

```text
Frame 100 → car_001
Frame 101 → car_001
Frame 102 → car_001
...
```

Use an established multi-object tracker initially.

---

# 13. Tracking output

For every tracked object:

```json
{
  "track_id": "car_001",
  "class": "car",
  "bbox": {
    "x": 421,
    "y": 285,
    "width": 82,
    "height": 67
  },
  "center": {
    "x": 462,
    "y": 318
  },
  "detection_confidence": 0.97,
  "tracking_confidence": 0.94
}
```

Store:

```text
tracking/tracks.jsonl
```

---

# 14. Tracking metrics

Store:

```text
unique_tracks
tracks_by_class
average_track_length
median_track_length
short_tracks
track_fragmentations
object_enter_events
object_exit_events
average_tracking_confidence
tracking_processing_time
tracking_fps
```

A track that appears for only one frame should be flagged as potentially unreliable rather than automatically considered a stable object.

---

# 15. Motion analysis

For every tracked object, maintain a trajectory:

```text
car_001:
    frame 100 → (x1, y1)
    frame 101 → (x2, y2)
    frame 102 → (x3, y3)
```

Calculate:

```text
delta_x
delta_y
trajectory
movement state
direction
```

Possible movement states:

```text
moving
stationary
unknown
```

Do not force a classification when evidence is insufficient.

---

# 16. Movement calculation

Use a configurable temporal window.

Example:

```text
current frame
previous 3–10 frames
```

Calculate displacement rather than using a single-frame difference.

This reduces noise.

Example:

```text
movement_distance =
distance(current_position, historical_position)
```

If movement is below a calibrated threshold over a sufficiently long window:

```text
stationary
```

otherwise:

```text
moving
```

Thresholds must be configurable.

---

# 17. Direction calculation

Initially use image-space trajectory.

Possible output:

```text
left
right
forward
backward
same_direction
opposite_direction
unknown
```

However, avoid pretending image-space movement automatically equals real-world direction.

Where possible, combine:

```text
trajectory
vehicle orientation
lane position
camera calibration
```

to improve classification.

---

# 18. Lane analysis

Create a separate `LaneDetector`.

Output:

```json
{
  "lane_count": 4,
  "ego_lane": 2,
  "lane_boundaries": [
    {
      "lane_id": 1,
      "left_boundary": [...],
      "right_boundary": [...]
    }
  ],
  "confidence": 0.93
}
```

The exact representation of lane geometry should be designed so the downstream video-generation team can reconstruct it.

Prefer normalized coordinates in addition to pixel coordinates:

```text
pixel coordinates
+
normalized 0–1 coordinates
```

This makes the JSON resolution-independent.

---

# 19. Road analysis

Determine scene-level road properties where feasible:

```text
road_type
lane_count
ego_lane
intersection
roundabout
straight_road
curve
merge
split
```

Use:

```text
unknown
```

rather than hallucinating a scene type.

---

# 20. Environment analysis

Track scene elements relevant to reconstruction:

```text
trees
buildings
pedestrians
traffic lights
traffic signs
street lights
road barriers
sidewalks
```

For countable objects, prefer actual detection/tracking rather than an LLM-generated count.

Where pixel-level reconstruction matters, use segmentation rather than only bounding boxes.

---

# 21. Optional segmentation

Make segmentation a modular component.

It can provide:

```text
road
sidewalk
vehicle
pedestrian
building
vegetation
sky
```

This is especially useful if the downstream team needs more than object bounding boxes.

Do not make segmentation mandatory for the first MVP if it significantly slows development.

---

# 22. Optional depth estimation

Design the schema to support depth later.

For example:

```json
{
  "depth": {
    "relative_depth_m": 18.5,
    "confidence": 0.82
  }
}
```

If reliable metric depth cannot be established, use:

```text
relative_depth
```

rather than claiming exact meters.

Depth is valuable for reconstructing driving scenarios.

---

# 23. Temporal state engine

Create:

```text
SceneState
```

which represents the current understanding of the scene.

It should maintain:

```text
active objects
object histories
lane history
road state
environment state
events
confidence
anomalies
```

For each new frame:

```text
previous state
      +
new CV observations
      ↓
updated state
```

---

# 24. Forward processing

Process:

```text
Frame 1 → Frame N
```

and produce:

```text
forward_state.jsonl
```

This is the normal chronological interpretation.

---

# 25. Backward validation

After the forward pass, process the video backward:

```text
Frame N → Frame 1
```

The backward pass does not need to regenerate everything from scratch if the architecture allows reuse.

Its primary purpose is validation.

Compare:

```text
forward interpretation
vs
backward interpretation
```

for important state variables.

---

# 26. Temporal consistency checks

Examples:

### Object identity

```text
Frame 100 → car_001
Frame 101 → car_001
Frame 102 → car_001
```

Good.

If:

```text
Frame 102 → car_009
```

with almost identical position and appearance, flag a potential tracking identity switch.

### Lane consistency

```text
4 → 4 → 4 → 3 → 3 → 3
```

Potential genuine transition.

But:

```text
4 → 4 → 3 → 4 → 4
```

Potential detection error.

### Object disappearance

If a car suddenly disappears and reappears one frame later, investigate occlusion/tracking failure.

---

# 27. Temporal confidence

Each important state should have:

```text
observation confidence
temporal confidence
```

Example:

```json
{
  "lane_count": {
    "value": 4,
    "confidence": 0.94,
    "temporal_confidence": 0.98
  }
}
```

This is more informative than a single generic confidence value.

---

# 28. Anomaly engine

Create deterministic anomaly rules.

Examples:

```text
lane_count_jump
track_identity_switch
object_teleportation
sudden_object_disappearance
impossible_motion
low_detection_confidence
low_tracking_confidence
blurred_frame
duplicate_frame
frame_decode_failure
forward_backward_disagreement
```

Every anomaly should include:

```json
{
  "type": "lane_count_jump",
  "severity": "medium",
  "message": "Lane count changed from 4 to 2 for one frame.",
  "frame_id": 503
}
```

---

# 29. Optional secondary AI validation

Do not run an LLM/vision model for every frame.

Only send anomalous frames or difficult segments to a secondary model if this feature is enabled.

Example:

```text
1000 frames
    ↓
CV pipeline
    ↓
35 suspicious frames
    ↓
secondary vision model
    ↓
validation
```

This preserves scalability.

The result should never blindly overwrite the deterministic result.

Store:

```text
primary_result
secondary_result
final_result
decision_reason
```

---

# 30. Final frame JSON schema

Use a versioned schema.

For example:

```json
{
  "schema_version": "1.0",

  "video": {
    "video_id": "RUN_VIDEO_001",
    "source_filename": "dashcam.mp4"
  },

  "frame": {
    "frame_id": 235,
    "timestamp_ms": 7833,
    "width": 1920,
    "height": 1080
  },

  "ego_vehicle": {
    "heading": "forward",
    "lane": 2
  },

  "road": {
    "lane_count": 4,
    "ego_lane": 2,
    "road_type": "straight",
    "confidence": 0.94
  },

  "objects": [
    {
      "track_id": "car_001",
      "type": "car",

      "bbox": {
        "x": 421,
        "y": 285,
        "width": 82,
        "height": 67
      },

      "bbox_normalized": {
        "x": 0.219,
        "y": 0.264,
        "width": 0.043,
        "height": 0.062
      },

      "center": {
        "x": 462,
        "y": 318
      },

      "lane": 2,

      "movement": {
        "state": "moving",
        "direction": "same_direction"
      },

      "confidence": {
        "detection": 0.97,
        "tracking": 0.94,
        "movement": 0.91
      }
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

This schema should remain extensible.

---

# 31. Important schema design rule

Do not store only counts.

For example, don't only write:

```json
{
  "cars": 5
}
```

Store the individual objects:

```json
{
  "track_id": "car_001",
  "position": ...,
  "lane": ...,
  "direction": ...
}
```

The downstream animation system needs **where each object is**, not merely how many objects exist.

Counts can be calculated later from the objects.

---

# 32. Coordinate system

Store both:

```text
pixel coordinates
normalized coordinates
```

Pixel:

```text
x = 420
y = 300
```

Normalized:

```text
x = 0.219
y = 0.278
```

Normalized coordinates make the output independent of the source resolution.

---

# 33. Final output files

For every run:

```text
outputs/RUN_ID/
│
├── input/
│   └── original_video.mp4
│
├── final/
│   ├── frame_analysis.jsonl
│   └── frame_analysis.json
│
├── detection/
│   └── detections.jsonl
│
├── tracking/
│   └── tracks.jsonl
│
├── motion/
│   └── motion.jsonl
│
├── temporal/
│   ├── forward.jsonl
│   ├── backward.jsonl
│   └── consistency.jsonl
│
├── validation/
│   └── validation.jsonl
│
├── metrics/
│   ├── video_metadata.json
│   ├── detection_metrics.json
│   ├── tracking_metrics.json
│   ├── motion_metrics.json
│   ├── temporal_metrics.json
│   ├── validation_metrics.json
│   └── run_metrics.json
│
├── annotated/
│   └── sampled_frames/
│
└── logs/
    └── pipeline.log
```

---

# 34. Metrics required after every stage

Create a unified metrics collector.

### Video

```text
duration
fps
resolution
frame_count
file_size
```

### Frame extraction

```text
frames_expected
frames_extracted
frames_failed
extraction_time
throughput
```

### Detection

```text
frames_processed
total_objects
objects_by_class
average_confidence
low_confidence_count
processing_time
FPS
```

### Tracking

```text
unique_tracks
tracks_by_class
average_track_length
track_fragmentations
identity_switch_candidates
tracking_time
FPS
```

### Motion

```text
moving_objects
stationary_objects
unknown_motion
direction_distribution
motion_anomalies
processing_time
```

### Lane

```text
frames_with_lane_detection
frames_without_lane_detection
lane_count_distribution
lane_count_changes
lane_confidence
processing_time
```

### Temporal

```text
forward_frames
backward_frames
consistent_frames
inconsistent_frames
object_identity_conflicts
lane_conflicts
state_conflicts
processing_time
```

### Validation

```text
valid_frames
invalid_frames
anomalous_frames
anomalies_by_type
secondary_validation_count
corrected_frames
unresolved_frames
```

### Overall

```text
total_processing_time
total_frames
successful_frames
failed_frames
overall_throughput
peak_memory
model_versions
configuration
```

---

# 35. Run manifest

Create:

```text
metrics/run_manifest.json
```

containing:

```json
{
  "run_id": "RUN_20260916_233100_a81f",
  "started_at": "...",
  "completed_at": "...",
  "status": "completed",

  "input": {
    "filename": "dashcam.mp4",
    "sha256": "..."
  },

  "models": {
    "object_detector": "...",
    "tracker": "...",
    "lane_detector": "...",
    "segmentation_model": null
  },

  "configuration": {
    "confidence_threshold": 0.4,
    "tracking_threshold": 0.5
  },

  "outputs": {
    "frame_analysis": "final/frame_analysis.jsonl"
  }
}
```

This gives you reproducibility.

---

# 36. Streamlit UI

The UI should have these sections.

## Upload

```text
Video Scene Analyzer

[ Upload Video ]

Filename:
Size:
Duration:
FPS:
Resolution:
Frames:
```

Then:

```text
[ Start Analysis ]
```

---

# 37. Processing progress

Display:

```text
Stage:
Object Detection

Progress:
████████████████░░░░ 80%

Frames:
1440 / 1800

Processing FPS:
21.4

Elapsed:
01:07

Estimated remaining:
00:17
```

Update after every stage.

---

# 38. Stage metrics in UI

Use expandable sections:

```text
Detection
--------------------------------
Frames processed: 1800
Objects detected: 6321
Average confidence: 0.91
Processing time: 83.4 sec
Throughput: 21.58 FPS
```

Then:

```text
Tracking
--------------------------------
Unique tracks: 184
Average track length: 34 frames
Potential ID switches: 7
```

Then:

```text
Motion
--------------------------------
Moving objects: 421
Stationary objects: 109
Unknown: 18
```

Then:

```text
Temporal Validation
--------------------------------
Consistent frames: 1742
Inconsistent frames: 58
```

---

# 39. Frame explorer

Add a UI component allowing:

```text
Frame number: [235]
```

Display:

```text
Original frame
+
Annotated frame
+
JSON
```

Example:

```text
┌───────────────────────────────┐
│                               │
│        Annotated Frame        │
│                               │
└───────────────────────────────┘

Frame: 235
Timestamp: 7833 ms

Objects:
car_001
car_002
pedestrian_001

Lane count: 4
Ego lane: 2

[View JSON]
```

This is extremely useful for debugging.

---

# 40. Visualization

For sampled frames, draw:

```text
bounding boxes
track IDs
lane boundaries
lane IDs
trajectory
confidence
```

Example:

```text
car_001 [0.97]
car_002 [0.93]

Lane 2
Lane 3
```

Allow the user to compare:

```text
Original
vs
Annotated
```

---

# 41. Timeline view

Add a simple timeline:

```text
Frame: 0                  900

Lane count:
████████████████████ 4
                 └── 3

Object count:
████████████████████████
```

Mark anomalies:

```text
             X
             │
─────────────┼──────────────
```

This makes temporal inconsistencies immediately visible.

---

# 42. Download controls

Provide buttons:

```text
Download Final JSONL
Download Final JSON
Download Run Metrics
Download Run Manifest
Download Annotated Frames
```

Optionally provide:

```text
Download Complete Run
```

as a ZIP.

---

# 43. Error handling

The pipeline must not crash because of one bad frame.

If:

```text
frame 503
```

cannot be decoded:

```json
{
  "frame_id": 503,
  "timestamp_ms": 16766,
  "validation": {
    "status": "failed",
    "anomalies": [
      {
        "type": "frame_processing_error"
      }
    ]
  }
}
```

Continue processing.

At the end:

```text
processed: 1799
failed: 1
```

---

# 44. Determinism

Make inference reproducible where possible.

Store:

```text
model version
model weights hash/version
configuration
thresholds
Python version
library versions
```

Avoid random behavior unless explicitly required.

Set seeds where applicable.

The same:

```text
video
+
model
+
configuration
```

should produce the same result as far as the underlying inference stack permits.

---

# 45. Model configuration

Do not hard-code thresholds.

Example:

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
```

These values should be experimental and tuned against your dataset.

---

# 46. Testing strategy

Create unit tests for deterministic components.

Test:

```text
timestamp calculation
bounding box normalization
distance calculation
trajectory calculation
movement classification
direction calculation
temporal smoothing
lane transition detection
JSON schema validation
```

Example:

```text
position:
(100,100)
→
(110,100)
→
(120,100)

expected:
moving
direction:
right
```

---

# 47. Integration tests

Use a short test video.

For example:

```text
10-second video
```

Verify:

```text
frames extracted == expected
JSON records == expected frame count
track IDs exist
metrics generated
output files generated
Streamlit displays result
```

---

# 48. Accuracy evaluation

This is essential.

Create a manually annotated ground-truth dataset.

For selected frames:

```text
Ground truth
      ↓
Expected objects
Expected lanes
Expected positions
Expected directions
Expected movement
      ↓
Compare
      ↑
System output
```

Calculate:

```text
precision
recall
F1
mAP
tracking metrics
lane accuracy
count accuracy
direction accuracy
movement accuracy
```

Do not use one overall "accuracy" number as the sole measure.

---

# 49. Test temporal consistency separately

Example ground truth:

```text
car_001 exists from frame 100–300
```

Measure whether your system maintains:

```text
car_001
```

throughout the sequence.

Track:

```text
identity switches
fragmentations
false disappearance
false reappearance
```

This is particularly important for your downstream reconstruction.

---

# 50. Performance optimization

After correctness:

### Batch inference

Process multiple frames together where supported.

### GPU

Use GPU when available.

### Parallelism

Separate independent video jobs.

### Intermediate caching

Don't rerun detection if only the JSON schema changes.

### JSONL streaming

Write frame results incrementally rather than keeping thousands of frames in memory.

### Avoid unnecessary frame images

Do not save every raw frame by default.

---

# 51. Processing architecture for thousands of videos

The first Streamlit version can process one uploaded video.

Design the backend so later you can add:

```text
multiple videos
      ↓
job queue
      ↓
workers
      ↓
GPU inference
      ↓
output/<run_id>
```

Do not redesign the core pipeline when batch processing is introduced.

The Streamlit app should simply call:

```python
process_video(video_path, config)
```

rather than containing the actual CV logic.

---

# 52. Version everything

Every output should identify:

```text
schema_version
pipeline_version
model_version
configuration_version
```

Example:

```json
{
  "schema_version": "1.0",
  "pipeline_version": "0.1.0",
  "object_detector_version": "...",
  "lane_detector_version": "..."
}
```

This matters because the downstream team may generate videos using JSON created six months earlier.

---

# 53. Important design decision — no silent corrections

Suppose:

```text
CV says:
4 lanes

Temporal engine says:
3 lanes

Secondary model says:
4 lanes
```

Do not simply output:

```text
4
```

without preserving what happened.

Instead:

```json
{
  "lane_count": 4,
  "validation": {
    "status": "resolved",
    "sources": {
      "primary_cv": 3,
      "temporal_state": 3,
      "secondary_validator": 4
    },
    "decision": "secondary_validator"
  }
}
```

This gives you an audit trail.

---

# 54. Important design decision — unknown is valid

The system should be allowed to say:

```json
{
  "direction": "unknown"
}
```

instead of incorrectly saying:

```json
{
  "direction": "opposite_direction"
}
```

Likewise:

```text
lane_count: unknown
depth: unknown
movement: unknown
```

when the evidence isn't sufficient.

For your use case, **explicit uncertainty is safer than fabricated precision**.

---

# 55. V1 scope

Do not try to solve everything immediately.

V1 should contain:

```text
✓ Streamlit upload
✓ Video metadata
✓ Frame extraction
✓ Object detection
✓ Object tracking
✓ Basic movement
✓ Basic direction
✓ Basic lane detection
✓ Temporal state
✓ Forward processing
✓ Backward validation
✓ Consistency checks
✓ Per-frame JSON
✓ JSONL output
✓ Metrics
✓ Annotated frames
✓ Run folders
✓ UI progress
✓ Frame explorer
✓ Download outputs
```

---

# 56. V2 scope

After V1 evaluation:

```text
→ better lane model
→ segmentation
→ depth estimation
→ improved direction
→ camera calibration
→ BEV coordinates
→ improved temporal reasoning
→ custom-trained detector
→ custom lane model
→ selective secondary vision model
```

---

# 57. V3 production scale

Then:

```text
multiple videos
      ↓
job queue
      ↓
GPU worker pool
      ↓
batch inference
      ↓
object storage
      ↓
metadata database
      ↓
monitoring
      ↓
downstream API
```

Streamlit can remain the demonstration/debugging UI while the processing engine becomes independently deployable.

---

# 58. Definition of Done

The implementation is considered complete for V1 only when:

```text
[ ] User can upload an MP4 from Streamlit.

[ ] System correctly reads FPS, resolution and frame count.

[ ] Every source frame receives exactly one frame_id.

[ ] Every source frame receives a JSON record.

[ ] Object detections are included in each applicable frame.

[ ] Objects have persistent track IDs.

[ ] Object coordinates are stored.

[ ] Normalized coordinates are stored.

[ ] Movement is calculated.

[ ] Direction is calculated where sufficiently supported.

[ ] Lane information is included where detectable.

[ ] Temporal state is maintained.

[ ] Forward pass is executed.

[ ] Backward validation is executed.

[ ] Temporal inconsistencies are detected.

[ ] Failed/uncertain frames are explicitly marked.

[ ] No frame is silently discarded.

[ ] Detection metrics are persisted.

[ ] Tracking metrics are persisted.

[ ] Motion metrics are persisted.

[ ] Lane metrics are persisted.

[ ] Temporal metrics are persisted.

[ ] Validation metrics are persisted.

[ ] Run manifest is persisted.

[ ] Final JSONL is persisted.

[ ] Final JSON is persisted.

[ ] Annotated frames can be generated.

[ ] Streamlit displays processing progress.

[ ] Streamlit displays stage metrics.

[ ] Streamlit provides a frame explorer.

[ ] Streamlit displays the frame JSON.

[ ] User can download final JSON.

[ ] User can download metrics.

[ ] Tests exist for deterministic logic.

[ ] A representative evaluation dataset exists.

[ ] Accuracy is measured separately by task.
```

---

# 59. The most important implementation rule

**Do not start by trying to make the output "perfect."**

Start by making the pipeline observable:

```text
Video
 ↓
Detection
 ↓
Metrics
 ↓
Tracking
 ↓
Metrics
 ↓
Motion
 ↓
Metrics
 ↓
Temporal
 ↓
Metrics
 ↓
Final JSON
```

Every stage must produce both:

```text
RESULT
+
METRICS
```

That way, when the final JSON is wrong, you can determine whether the problem originated from:

```text
detection
    ↓
tracking
    ↓
lane detection
    ↓
motion
    ↓
temporal logic
    ↓
validation
    ↓
JSON generation
```

rather than having one black-box "AI video analyzer" that simply produces a wrong answer.

---

## Final architecture in one sentence

**Build a Python/Streamlit video-analysis application where pretrained computer-vision models perform perception, tracking maintains object identity, deterministic Python algorithms calculate motion/direction/state, forward and backward temporal passes enforce consistency, every frame produces a versioned JSON record, and every processing stage persists measurable metrics and intermediate artifacts under a unique run directory.**

This is the architecture I would give to Copilot as the implementation specification.
