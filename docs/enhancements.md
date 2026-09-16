# DeepSense Video Scene Analyzer — Production Enhancements

This document tracks future enhancements needed to move DeepSense from a working prototype toward a production-ready video analysis system. It is intentionally planning-only: no implementation is implied by the items below.

## Current baseline

The repository already contains modular stages for ingestion, frame extraction, detection, tracking, lane/road analysis, motion, environment analysis, temporal processing, validation, output persistence, a CLI runner, and a Streamlit interface.

The most important production gap is not the per-frame schema itself; it is operational hardening around long-running, high-resolution jobs, reproducibility, observability, resource management, and reliable user-facing completion.

## Priority roadmap

### P0 — Reliability and correctness

- **Complete-run state machine:** Represent runs as `queued`, `running`, `completed`, `failed`, or `cancelled`, with durable status updates and resumable stage boundaries.
- **Crash-safe finalization:** Guarantee that interrupted runs produce a clear failure/cancelled manifest and never appear as completed.
- **Frame-count invariant:** Compare decoded frames, metadata frame count, and final JSON records; explicitly explain variable-frame-rate or corrupt-video discrepancies.
- **Idempotent reruns:** Derive a stable input fingerprint and configuration hash so users can detect duplicate work and optionally reuse a completed run.
- **Cancellation support:** Allow the UI and CLI to stop a run cleanly, close writers, retain partial diagnostics, and release resources.
- **Backpressure and bounded memory:** Keep frame queues, JSON materialization, image buffers, and UI state bounded for multi-minute or multi-hour videos.
- **Configuration validation:** Validate YAML at startup with typed schemas, clear errors, versioned defaults, and compatibility checks.
- **Schema versioning:** Version the per-frame JSON contract and provide migration or compatibility tooling for future schema changes.
- **Golden fixture suite:** Maintain small deterministic videos covering decode failures, no detections, lane changes, occlusions, duplicates, variable frame rates, and empty scenes.

### P1 — Performance and resource efficiency

- **Separate analysis resolution from archival resolution:** Process at a configurable model resolution while retaining source metadata and optional keyframes.
- **Configurable annotation sampling:** Save annotated images every N frames or only for anomalies; do not require a 4K JPEG for every frame.
- **Single-pass optimization:** Avoid repeating expensive detection, lane, and environment work in the reverse validation pass where cached intermediate data is sufficient.
- **Parallel stage strategy:** Evaluate safe parallelism for independent stages without compromising frame order, deterministic tracking, or resource limits.
- **Hardware acceleration:** Support CPU, CUDA, Apple Silicon/MPS, and model-specific device selection with transparent capability reporting.
- **Model lifecycle management:** Pin model versions, cache weights locally, verify checksums, and expose download/provisioning status.
- **Storage policies:** Add retention rules, compression options, artifact size estimates, and configurable cleanup for raw frames, annotations, and temporary files.
- **Throughput benchmarks:** Track frames per second, stage latency, peak RSS, disk growth, and p95 processing time across representative resolutions.

### P1 — Detection, tracking, and analysis quality

- **Production detector backends:** Support a tested YOLO/ONNX/TensorRT backend matrix with explicit quality and performance profiles.
- **Robust tracking:** Evaluate ByteTrack, BoT-SORT, or equivalent against the current IoU/velocity tracker, especially under occlusion and crowded traffic.
- **Camera calibration:** Support intrinsic calibration, camera mounting profiles, and perspective-aware distance/speed estimation.
- **Lane model improvements:** Add segmentation or learned lane detection for poor lighting, curves, worn markings, intersections, and construction zones.
- **Confidence calibration:** Measure precision/recall and calibrate confidence values against labeled validation clips.
- **Temporal event quality:** Add benchmarked rules for lane changes, cut-ins, near misses, stopped vehicles, and object disappearance/reappearance.
- **Environment quality evaluation:** Replace heuristic-only counts with evaluated models or clearly labeled approximate estimates.
- **Human review loop:** Let users correct selected detections, lanes, or events and export corrections for future evaluation or training.

### P1 — Observability and operations

- **Structured logs:** Emit machine-readable logs with run ID, stage, frame ID, duration, error class, and backend information.
- **Metrics dashboard:** Track success rate, failure rate, throughput, stage latency, detection counts, anomalies, disk use, and queue time.
- **Error taxonomy:** Standardize retryable, input, dependency, model, resource, and data-quality errors.
- **Run diagnostics bundle:** Package logs, manifest, configuration, versions, metrics, and representative anomalous frames into a support artifact.
- **Health checks:** Add startup checks for Python version, FFmpeg/FFprobe, OpenCV codecs, writable output storage, model availability, and memory headroom.
- **Alerting:** Notify operators when jobs fail, stall, exceed resource budgets, or produce unusual anomaly rates.
- **Audit trail:** Record who started a run, which input and configuration were used, when artifacts were created, and whether a result was manually reviewed.

### P1 — UI and user experience

- **Run queue and history:** Browse, search, filter, rename, compare, and delete runs with explicit confirmation and retention rules.
- **Live stage progress:** Show stage-level progress, current frame, estimated completion time, throughput, and resource usage.
- **Full-response browser:** Support frame search, timestamp navigation, anomaly filters, object filters, and complete JSON inspection without loading all records into browser memory.
- **Artifact explorer:** Display links and summaries for input metadata, frame index, detections, tracks, validation, metrics, annotations, and logs.
- **Export options:** Provide JSON, JSONL, CSV summaries, images, ZIP bundles, and optionally Parquet for large datasets.
- **Failure recovery UI:** Explain why a run failed, show the last completed stage/frame, and offer safe retry options.
- **Accessibility:** Add keyboard navigation, readable contrast, screen-reader labels, and non-color-only status indicators.
- **Authentication and authorization:** Add access control before exposing the UI beyond a trusted local machine.

### P2 — Testing and release engineering

- **Coverage expansion:** Add unit, integration, property-based, and end-to-end tests with an agreed coverage threshold.
- **Browser tests:** Verify upload, run initiation, progress, run selection, JSON display, downloads, and failure states using Playwright.
- **Contract tests:** Validate every emitted record against the versioned schema and test backward compatibility.
- **Load tests:** Process long videos and concurrent jobs while recording CPU, memory, disk, and throughput limits.
- **Fault-injection tests:** Simulate missing codecs, truncated videos, disk-full conditions, model load failures, corrupt frames, and interrupted processes.
- **Reproducibility tests:** Run the same input/configuration twice and compare deterministic fields, with documented exceptions for nondeterministic backends.
- **CI pipeline:** Run formatting, linting, type checks, tests, security scans, dependency checks, and a small smoke video on every change.
- **Release artifacts:** Publish pinned dependency files, model manifests, container images, changelogs, and reproducible release metadata.

### P2 — Deployment and scale

- **Containerization:** Provide a production Docker image with FFmpeg, pinned Python dependencies, model provisioning, non-root execution, and health checks.
- **Worker architecture:** Move long-running processing out of the Streamlit process into a worker service with a queue and durable job store.
- **Object storage:** Store source videos and generated artifacts in S3-compatible storage rather than relying only on local disk.
- **Database-backed metadata:** Store run status, ownership, configuration, metrics, and artifact references in a transactional database.
- **Horizontal scaling:** Support multiple workers with bounded concurrency and device-aware scheduling.
- **Multi-tenant isolation:** Apply per-user quotas, storage limits, job priorities, and strict artifact access controls.
- **Deployment strategy:** Define staging/production environments, migrations, rollback procedures, backups, and disaster recovery objectives.

## Security and privacy

- Validate file type from content, not only filename extension.
- Enforce upload size, duration, resolution, and decompression-bomb limits.
- Sanitize all displayed filenames, metadata, logs, and user-provided paths.
- Prevent path traversal when selecting output roots or downloading artifacts.
- Run video processing with least-privilege filesystem and OS permissions.
- Avoid logging sensitive video metadata or image content unnecessarily.
- Define retention and deletion controls for uploaded videos and generated frames.
- Scan dependencies and container images for known vulnerabilities.
- Document whether videos may contain faces, license plates, location data, or other personal information.
- Add redaction or blurring options for faces and license plates before exporting artifacts.

## Data and model governance

- Maintain a model card for every detector, tracker, lane model, and environment model.
- Record model name, version, checksum, training source, license, and runtime backend in every manifest.
- Build a representative evaluation dataset across weather, lighting, road types, camera positions, traffic density, and geographies.
- Track quality metrics by class and scenario rather than only overall averages.
- Monitor drift in camera characteristics, road environments, object distributions, and anomaly rates.
- Define a review and approval process for model upgrades.
- Preserve provenance for derived records, corrections, exports, and manually overridden values.

## Suggested implementation order

1. Crash-safe lifecycle, cancellation, frame-count invariants, and configuration validation.
2. Resource controls: annotation sampling, bounded memory, storage estimates, and retention.
3. Full test matrix, browser tests, fault injection, and CI checks.
4. Structured observability, diagnostics bundles, and run history improvements.
5. Model/version governance and measured quality evaluation.
6. Worker queue, object storage, database metadata, authentication, and scalable deployment.

## Definition of production readiness

The repository should be considered production-ready only when a supported deployment can process representative videos repeatedly without silent frame loss, preserve complete provenance, recover or fail clearly under faults, stay within documented resource limits, expose actionable diagnostics, protect uploaded data, and pass automated correctness, security, performance, and browser-level verification gates.
