"""Streamlit surface for running and browsing completed pipeline runs."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from src.output.writer import iter_jsonl, read_json
from src.pipeline.runner import run_pipeline

OUTPUT_ROOT = Path("outputs")

PIPELINE_STAGES = (
    ("ingestion", "1. Ingestion", "Probe metadata and prepare the input video."),
    ("forward", "2. Forward analysis", "Read every frame and run detection, tracking, lanes, motion, and environment analysis."),
    ("backward", "3. Backward validation", "Read frames in reverse and compare the independent pass with the forward pass."),
    ("finalization", "4. Finalization", "Write validated JSON/JSONL, metrics, annotations, and the run manifest."),
)
STAGE_WEIGHTS = {"forward": 0.45, "backward": 0.45, "finalization": 0.10}
FINALIZATION_ESTIMATE_SECONDS = 5


def _format_duration(seconds: float | None) -> str:
    """Return a short human-readable duration for the live progress panel."""
    if seconds is None or seconds < 0:
        return "calculating…"
    rounded = max(0, int(seconds))
    minutes, remainder = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def _overall_progress(stage: str, done: int, total: int) -> float:
    """Map a stage's frame progress to one monotonic overall progress value."""
    completed_weight = 0.0
    for key, _, _ in PIPELINE_STAGES:
        if key == stage:
            break
        completed_weight += STAGE_WEIGHTS.get(key, 0.0)
    stage_fraction = min(1.0, done / total) if total > 0 else 0.0
    return min(1.0, completed_weight + STAGE_WEIGHTS.get(stage, 0.0) * stage_fraction)


def _render_pipeline_status(st: Any, panel: Any, state: dict[str, Any]) -> None:
    """Render the checklist and live timing information without changing pipeline state."""
    stage = state.get("stage")
    done = int(state.get("done", 0))
    total = int(state.get("total", 0))
    elapsed = max(0.0, time.monotonic() - float(state["started"]))
    rate = done / elapsed if done > 0 and elapsed > 0 else 0.0
    remaining_frames = max(0, total - done) if total else None
    eta = None
    if rate > 0 and remaining_frames is not None:
        remaining_work = remaining_frames
        if stage == "forward":
            remaining_work += total
        eta = remaining_work / rate + (FINALIZATION_ESTIMATE_SECONDS if stage in {"forward", "backward"} else 0)
    current_label = next((label for key, label, _ in PIPELINE_STAGES if key == stage), "Preparing")

    with panel.container():
        st.markdown("#### Live pipeline status")
        metrics = st.columns(4)
        metrics[0].metric("Currently running", current_label)
        metrics[1].metric("Elapsed", _format_duration(elapsed))
        metrics[2].metric("Frames left in current stage", "—" if remaining_frames is None else f"{remaining_frames:,}")
        metrics[3].metric("Estimated time left", _format_duration(eta))
        if rate:
            st.caption(f"Current throughput: {rate:.1f} frames/sec. The estimate improves as more frames are processed.")
        else:
            st.caption("Calculating throughput and estimated completion time…")

        rows = []
        stage_index = {key: index for index, (key, _, _) in enumerate(PIPELINE_STAGES)}
        current_index = stage_index.get(stage, -1)
        for key, label, description in PIPELINE_STAGES:
            index = stage_index[key]
            if state.get("finished") or index < current_index:
                status = "✅ Complete"
            elif key == stage:
                status = f"🔄 Running ({done:,}/{total:,})" if total else "🔄 Running"
            else:
                status = "⏳ Pending"
            rows.append({"Stage": label, "Status": status, "What it does": description})
        st.table(rows)


def _runs(root: Path) -> list[Path]:
    return sorted((path for path in root.glob("RUN_*") if path.is_dir()), reverse=True)


def _download(st: Any, path: Path, label: str, mime: str = "application/octet-stream") -> None:
    if path.exists() and path.is_file():
        st.download_button(label, path.read_bytes(), file_name=path.name, mime=mime, key=f"download-{path}")


def _show_run(st: Any, run_dir: Path) -> None:
    st.subheader(run_dir.name)
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    metadata = manifest.get("metadata") or {}
    st.write({"status": manifest.get("status"), "filename": metadata.get("filename"), "fps": metadata.get("fps"), "frame_count": metadata.get("frame_count"), "codec": metadata.get("codec"), "detector": (manifest.get("components") or {}).get("detector", {}).get("engine")})

    final_jsonl = run_dir / "final" / "frame_analysis.jsonl"
    records = list(iter_jsonl(final_jsonl)) if final_jsonl.exists() else []
    if records:
        frame_ids = [int(record.get("frame", {}).get("frame_id", 0)) for record in records]
        selected = st.slider("Frame", min_value=min(frame_ids), max_value=max(frame_ids), value=min(frame_ids), key=f"frame-{run_dir.name}")
        record = next((item for item in records if item.get("frame", {}).get("frame_id") == selected), records[0])
        st.json(record, expanded=True)
        annotated = run_dir / "annotated" / f"annotated_{selected:06d}.jpg"
        if annotated.exists():
            st.image(str(annotated), caption=f"Annotated frame {selected}")
    else:
        st.warning("No final records were persisted for this run.")

    st.subheader("Downloads")
    _download(st, run_dir / "final" / "frame_analysis.json", "Download full JSON", "application/json")
    _download(st, final_jsonl, "Download JSONL", "application/x-ndjson")
    _download(st, run_dir / "metrics" / "overall.json", "Download metrics", "application/json")
    _download(st, manifest_path, "Download manifest", "application/json")
    zip_path = run_dir / f"{run_dir.name}.zip"
    _download(st, zip_path, "Download ZIP", "application/zip")

    metrics_path = run_dir / "metrics" / "overall.json"
    if metrics_path.exists():
        with st.expander("Metrics and timeline"):
            st.json(read_json(metrics_path), expanded=False)
            timeline = run_dir / "annotated" / "timeline.png"
            if timeline.exists():
                st.image(str(timeline))


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="DeepSense Video Scene Analyzer", layout="wide")
    st.title("DeepSense Video Scene Analyzer")
    st.caption("One explicit structured record per decoded frame; uncertain values remain unknown.")
    root_text = st.text_input("Output root", str(OUTPUT_ROOT))
    root = Path(root_text)
    root.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader("Upload a road or dashcam video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if uploaded is not None and st.button("Start analysis", type="primary"):
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(uploaded.getbuffer())
            input_path = Path(temporary.name)

        st.subheader("Run overview")
        st.write({"input file": uploaded.name, "size": f"{uploaded.size / (1024 * 1024):.1f} MB", "supported stages": len(PIPELINE_STAGES)})
        progress_bar = st.progress(0.0, text="Preparing pipeline…")
        status = st.empty()
        live_panel = st.empty()
        state: dict[str, Any] = {"started": time.monotonic(), "stage": None, "done": 0, "total": 0, "finished": False}
        _render_pipeline_status(st, live_panel, state)

        def progress(stage: str, done: int, total: int, message: str) -> None:
            state.update({"stage": stage, "done": done, "total": total})
            progress_bar.progress(_overall_progress(stage, done, total), text=f"{stage.title()}: {done:,}/{total:,} frames")
            status.write(message)
            _render_pipeline_status(st, live_panel, state)

        try:
            result = run_pipeline(input_path, root, progress_callback=progress)
        finally:
            input_path.unlink(missing_ok=True)
        state.update({"stage": "finalization", "done": 1, "total": 1, "finished": result.status == "completed"})
        final_progress = 1.0 if result.status == "completed" else _overall_progress("finalization", 0, 1)
        progress_bar.progress(final_progress, text="Completed" if result.status == "completed" else "Run failed")
        _render_pipeline_status(st, live_panel, state)
        if result.status == "completed":
            st.success(f"Completed {result.frame_count:,} frames in {_format_duration(time.monotonic() - state['started'])}: {result.run_dir}")
        else:
            st.error(f"Run failed; inspect manifest: {result.manifest}")

    runs = _runs(root)
    if runs:
        selected_name = st.selectbox("Browse completed run", [path.name for path in runs])
        _show_run(st, root / selected_name)
    else:
        st.info("No runs yet. Upload a video to start.")


if __name__ == "__main__":
    main()
