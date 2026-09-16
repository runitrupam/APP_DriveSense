"""Streamlit surface for running and browsing completed pipeline runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.output.writer import iter_jsonl, read_json
from src.pipeline.runner import run_pipeline

OUTPUT_ROOT = Path("outputs")


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
        progress_bar = st.progress(0.0)
        status = st.empty()

        def progress(stage: str, done: int, total: int, message: str) -> None:
            if total:
                progress_bar.progress(min(1.0, done / total), text=f"{stage}: {done}/{total}")
            status.write(message)

        result = run_pipeline(input_path, root, progress_callback=progress)
        input_path.unlink(missing_ok=True)
        if result.status == "completed":
            st.success(f"Completed {result.frame_count} frames: {result.run_dir}")
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
