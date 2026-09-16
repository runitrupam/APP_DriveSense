#!/usr/bin/env python3
"""Run DeepSense from a terminal and print the generated run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow execution as ``python scripts/run_pipeline.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.runner import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Process every frame of a road video")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--output-root", default="outputs", help="run output root")
    parser.add_argument("--config", default=None, help="YAML config path")
    args = parser.parse_args()

    def progress(stage: str, done: int, total: int, message: str) -> None:
        if total and (done == total or done == 0):
            print(f"{stage}: {done}/{total} {message}", flush=True)

    result = run_pipeline(args.video, args.output_root, args.config, progress)
    print(json.dumps({
        "status": result.status,
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "frame_count": result.frame_count,
        "successful_frames": result.successful_frames,
        "failed_frames": result.failed_frames,
        "final_jsonl": str(result.final_jsonl),
        "final_json": str(result.final_json),
    }, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
