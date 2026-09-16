from __future__ import annotations

import json

from src.output.writer import RunStore, iter_jsonl
from src.validation.validator import validate_frame


def test_run_store_streams_valid_json_array(tmp_path):
    store = RunStore.create(tmp_path, run_id="RUN_test")
    store.append_jsonl("final", "records.jsonl", {"frame_id": 0})
    store.append_jsonl("final", "records.jsonl", {"frame_id": 1})
    store.close()
    store.write_json_array("final", "records.json", iter_jsonl(store.path("final", "records.jsonl")))

    assert json.loads(store.path("final", "records.json").read_text()) == [{"frame_id": 0}, {"frame_id": 1}]


def test_validator_preserves_forward_backward_disagreement():
    forward = {
        "frame_id": 2,
        "read_ok": True,
        "primary": {"lane_count": 4, "ego_lane": 2, "road_type": "straight"},
        "objects": [],
    }
    backward = {
        "frame_id": 2,
        "read_ok": True,
        "primary": {"lane_count": 3, "ego_lane": 2, "road_type": "straight"},
        "objects": [],
    }

    result = validate_frame(forward, backward)

    assert result.consistent is False
    assert "forward_backward_disagreement" in result.anomalies
    assert result.details[0]["field"] == "lane_count"
