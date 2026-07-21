import json

import app.services.human_feedback as hf


def test_append_training_record_roundtrip(tmp_path, monkeypatch):
    log = tmp_path / "fb.jsonl"
    monkeypatch.setenv("HUMAN_FEEDBACK_JSONL", str(log))
    hf.append_training_record({"a": 1, "t": "x"})
    hf.append_training_record({"a": 2})
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1
    assert json.loads(lines[1])["a"] == 2


def test_summarize_tool_calls_strips_payloads():
    tool_calls = [
        {
            "table": "company_profiles",
            "row_count": 3,
            "error": None,
            "row_entity_sanity_passed": True,
            "rows_df": "would be huge",
        }
    ]
    s = hf.summarize_tool_calls_for_training(tool_calls)
    assert s == [
        {
            "table": "company_profiles",
            "row_count": 3,
            "had_error": False,
            "row_entity_sanity_passed": True,
        }
    ]
