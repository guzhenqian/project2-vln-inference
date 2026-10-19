"""离线检查提示词切换、实际发送内容和原始回答保留；不连接模型服务。"""

import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Qwen3VL-main/Qwen3VL-main"))
from app.vln.inference_pipeline import InferencePipeline, InferenceRequest
from app.vln.prompts import get_system_prompt


class DiagnosticCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.traces = []

    def emit(self, record):
        message = record.getMessage()
        if message.startswith("VLN_DIAGNOSTIC "):
            self.traces.append(json.loads(message.split(" ", 1)[1]))


class RecordedEngine:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_from_messages(self, **kwargs):
        self.calls.append(kwargs)
        return self.response.copy()


async def main():
    baseline_hash = hashlib.sha256(get_system_prompt("baseline_v1").encode()).hexdigest()
    assert baseline_hash == "5568b46e82d4fef204afa2c0160333f97a381f4749deabd0ba520f56e930c70c"
    assert get_system_prompt("grounded_stop_v2") != get_system_prompt("baseline_v1")
    print("PASS: original baseline system prompt unchanged")
    fixture = json.loads((ROOT / "reports/repeated_measurements_20261019.json").read_text(encoding="utf-8"))
    examples = [next(row for row in fixture["measured_cases"] if row["nonzero_raw_waypoints"]),
                next(row for row in fixture["measured_cases"] if row["original_json"]["reached_goal"])]
    logger = logging.getLogger("app.vln.inference_pipeline")
    capture = DiagnosticCapture()
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    pipeline = InferencePipeline()
    tasks = []
    payloads = {}
    try:
        for profile in ("baseline_v1", "grounded_stop_v2"):
            for index, example in enumerate(examples):
                task = pipeline._task_manager.create_task(fixture["instruction"])
                tasks.append(task)
                engine = RecordedEngine(example["model_response"])
                with patch.dict("os.environ", {"PROJECT2_NAV_PROMPT_PROFILE": profile, "PROJECT2_DIAGNOSTICS": "1"}):
                    result = await pipeline._process_request(
                        InferenceRequest(task.task_id, 1, 1.0, "data:image/png;base64,Zml4dHVyZQ=="), engine)
                assert not result.error, result.error
                assert len(engine.calls) == 1
                call = engine.calls[0]
                assert call["max_tokens"] == 512 and call["temperature"] == 0.3
                assert call["messages"][0] == {"role": "system", "content": get_system_prompt(profile)}
                trace = capture.traces[-1]
                assert trace["prompt_profile"] == profile
                assert trace["system_prompt_sha256"] == hashlib.sha256(call["messages"][0]["content"].encode()).hexdigest()
                assert trace["model_response"] == example["model_response"]
                # v2不能在后处理里强制制造“模型学会停止”的假象。
                assert result.reached_goal == example["original_json"]["reached_goal"]
                assert [dict(dx=w.dx, dy=w.dy, dtheta=w.dtheta) for w in result.waypoints] == example["original_json"]["waypoints"]
                payloads[(profile, index)] = call["messages"][1:]
        for index in range(2):
            assert payloads[("baseline_v1", index)] == payloads[("grounded_stop_v2", index)]
        print("PASS: only system prompt differs; image, user instruction and generation settings unchanged")
        print("PASS: diagnostics hash actual sent system prompt; recorded model output is not overwritten")
        engine = RecordedEngine(examples[0]["model_response"])
        with patch.dict("os.environ", {"PROJECT2_NAV_PROMPT_PROFILE": "typo", "PROJECT2_DIAGNOSTICS": "1"}):
            result = await pipeline._process_request(InferenceRequest(tasks[0].task_id, 1, 1.0, "fixture"), engine)
        assert result.error and not engine.calls and result.waypoints == [] and not result.reached_goal
        print("PASS: unknown profile fails before calling engine")
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)
        for task in tasks:
            pipeline._task_manager.delete_task(task.task_id)
    print("PROMPT_PROFILES_CHECKED: offline fixtures only; live v2 quality remains unmeasured.")


if __name__ == "__main__":
    asyncio.run(main())
