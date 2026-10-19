"""用模拟模型响应核查诊断记录；不访问模型、不运行服务。"""

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "Qwen3VL-main/Qwen3VL-main"))
from app.engine import LMDeployEngine
from app.vln.inference_pipeline import InferencePipeline, InferenceRequest
from app.vln import inference_pipeline as module

raw_text = json.dumps({"environment": "桌面", "action": "测试动作", "waypoints": [
    {"dx": 0.2, "dy": 0.0, "dtheta": 0.0}], "progress": 0.0, "reached_goal": False})
backend = Mock()
backend.json.return_value = {"choices": [{"message": {"content": raw_text}, "finish_reason": "length"}],
                             "usage": {"prompt_tokens": 20, "completion_tokens": 30}}
engine = LMDeployEngine.__new__(LMDeployEngine)
engine._resolved_model = "mock"
engine.model_name = "mock"
engine.base_url = "http://unused.invalid"
engine.timeout_s = 1
with patch("app.engine.requests.post", return_value=backend):
    response = engine.generate_from_messages([{"role": "user", "content": "test"}])
assert response["finish_reason"] == "length" and response["text"] == raw_text
print("PASS: actual backend finish_reason is preserved")


async def check():
    pipeline = InferencePipeline()
    task = pipeline._task_manager.create_task("诊断测试")
    request = InferenceRequest(task.task_id, 1, time.time(), base64.b64encode(b"test-image").decode())
    fake = Mock()
    fake.generate_from_messages.return_value = response
    try:
        with patch.dict(os.environ, {"PROJECT2_DIAGNOSTICS": "1"}):
            with patch.object(module.logger, "info") as log:
                result = await pipeline._process_request(request, fake)
        assert not result.error and result.reached_goal is False and result.waypoints[0].dx == 0.2
        diagnostic = json.loads(log.call_args.args[1])
        assert diagnostic["model_response"]["text"] == raw_text
        assert diagnostic["model_response"]["finish_reason"] == "length"
        assert diagnostic["parsed"]["reached_goal"] is False
        assert diagnostic["derived_velocity"]["linear_vel"] == 1.0
        assert diagnostic["frame_id"] == 1 and "image_base64" not in diagnostic
        print("PASS: one record links raw output, termination reason, parsed waypoints and speed")
        fake.generate_from_messages.side_effect = RuntimeError("simulated backend failure")
        with patch.dict(os.environ, {"PROJECT2_DIAGNOSTICS": "1"}):
            with patch.object(module.logger, "info") as log:
                result = await pipeline._process_request(request, fake)
        assert result.error == "simulated backend failure"
        assert json.loads(log.call_args.args[1])["error"] == result.error
        print("PASS: model error remains visible in diagnostic record")
        fake.generate_from_messages.side_effect = None
        with patch.dict(os.environ, {"PROJECT2_DIAGNOSTICS": "0"}):
            with patch.object(module.logger, "info") as log:
                result = await pipeline._process_request(request, fake)
        assert not result.error and not log.called
        print("PASS: diagnostics can be disabled")
    finally:
        pipeline._task_manager.delete_task(task.task_id)


asyncio.run(check())
print("PASS: mocked checks only; real video diagnostic capture is pending.")
