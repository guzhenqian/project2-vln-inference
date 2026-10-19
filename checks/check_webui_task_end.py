"""以模拟响应核查页面任务结束处理，不访问服务、不运行模型。"""

import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import numpy as np

source = Path(__file__).resolve().parents[1] / "Qwen3VL-main/Qwen3VL-main/app/vln_webui.py"
spec = importlib.util.spec_from_file_location("teacher_vln_webui", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def response(code, body):
    obj = Mock()
    obj.status_code = code
    obj.ok = code < 400
    obj.json.return_value = body
    obj.text = str(body)
    if code >= 400:
        obj.raise_for_status.side_effect = module.requests.HTTPError(str(code))
    return obj


class FakeVideo:
    def __init__(self):
        self.read_count = 0

    def isOpened(self):
        return True

    def read(self):
        self.read_count += 1
        if self.read_count > 1:
            raise AssertionError("Video must stop after the terminal/error response")
        return True, np.zeros((16, 16, 3), dtype=np.uint8)


def run_case(name, frame_response, task_response, expected_text):
    controller = module.VLNWebUI()
    controller.current_task_id = "test_task"
    controller.video_cap = FakeVideo()
    with patch.object(module.requests, "post", return_value=frame_response) as post:
        with patch.object(module.requests, "get", return_value=task_response):
            outputs = list(controller.run_video())
    assert len(outputs) == 1 and post.call_count == 1
    assert not controller.is_running and controller._stop_event.is_set()
    assert expected_text in outputs[0][1], outputs[0][1]
    print("PASS:", name)


run_case("400 after task completion stops video", response(400, {"detail": "completed"}),
         response(200, {"status": "completed", "current_environment": "办公桌", "current_action": "停止"}),
         "业务层已标记结束")
run_case("200 reached_goal stops video", response(200, {"reached_goal": True}), None,
         "业务层已标记结束")
run_case("unexpected 400 stays visible and stops video", response(400, {"detail": "bad frame"}),
         response(200, {"status": "running"}), "bad frame")
run_case("500 stays visible and stops video", response(500, {"detail": "server failed"}), None,
         "server failed")

controller = module.VLNWebUI()
controller.current_task_id = "test_task"
controller.is_running = True
with patch.object(module.requests, "post", return_value=response(200, {"reached_goal": False})):
    _, result = controller.process_frame(np.zeros((16, 16, 3), dtype=np.uint8))
assert controller.is_running and "error" not in result and not controller._stop_event.is_set()
print("PASS: running task continues")
print("PASS: five mocked cases; browser retest is still required.")
