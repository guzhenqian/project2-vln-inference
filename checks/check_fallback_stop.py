"""离线检查无效模型结果与异步首帧保持停止，并回放100条真实历史输出。"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Qwen3VL-main/Qwen3VL-main"))
from app import vln_api
from app.vln import TaskStatus, Waypoint
from app.vln import inference_pipeline as module
from app.vln.inference_pipeline import InferencePipeline, InferenceRequest, InferenceResult
from app.vln.waypoint_parser import WaypointParser


def main():
    parser = WaypointParser()
    checks = []

    def passed(name, **evidence):
        checks.append(dict(check=name, result="passed", **evidence))
        print("PASS:", name, flush=True)

    invalid_texts = ["", "无法判断当前路径。", "尚未到达目标，请停止。", "任务尚未完成。",
                     "The goal has not been reached. Do not move forward.", "forward", "left", "到达目标",
                     '{"waypoints":[', "[]", "null", "123", '"forward"',
                     '{"reached_goal":"false","waypoints":[]}',
                     '{"reached_goal":"true","waypoints":[]}',
                     '{"reached_goal":1,"waypoints":[]}',
                     '{"reached_goal":false,"waypoints":"forward"}',
                     '{"reached_goal":false,"waypoints":[{"dx":0.3}]}',
                     '{"reached_goal":false,"waypoints":[{"dx":true,"dy":0,"dtheta":0}]}',
                     '{"reached_goal":false,"waypoints":[{"dx":NaN,"dy":0,"dtheta":0}]}',
                     '{"reached_goal":false,"waypoints":[{"dx":Infinity,"dy":0,"dtheta":0}]}',
                     '{"reached_goal":false,"waypoints":[{"dx":"0.3","dy":0,"dtheta":0}]}',
                     '{"reached_goal":true,"waypoints":[{"dx":0.3,"dy":0,"dtheta":0}]}',
                     '{"reached_goal":false,"waypoints":[],"progress":NaN}',
                     '{"reached_goal":false,"waypoints":[],"progress":2}',
                     '{"waypoints":[{"dx":0.3,"dy":0,"dtheta":0}]}']
    for text in invalid_texts:
        result = parser.parse(text)
        assert result["waypoints"] == [] and result["reached_goal"] is False, text
        assert parser.waypoints_to_velocity(result["waypoints"]) == (0.0, 0.0)
        assert result["action"] == "停止" and result["parse_status"] == "fallback_stop"
        assert result["parse_error"] and result["raw_output"] == text
    passed("invalid text/JSON stops without declaring arrival; raw answer preserved", cases=len(invalid_texts))

    fixtures = ["reports/repeated_measurements_20261019.json", "reports/grounded_stop_v2_measurements_20261019.json"]
    count = 0
    for filename in fixtures:
        record = json.loads((ROOT / filename).read_text(encoding="utf-8"))
        for row in record["measured_cases"]:
            text = row["model_response"]["text"]
            actual = parser.parse(text)
            expected = row["parsed"]
            assert actual["parse_status"] == "valid_json", (filename, row["case_id"], row["repeat_index"])
            for field in ("environment", "action", "reasoning", "progress", "reached_goal"):
                assert actual[field] == expected[field]
            assert [wp.to_dict() for wp in actual["waypoints"]] == expected["waypoints"]
            assert actual["raw_output"] == text
            count += 1
    assert count == 100
    passed("100 historical valid JSON answers retain their original parsed fields", cases=count)
    true_result = parser.parse('{"reached_goal":true,"waypoints":[]}')
    move_result = parser.parse('{"reached_goal":false,"waypoints":[{"dx":0.2,"dy":0,"dtheta":0}]}')
    assert true_result["parse_status"] == "valid_json" and true_result["reached_goal"] is True
    assert move_result["parse_status"] == "valid_json" and parser.waypoints_to_velocity(move_result["waypoints"]) == (1.0, 0.0)
    passed("valid structured movement and true boolean remain supported")

    async def check_pipeline_and_api():
        pipeline = InferencePipeline()
        manager = pipeline._task_manager
        task = manager.create_task("离线检查：无效结果必须停止")
        manager.update_task_status(task.task_id, TaskStatus.RUNNING)
        engine = Mock()
        engine.generate_from_messages.return_value = dict(text="尚未到达目标，do not move forward", finish_reason="stop")
        try:
            with patch.dict("os.environ", {"PROJECT2_DIAGNOSTICS": "1", "PROJECT2_NAV_PROMPT_PROFILE": "grounded_stop_v2"}):
                with patch.object(module.logger, "info") as log:
                    result = await pipeline._process_request(InferenceRequest(task.task_id, 1, time.time(), "fixture"), engine)
            trace = json.loads(log.call_args.args[1])
            assert not result.error and not result.reached_goal and result.waypoints == []
            assert trace["parsed"]["parse_status"] == "fallback_stop" and trace["parsed"]["parse_error"]
            assert trace["model_response"]["text"] == engine.generate_from_messages.return_value["text"]
            pipeline._update_task_with_result(InferenceRequest(task.task_id, 1, time.time(), "fixture"), result)
            assert task.status == TaskStatus.RUNNING
            passed("pipeline records fallback status and raw text; negated arrival does not complete task")

            for accepted in (True, False):
                for visualization in (True, False):
                    task.config.enable_visualization = visualization
                    fake = SimpleNamespace(submit_frame=AsyncMock(return_value=accepted), get_latest_result=lambda _: None)
                    visualizer = Mock()
                    visualizer.base64_to_frame.return_value = object()
                    visualizer.frame_to_base64.return_value = "fixture-rendered"
                    with patch.object(vln_api, "get_inference_pipeline", return_value=fake), patch.object(vln_api, "_visualizer", visualizer):
                        response = await vln_api.process_frame(vln_api.FrameRequest(task_id=task.task_id, frame="fixture"))
                    assert response.waypoints == [] and response.linear_vel == 0.0 and response.angular_vel == 0.0
                    assert response.reached_goal is False and response.inference_time == 0.0
                    assert "保持停止" in response.action
                    if visualization:
                        kwargs = visualizer.render_frame.call_args.kwargs
                        assert kwargs["waypoints"] == [] and kwargs["linear_vel"] == 0.0 and kwargs["angular_vel"] == 0.0
            passed("async no-result response and overlay stop for accepted/skipped frames", cases=4)

            task.config.enable_visualization = False
            cached = InferenceResult(task.task_id, 1, time.time(), [Waypoint(dx=0.2, dy=0, dtheta=0)], "fixture", "前进", "fixture", 0.0, False, 0.1)
            fake = SimpleNamespace(submit_frame=AsyncMock(return_value=False), get_latest_result=lambda _: cached)
            with patch.object(vln_api, "get_inference_pipeline", return_value=fake):
                response = await vln_api.process_frame(vln_api.FrameRequest(task_id=task.task_id, frame="fixture"))
            assert response.linear_vel == 1.0 and response.waypoints[0].dx == 0.2
            passed("existing valid cached result is not replaced with a synthetic stop")
        finally:
            manager.delete_task(task.task_id)

    asyncio.run(check_pipeline_and_api())
    started = datetime.now(timezone.utc)
    output = ROOT / "logs" / f"fallback_stop_check_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(purpose="offline_regression_not_new_model_or_navigation_evaluation",
                                     checked_at_utc=started.isoformat(), checks=checks), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved check record:", output)
    print("FALLBACK_STOP_CHECKED: offline replay and mocks only; running service not restarted.")


if __name__ == "__main__":
    main()
