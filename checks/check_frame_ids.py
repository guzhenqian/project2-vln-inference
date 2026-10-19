"""项目2：检查排队、丢帧和重置采样器时的请求编号；不调用模型。"""

import asyncio
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "Qwen3VL-main/Qwen3VL-main"))
from app.vln.inference_pipeline import InferencePipeline


async def main():
    # 读取真实PNG进行原采样器检查；不启动推理worker。
    image = base64.b64encode((root / "data/tum_rgbd/sample_rgb.png").read_bytes()).decode("ascii")
    pipeline = InferencePipeline()
    pipeline._running = True
    tasks = []
    checks = []

    def new_task():
        task = pipeline._task_manager.create_task("帧编号检查，不运行模型")
        tasks.append(task)
        return task

    def passed(name, **evidence):
        checks.append({"check": name, "result": "passed", **evidence})
        print("PASS:", name, evidence, flush=True)

    try:
        task = new_task()
        start = time.time()
        submitted = await asyncio.gather(*[
            pipeline.submit_frame(task.task_id, image, timestamp=start + 2.1 * i)
            for i in range(3)
        ])
        assert all(submitted), "Expected three accepted requests"
        queued = [pipeline._queue.get_nowait() for _ in range(3)]
        ids = [request.frame_id for request in queued]
        assert ids == [1, 2, 3], f"Duplicate or unexpected IDs while no inference has completed: {ids}"
        assert task.frame_count == 0, "Number allocation must not mark inference as completed"
        passed("concurrent submissions have unique IDs before completion", ids=ids, completed=task.frame_count)

        # 过近的同图被采样器跳过，不分配请求编号。
        accepted = await pipeline.submit_frame(task.task_id, image, timestamp=start + 4.21)
        assert not accepted
        assert await pipeline.submit_frame(task.task_id, image, timestamp=start + 6.3)
        assert pipeline._queue.get_nowait().frame_id == 4
        passed("sampling skip does not allocate a request ID")

        # 清理缓存/采样器后，相同任务仍不能复用已分配的编号。
        pipeline.remove_task(task.task_id)
        assert await pipeline.submit_frame(task.task_id, image, timestamp=start + 8.4)
        assert pipeline._queue.get_nowait().frame_id == 5
        passed("sampler/cache reset does not reuse task IDs")

        other = new_task()
        assert await pipeline.submit_frame(other.task_id, image, timestamp=start)
        assert pipeline._queue.get_nowait().frame_id == 1
        passed("each new task starts at ID 1")

        # 队列容量1时，第二个请求替换旧请求，编号仍增加。
        pipeline._queue = asyncio.Queue(maxsize=1)
        assert await pipeline.submit_frame(other.task_id, image, timestamp=start + 2.1)
        assert await pipeline.submit_frame(other.task_id, image, timestamp=start + 4.2)
        assert pipeline._queue.get_nowait().frame_id == 3
        assert pipeline.stats["dropped"] == 1
        passed("queue drop leaves a gap and never reuses an ID")

        # 提交超时消耗的编号也不复用；编号缺口不等于处理错误。
        with patch.object(pipeline._queue, "put", new=AsyncMock(side_effect=asyncio.TimeoutError)):
            assert not await pipeline.submit_frame(other.task_id, image, timestamp=start + 6.3)
        assert await pipeline.submit_frame(other.task_id, image, timestamp=start + 8.4)
        assert pipeline._queue.get_nowait().frame_id == 5
        passed("submission timeout never causes ID reuse")
    finally:
        pipeline._running = False
        for task in tasks:
            pipeline.remove_task(task.task_id)
            pipeline._task_manager.delete_task(task.task_id)

    started = datetime.now(timezone.utc)
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"frame_ids_check_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    log_path.write_text(json.dumps({
        "purpose": "request_id_code_check_not_model_or_performance_test",
        "checked_at_utc": started.isoformat(), "checks": checks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved check record:", log_path)
    print("PASS: request IDs are unique per task; no model inference was executed.")


if __name__ == "__main__":
    asyncio.run(main())
