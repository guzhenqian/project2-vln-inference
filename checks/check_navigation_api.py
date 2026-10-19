"""项目2：用新任务和单帧检查老师的同步导航接口；不是导航质量评测。"""

import argparse
import base64
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Check one fresh VLN task with one image.")
    parser.add_argument("--base-url", default="http://127.0.0.1:20000")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image_path = root / "data/tum_rgbd/sample_rgb.png"
    image_bytes = image_path.read_bytes()
    started = datetime.now(timezone.utc)
    record = {
        "purpose": "single_frame_navigation_functional_check_not_quality_or_performance_benchmark",
        "started_at_utc": started.isoformat(),
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "calls": [],
        "quality_review": "pending; parsed fields alone do not prove valid navigation",
    }
    task_id = None
    problem = None

    def call(method, path, payload=None, timeout=15):
        entry = {"method": method, "url": args.base_url.rstrip("/") + path, "request": payload}
        record["calls"].append(entry)
        request = Request(
            entry["url"], method=method,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:
                entry["http_status"] = response.status
                entry["response"] = json.loads(response.read().decode("utf-8"))
            print(method, path, "HTTP", entry["http_status"], flush=True)
            return entry["response"]
        except HTTPError as error:
            entry["http_status"] = error.code
            entry["error_body"] = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{path}: HTTP {error.code}: {entry['error_body']}") from error
        except Exception as error:
            entry["error"] = str(error)
            raise
        finally:
            entry["client_elapsed_s"] = time.perf_counter() - start

    try:
        print("API:", args.base_url, flush=True)
        print("Image:", image_path, flush=True)
        # 新建任务且只发一帧，避免把其他帧的缓存结果算成本次结果。
        created = call("POST", "/vln/task/create", {
            "instruction": "观察当前桌面场景，寻找键盘。如果画面不能确认机器人可通行的地面路径，请保持停止并说明原因。",
            "config": {"enable_visualization": False},
        })
        task_id = created["task_id"]
        print("Task ID:", task_id, flush=True)
        print("Waiting for the first frame result (server timeout: 60 seconds)...", flush=True)
        result = call("POST", "/vln/frame/sync", {
            "task_id": task_id,
            "frame": "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii"),
        }, timeout=90)
        print("Navigation response:\n" + json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        status = call("GET", f"/vln/task/{task_id}")
        if result.get("task_id") != task_id or result.get("frame_id") != 1 or status.get("frame_count") != 1:
            raise ValueError("Task/frame identity mismatch; share the saved log.")
        if status.get("error_message"):
            raise ValueError("Task reports an error: " + status["error_message"])
        for field in ("environment", "action"):
            if not isinstance(result.get(field), str) or not result[field].strip():
                raise ValueError(f"Missing {field}; HTTP 200 alone does not establish inference success.")
        for field in ("inference_time", "linear_vel", "angular_vel", "progress"):
            value = result.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric field: {field}")
        if result["inference_time"] <= 0:
            raise ValueError("No positive processing time returned.")
        if not isinstance(result.get("waypoints"), list):
            raise ValueError("Invalid waypoints list.")
        for waypoint in result["waypoints"]:
            for field in ("dx", "dy", "dtheta"):
                value = waypoint.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Invalid waypoint field: {field}")
        record["structural_check"] = "passed; visual and navigation correctness still require review"
    except Exception as error:
        problem = str(error)
        record["error"] = problem
        print("Error:", problem, flush=True)
    finally:
        # 只停止本脚本创建的任务，保留任务状态及日志用于排查。
        if task_id:
            try:
                call("POST", f"/vln/task/{task_id}/stop", {})
            except Exception as error:
                record["cleanup_error"] = str(error)
                print("Task cleanup warning:", error)
        log_dir = root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"navigation_smoke_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        log_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved request and response:", log_path)
    if problem:
        raise SystemExit("CHECK_FAILED: share this output and the business service log.")
    print("STRUCTURE_OK: first-frame fields returned; manually review environment, action and waypoints.")
    print("Note: the original API omits raw model text and inference error; this check is not a quality benchmark.")


if __name__ == "__main__":
    main()
