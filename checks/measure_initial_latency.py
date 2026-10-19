"""项目2：一次预热和五次独立同步导航请求；记录初步耗时，不测视频吞吐。"""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:20000")
    parser.add_argument("--check-only", action="store_true", help="Validate local inputs only; no HTTP requests.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/baseline/v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if len(cases) != 5 or [case["video_frame_index"] for case in cases] != [0, 30, 60, 90, 120]:
        raise ValueError("Expected the five fixed v1 cases")
    for case in cases:
        if sha256(root / case["image_path"]) != case["image_sha256"]:
            raise ValueError("Input image changed: " + case["case_id"])
    print("Fixed inputs verified:", len(cases), flush=True)
    print("Scope: one warm-up + five serial, independent single-frame navigation requests.", flush=True)
    if args.check_only:
        print("CHECK_OK: local inputs are ready; no HTTP request or model inference was performed.")
        return

    started = datetime.now(timezone.utc)
    output = root / "logs" / f"initial_latency_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output.parent.mkdir(exist_ok=True)
    record = {
        "purpose": "initial_latency_measurement_five_cases_not_stable_benchmark_or_video_hz",
        "started_at_utc": started.isoformat(), "base_url": args.base_url,
        "manifest": manifest, "manifest_sha256": sha256(manifest_path),
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "client_time_definition": "synchronous frame HTTP round trip; excludes image encoding, task creation and task stop",
        "service_time_definition": "original inference_time: prompt/model HTTP call/parsing; excludes queue waiting and is not pure GPU time",
        "configuration": {"concurrency": 1, "fresh_task_per_request": True, "visualization": False,
                          "generation": manifest["existing_navigation_generation"], "random_seed": None,
                          "diagnostic_logging": "expected enabled; checked against actual traces"},
        "quality_review": "pending human review of every original output; structural checks do not establish navigation quality",
        "source_sha256": {}, "calls": [], "attempts": [],
    }
    for relative in ["app/engine.py", "app/vln/__init__.py", "app/vln/inference_pipeline.py",
                     "app/vln/prompts.py", "app/vln/waypoint_parser.py"]:
        record["source_sha256"][relative] = sha256(root / "Qwen3VL-main/Qwen3VL-main" / relative)
    record["packages"] = {}
    for name in ["torch", "lmdeploy", "transformers", "fastapi"]:
        try:
            record["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record["packages"][name] = None

    def http(method, path, payload=None, timeout=15):
        # 序列化在计时之前完成；完整响应读完时停止计时。
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(args.base_url.rstrip("/") + path, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        entry = {"method": method, "path": path}
        record["calls"].append(entry)
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
                elapsed = time.perf_counter() - start
                entry.update({"http_status": response.status, "client_elapsed_s": elapsed,
                              "response": json.loads(raw.decode("utf-8"))})
            return entry["response"], elapsed
        except HTTPError as error:
            entry.update({"http_status": error.code, "error_body": error.read().decode("utf-8", errors="replace")})
            raise RuntimeError(f"{path}: HTTP {error.code}: {entry['error_body']}") from error
        except Exception as error:
            entry["error"] = str(error)
            raise

    def run_case(case, warmup):
        row = {"case_id": case["case_id"], "warmup": warmup,
               "image_sha256": case["image_sha256"], "structural_check": False}
        record["attempts"].append(row)
        image_url = "data:image/png;base64," + base64.b64encode((root / case["image_path"]).read_bytes()).decode("ascii")
        row["submitted_image_base64_sha256"] = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
        task_id = None
        try:
            task, _ = http("POST", "/vln/task/create", {
                "instruction": manifest["instruction"], "config": {"enable_visualization": False},
            })
            task_id = task["task_id"]
            row["task_id"] = task_id
            result, elapsed = http("POST", "/vln/frame/sync", {"task_id": task_id, "frame": image_url}, timeout=90)
            row.update({"client_elapsed_s": elapsed, "response": result})
            status, _ = http("GET", f"/vln/task/{task_id}")
            row["task_status"] = status
            if result.get("task_id") != task_id or result.get("frame_id") != 1 or status.get("frame_count") != 1:
                raise ValueError("Fresh task did not return exactly its first processed frame")
            if not result.get("environment") or not result.get("action") or status.get("error_message"):
                raise ValueError("Empty output or business task error")
            duration = result.get("inference_time")
            if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid service processing time")
            row["service_processing_s"] = duration
            row["structural_check"] = True
            label = "WARMUP" if warmup else "MEASURE"
            print(f"{label} {case['case_id']}: client={elapsed:.3f}s, service={duration:.3f}s", flush=True)
        except Exception as error:
            row["error"] = str(error)
            raise
        finally:
            if task_id:
                try:
                    http("POST", f"/vln/task/{task_id}/stop", {})
                except Exception as error:
                    row["cleanup_error"] = str(error)

    problem = None
    try:
        health, _ = http("GET", "/health")
        if health.get("status") != "healthy" or health.get("model") != "Qwen3-VL-2B-Instruct":
            raise ValueError("Unexpected business health/model")
        before, _ = http("GET", "/vln/pipeline/stats")
        record["stats_before"] = before
        if before["queue_size"] != 0:
            raise ValueError("Pipeline is busy; stop page playback and let queued requests finish")
        run_case(cases[0], warmup=True)
        for case in cases:
            run_case(case, warmup=False)
        after, _ = http("GET", "/vln/pipeline/stats")
        record["stats_after"] = after
        delta = {key: after[key] - before[key] for key in ["total_requests", "processed", "dropped", "errors"]}
        record["stats_delta"] = delta
        if delta != {"total_requests": 6, "processed": 6, "dropped": 0, "errors": 0}:
            raise ValueError(f"Unexpected pipeline counts/interference: {delta}")

        # 每个新任务只有一张图；匹配服务诊断核实真实输入、原始输出和终止原因。
        target_ids = {row["task_id"] for row in record["attempts"]}
        traces = {}
        for retry in range(4):
            traces = {}
            for log in sorted((root / "logs").glob("business_api_*.log")):
                for line_number, line in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if "VLN_DIAGNOSTIC " not in line:
                        continue
                    try:
                        trace = json.loads(line.split("VLN_DIAGNOSTIC ", 1)[1])
                    except ValueError:
                        # tee可能正在写最后一行，下一轮再读取。
                        continue
                    if trace.get("task_id") in target_ids:
                        traces.setdefault(trace["task_id"], []).append({"log": str(log), "line": line_number, "trace": trace})
            if len(traces) == 6:
                break
            time.sleep(0.1)
        for row in record["attempts"]:
            matches = traces.get(row["task_id"], [])
            row["diagnostic_matches"] = matches
            if len(matches) != 1:
                raise ValueError("Expected one diagnostic per fresh task; check PROJECT2_DIAGNOSTICS")
            trace = matches[0]["trace"]
            if trace.get("error") or trace.get("frame_id") != 1 or trace.get("image_base64_sha256") != row["submitted_image_base64_sha256"]:
                raise ValueError("Diagnostic input/result identity or inference error")
            if trace.get("generation") != manifest["existing_navigation_generation"]:
                raise ValueError("Actual generation settings differ from fixed manifest")
            model_response = trace["model_response"]
            row["finish_reason"] = model_response.get("finish_reason")
            row["completion_tokens"] = model_response.get("completion_tokens")
            text = model_response["text"].strip()
            if text.startswith("```"):
                text = "\n".join(text.splitlines()[1:-1])
            try:
                raw_json = json.loads(text)
                row["original_json_valid"] = isinstance(raw_json, dict)
            except ValueError:
                row["original_json_valid"] = False

        measured = [row for row in record["attempts"] if not row["warmup"]]
        record["summary"] = {"measured_count": len(measured), "warmup_count": 1}
        for field in ["client_elapsed_s", "service_processing_s"]:
            values = [row[field] for row in measured]
            record["summary"][field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        record["summary"]["normal_finish_count"] = sum(row["finish_reason"] == "stop" for row in measured)
        record["summary"]["valid_original_json_count"] = sum(row["original_json_valid"] for row in measured)
        print("Summary (warm-up excluded):\n" + json.dumps(record["summary"], indent=2), flush=True)
    except Exception as error:
        problem = str(error)
        record["error"] = problem
        print("Error:", problem, flush=True)
    finally:
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved measurement:", output, flush=True)
    if problem:
        raise SystemExit("CHECK_INCOMPLETE: share this output; partial data was preserved.")
    print("LATENCY_RECORDED: five serial single-frame samples; quality review and broader measurement are pending.")
    print("No video Hz, stable tail latency, GPU kernel time or peak memory result is claimed.")


if __name__ == "__main__":
    main()
