"""项目2：一次预热和固定五图的串行测量；支持同图重复，不测视频吞吐。"""

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


def summarize_rows(rows):
    summary = {"measured_count": len(rows)}
    for field in ["client_elapsed_s", "service_processing_s", "completion_tokens"]:
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))
                  and not isinstance(row[field], bool) and math.isfinite(row[field])]
        summary[field] = {"count": len(values)}
        if values:
            summary[field].update({"median": statistics.median(values), "min": min(values),
                                   "max": max(values),
                                   "sample_stdev": statistics.stdev(values) if len(values) > 1 else None})
    summary["normal_finish_count"] = sum(row.get("finish_reason") == "stop" for row in rows)
    summary["valid_original_json_count"] = sum(row.get("original_json_valid", False) for row in rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:20000")
    parser.add_argument("--repeats", type=int, default=1, help="Rounds of all five images after one warm-up (1-100).")
    parser.add_argument("--prompt-profile", choices=["baseline_v1", "grounded_stop_v2"], default="baseline_v1",
                        help="Expected business prompt version; verify against actual diagnostics.")
    parser.add_argument("--check-only", action="store_true", help="Validate local inputs only; no HTTP requests.")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error("--repeats must be between 1 and 100")
    expected_requests = 1 + 5 * args.repeats
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
    print(f"Scope: one warm-up + {5 * args.repeats} serial, independent single-frame navigation requests "
          f"({args.repeats} rounds of five images).", flush=True)
    print("Expected prompt profile:", args.prompt_profile, flush=True)
    if args.check_only:
        print("CHECK_OK: local inputs are ready; no HTTP request or model inference was performed.")
        return

    started = datetime.now(timezone.utc)
    prefix = "initial_latency" if args.repeats == 1 else "repeated_latency"
    if args.prompt_profile != "baseline_v1":
        prefix += "_" + args.prompt_profile
    output = root / "logs" / f"{prefix}_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output.parent.mkdir(exist_ok=True)
    record = {
        "purpose": "fixed_five_image_serial_latency_not_stable_benchmark_or_video_hz",
        "measurement_status": "running",
        "started_at_utc": started.isoformat(), "base_url": args.base_url,
        "manifest": manifest, "manifest_sha256": sha256(manifest_path),
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "client_time_definition": "synchronous frame HTTP round trip; excludes image encoding, task creation and task stop",
        "service_time_definition": "original inference_time: prompt/model HTTP call/parsing; excludes queue waiting and is not pure GPU time",
        "configuration": {"concurrency": 1, "fresh_task_per_request": True, "visualization": False,
                          "repeats_per_image": args.repeats, "planned_measured_count": 5 * args.repeats,
                          "expected_prompt_profile": args.prompt_profile,
                          "order": "one warm-up on first image, then manifest order in each round",
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

    def save_record():
        # 检查点写入使用替换，避免中断时破坏此前已保存的完整JSON。
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    def read_traces(target_ids):
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
                        continue  # tee可能正在写最后一行。
                    if trace.get("task_id") in target_ids:
                        traces.setdefault(trace["task_id"], []).append({"log": str(log), "line": line_number, "trace": trace})
            if len(traces) == len(target_ids) and all(len(items) == 1 for items in traces.values()):
                return traces
            time.sleep(0.1)
        return traces

    def check_prompt_trace(trace):
        # 旧基线日志没有版本；v2必须有显式版本及有效的实际系统提示词哈希。
        actual_profile = trace.get("prompt_profile", "baseline_v1")
        if actual_profile != args.prompt_profile:
            raise ValueError(f"Unexpected business prompt profile: {actual_profile}; expected {args.prompt_profile}")
        prompt_hash = trace.get("system_prompt_sha256")
        if args.prompt_profile != "baseline_v1" and not prompt_hash:
            raise ValueError("Missing actual system prompt hash for non-baseline measurement")
        if prompt_hash and (not isinstance(prompt_hash, str) or len(prompt_hash) != 64
                            or any(c not in "0123456789abcdef" for c in prompt_hash)):
            raise ValueError("Invalid actual system prompt hash")
        return actual_profile, prompt_hash

    def run_case(case, warmup, repeat_index=0):
        row = {"case_id": case["case_id"], "warmup": warmup,
               "repeat_index": repeat_index, "started_at_utc": datetime.now(timezone.utc).isoformat(),
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
            print(f"{label} round={repeat_index} {case['case_id']}: "
                  f"client={elapsed:.3f}s, service={duration:.3f}s", flush=True)
        except Exception as error:
            row["error"] = str(error)
            raise
        finally:
            if task_id:
                try:
                    http("POST", f"/vln/task/{task_id}/stop", {})
                except Exception as error:
                    row["cleanup_error"] = str(error)
            save_record()
        if row.get("cleanup_error"):
            raise RuntimeError("Task cleanup failed; stopping to avoid interference with later requests")

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
        # 先用预热的真实诊断确认服务版本，避免测完50次才发现切错提示词。
        warmup_row = record["attempts"][0]
        warmup_matches = read_traces({warmup_row["task_id"]}).get(warmup_row["task_id"], [])
        warmup_row["diagnostic_matches"] = warmup_matches
        if len(warmup_matches) != 1:
            raise ValueError("Expected one warm-up diagnostic before measured requests")
        check_prompt_trace(warmup_matches[0]["trace"])
        for repeat_index in range(1, args.repeats + 1):
            for case in cases:
                run_case(case, warmup=False, repeat_index=repeat_index)
        after, _ = http("GET", "/vln/pipeline/stats")
        record["stats_after"] = after
        delta = {key: after[key] - before[key] for key in ["total_requests", "processed", "dropped", "errors"]}
        record["stats_delta"] = delta
        if delta != {"total_requests": expected_requests, "processed": expected_requests, "dropped": 0, "errors": 0}:
            raise ValueError(f"Unexpected pipeline counts/interference: {delta}")

        # 每个新任务只有一张图；匹配服务诊断核实真实输入、原始输出和终止原因。
        target_ids = {row["task_id"] for row in record["attempts"]}
        traces = read_traces(target_ids)
        prompt_hashes = set()
        for row in record["attempts"]:
            matches = traces.get(row["task_id"], [])
            row["diagnostic_matches"] = matches
            if len(matches) != 1:
                raise ValueError("Expected one diagnostic per fresh task; check PROJECT2_DIAGNOSTICS")
            trace = matches[0]["trace"]
            actual_profile, prompt_hash = check_prompt_trace(trace)
            if prompt_hash:
                prompt_hashes.add(prompt_hash)
            row["prompt_profile"] = actual_profile
            row["system_prompt_sha256"] = prompt_hash
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
                row["original_json"] = raw_json
            except ValueError:
                row["original_json_valid"] = False

        if len(prompt_hashes) > 1:
            raise ValueError("System prompt changed during this measurement")
        record["system_prompt_sha256"] = next(iter(prompt_hashes), None)
        measured = [row for row in record["attempts"] if not row["warmup"]]
        record["summary"] = summarize_rows(measured)
        record["summary"]["warmup_count"] = 1
        record["per_case_summary"] = {
            case["case_id"]: summarize_rows([row for row in measured if row["case_id"] == case["case_id"]])
            for case in cases
        }
        record["per_round_summary"] = {
            str(index): summarize_rows([row for row in measured if row["repeat_index"] == index])
            for index in range(1, args.repeats + 1)
        }
        record["quality_review_rows"] = [{
            "case_id": row["case_id"], "repeat_index": row["repeat_index"], "task_id": row["task_id"],
            "client_elapsed_s": row["client_elapsed_s"], "completion_tokens": row["completion_tokens"],
            "finish_reason": row["finish_reason"], "original_json_valid": row["original_json_valid"],
            "raw_model_text": row["diagnostic_matches"][0]["trace"]["model_response"]["text"],
            "original_json": row.get("original_json"),
            "parsed": row["diagnostic_matches"][0]["trace"].get("parsed"),
            "derived_velocity": row["diagnostic_matches"][0]["trace"].get("derived_velocity"),
            "human_quality_verdict": "pending",
        } for row in measured]
        record["measurement_status"] = "complete"
        print("Summary (warm-up excluded):\n" + json.dumps(record["summary"], indent=2), flush=True)
        if args.repeats > 1:
            print("Per-image summary:\n" + json.dumps(record["per_case_summary"], indent=2), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        problem = str(error) or "interrupted"
        record["error"] = problem
        record["measurement_status"] = "incomplete"
        print("Error:", problem, flush=True)
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_record()
        print("Saved measurement:", output, flush=True)
    if problem:
        raise SystemExit("CHECK_INCOMPLETE: share this output; partial data was preserved.")
    print(f"LATENCY_RECORDED: {5 * args.repeats} serial single-frame samples; "
          "quality review and broader measurement are pending.")
    print("No video Hz, stable tail latency, GPU kernel time or peak memory result is claimed.")


if __name__ == "__main__":
    main()
