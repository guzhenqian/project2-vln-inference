"""三张TUM地面/障碍图片的独立单帧诊断；不自动判定导航质量或测稳定性能。"""

import argparse
import base64
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/tum_rgbd/navigation_candidates"
PROFILE = "grounded_stop_v2"
PROMPT_SHA256 = "cc15062cc1e796901332ea2aee43d038918992d23175b5973aa5f2ca75b22d71"
GENERATION = {"max_tokens": 512, "temperature": 0.3}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(manifest=None):
    if manifest is not None:
        selection_path = Path(manifest).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        cases = selection["cases"]
        if not cases or len({c["case_id"] for c in cases}) != len(cases):
            raise ValueError("Empty or duplicate cases")
        root = selection_path.parent
        for case in cases:
            path = (root / case["image"]).resolve()
            if not path.is_relative_to(root) or digest(path) != case["sha256"]:
                raise ValueError("Image path/hash mismatch: " + case["case_id"])
        if not isinstance(selection.get("instruction"), str) or not selection["instruction"].strip():
            raise ValueError("Missing instruction")
        return selection, selection["sources"], {"selection_sha256": digest(selection_path)}
    selection_path = INPUTS / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_path = INPUTS / selection["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert [c["frame_index"] for c in selection["cases"]] == [153, 461, 1536], "Unexpected cases"
    frames = {f["frame_index"]: f for f in source["frames"]}
    for case in selection["cases"]:
        frame = frames[case["frame_index"]]
        path = INPUTS / case["image"]
        assert path.resolve() == (ROOT / frame["image_path"]).resolve(), "Image path differs"
        assert digest(path) == frame["sha256"], "Image hash differs: " + case["case_id"]
        assert (frame["width"], frame["height"]) == (640, 480), "Unexpected dimensions"
    return selection, source, {
        "selection_sha256": digest(selection_path),
        "source_manifest_sha256": digest(source_path),
    }


def read_diagnostic(task_id):
    for _ in range(10):
        matches = []
        for path in sorted((ROOT / "logs").glob("business_api_*.log")):
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if "VLN_DIAGNOSTIC " not in line:
                    continue
                try:
                    trace = json.loads(line.split("VLN_DIAGNOSTIC ", 1)[1])
                except ValueError:
                    continue
                if trace.get("task_id") == task_id:
                    matches.append({"log": path.name, "line": number, "trace": trace})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Duplicate diagnostics for fresh task")
        time.sleep(0.1)
    raise ValueError("Missing diagnostic; check business logging")


def validate_trace(trace, task_id, image_hash, instruction):
    if (trace.get("task_id") != task_id or trace.get("frame_id") != 1
            or trace.get("image_base64_sha256") != image_hash
            or trace.get("instruction") != instruction or trace.get("error")):
        raise ValueError("Diagnostic input identity or inference error")
    if trace.get("prompt_profile") != PROFILE or trace.get("system_prompt_sha256") != PROMPT_SHA256:
        raise ValueError("Business prompt differs from expected grounded_stop_v2")
    if trace.get("generation") != GENERATION:
        raise ValueError("Generation parameters differ from expected configuration")
    if trace.get("parsed", {}).get("parse_status") not in ("valid_json", "fallback_stop"):
        raise ValueError("Missing new parser status; check business service version")


def output_observation(trace):
    """描述原始输出，不将解析兜底或零速度计为模型正确停止。"""
    observed = {
        "raw_waypoint_state": "unknown",
        "raw_action": None, "raw_reached_goal": None, "raw_progress": None,
        "parse_status": trace["parsed"]["parse_status"],
        "finish_reason": trace["model_response"].get("finish_reason"),
    }
    text = trace["model_response"]["text"].strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            return observed
        observed.update(raw_action=raw.get("action"), raw_reached_goal=raw.get("reached_goal"),
                        raw_progress=raw.get("progress"))
        points = raw.get("waypoints")
        if not isinstance(points, list):
            return observed
        if not points:
            observed["raw_waypoint_state"] = "empty"
            return observed
        values = []
        for point in points:
            if not isinstance(point, dict):
                return observed
            for key in ("dx", "dy", "dtheta"):
                value = point.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    return observed
                values.append(value)
        observed["raw_waypoint_state"] = "nonzero" if any(v != 0 for v in values) else "all_zero"
    except (ValueError, TypeError):
        pass
    return observed


def summarize(attempts, cases):
    result = {}
    for case in cases:
        rows = [r["observation"] for r in attempts if r["case_id"] == case["case_id"] and "observation" in r]
        result[case["case_id"]] = {
            "recorded_count": len(rows),
            **{f"raw_waypoints_{kind}_count": sum(r["raw_waypoint_state"] == kind for r in rows)
               for kind in ("nonzero", "empty", "all_zero", "unknown")},
            "raw_action_stop_count": sum(r["raw_action"] == "停止" for r in rows),
            "raw_reached_goal_true_count": sum(r["raw_reached_goal"] is True for r in rows),
            "fallback_stop_count": sum(r["parse_status"] == "fallback_stop" for r in rows),
            "normal_finish_count": sum(r["finish_reason"] == "stop" for r in rows),
            "quality_verdict": "pending_manual_review",
        }
    return result


def main(argv=None):
    global INPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:20000")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=1, help="Rounds of selected images (1-20).")
    parser.add_argument("--manifest", type=Path, help="Use a separately versioned diagnostic image manifest.")
    parser.add_argument("--case-ids", nargs="+", help="Repeat only these case IDs; preserve manifest order.")
    parser.add_argument("--quiet", action="store_true", help="Keep raw answers in JSON, print concise progress.")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    selection, source, hashes = load_inputs(args.manifest)
    if args.manifest:
        INPUTS = args.manifest.resolve().parent
    if args.case_ids:
        wanted = set(args.case_ids)
        if not wanted.issubset({c["case_id"] for c in selection["cases"]}):
            parser.error("Unknown case ID")
        selection["cases"] = [c for c in selection["cases"] if c["case_id"] in wanted]
    expected_count = len(selection["cases"]) * args.repeats
    print(f"INPUTS_OK: {len(selection['cases'])} images verified. Profile:", PROFILE, flush=True)
    print("Instruction:", selection["instruction"], flush=True)
    print(f"Scope: {args.repeats} rounds, {expected_count} fresh tasks, no warmup.", flush=True)
    if args.check_only:
        print("CHECK_OK: no HTTP requests or model inference.")
        return

    started = datetime.now(timezone.utc)
    output = ROOT / "logs" / f"navigation_candidates_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output.parent.mkdir(exist_ok=True)
    record = {
        "status": "running", "started_at_utc": started.isoformat(),
        "purpose": "repeated independent single-frame qualitative diagnostics; no warmup; not a latency benchmark",
        "configuration": {"repeats_per_image": args.repeats, "planned_count": expected_count,
                          "order": [c["case_id"] for c in selection["cases"]], "concurrency": 1,
                          "fresh_task_per_request": True, "random_seed": None},
        "selection": selection, "source_manifest": source, **hashes,
        "expected_profile": PROFILE, "expected_prompt_sha256": PROMPT_SHA256,
        "expected_generation": GENERATION, "base_url": args.base_url,
        "script_sha256": digest(Path(__file__)),
        "timing": "sync HTTP round trip only; excludes task creation/cleanup; not pure GPU time",
        "calls": [], "attempts": [],
    }

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    def http(method, route, payload=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(args.base_url.rstrip("/") + route, data=data, method=method,
                          headers={"Content-Type": "application/json"})
        call = {"method": method, "route": route}
        record["calls"].append(call)
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
                call.update(http_status=response.status, response=result,
                            elapsed_s=time.perf_counter() - start)
            return result, call["elapsed_s"]
        except Exception as error:
            call["error"] = str(error)
            raise

    try:
        health, _ = http("GET", "/health")
        if health.get("status") != "healthy" or health.get("model") != "Qwen3-VL-2B-Instruct":
            raise ValueError("Unexpected health/model")
        before, _ = http("GET", "/vln/pipeline/stats")
        record["stats_before"] = before
        if before["queue_size"] != 0:
            raise ValueError("Pipeline busy; pause video playback first")
        schedule = [(repeat, case) for repeat in range(1, args.repeats + 1) for case in selection["cases"]]
        for repeat, case in schedule:
            row = {"case_id": case["case_id"], "repeat_index": repeat, "quality_verdict": "pending_manual_review"}
            record["attempts"].append(row)
            image = "data:image/png;base64," + base64.b64encode((INPUTS / case["image"]).read_bytes()).decode("ascii")
            image_hash = hashlib.sha256(image.encode("utf-8")).hexdigest()
            row["submitted_image_base64_sha256"] = image_hash
            task_id = None
            try:
                task, _ = http("POST", "/vln/task/create", {
                    "instruction": selection["instruction"], "config": {"enable_visualization": False}})
                task_id = row["task_id"] = task["task_id"]
                result, elapsed = http("POST", "/vln/frame/sync", {"task_id": task_id, "frame": image})
                row.update(response=result, client_elapsed_s=elapsed)
                status, _ = http("GET", f"/vln/task/{task_id}")
                row["task_status_before_cleanup"] = status
                if (result.get("task_id") != task_id or result.get("frame_id") != 1
                        or status.get("frame_count") != 1 or status.get("error_message")):
                    raise ValueError("Fresh-task response identity or task error")
                diagnostic = row["diagnostic"] = read_diagnostic(task_id)
                trace = diagnostic["trace"]
                validate_trace(trace, task_id, image_hash, selection["instruction"])
                parsed = trace["parsed"]
                observation = row["observation"] = output_observation(trace)
                print(f"\nround={repeat} {case['case_id']}: client={elapsed:.3f}s, "
                      f"parse_status={parsed['parse_status']}, raw_waypoints={observation['raw_waypoint_state']}, "
                      f"raw_action={observation['raw_action']}", flush=True)
                if args.repeats == 1 and not args.quiet:
                    print("RAW MODEL:", trace["model_response"]["text"], flush=True)
                    print("PARSED:", json.dumps(parsed, ensure_ascii=False), flush=True)
                    print("VELOCITY:", json.dumps(trace.get("derived_velocity"), ensure_ascii=False), flush=True)
            finally:
                if task_id:
                    try:
                        http("POST", f"/vln/task/{task_id}/stop", {})
                    except Exception as error:
                        row["cleanup_error"] = str(error)
                save()
            if row.get("cleanup_error"):
                raise ValueError("Task cleanup failed; stop further requests")
        after, _ = http("GET", "/vln/pipeline/stats")
        record["stats_after"] = after
        delta = {k: after[k] - before[k] for k in ("total_requests", "processed", "dropped", "errors")}
        record["stats_delta"] = delta
        if delta != {"total_requests": expected_count, "processed": expected_count, "dropped": 0, "errors": 0}:
            raise ValueError(f"Unexpected counts or concurrent requests: {delta}")
        record["per_case_summary"] = summarize(record["attempts"], selection["cases"])
        print("\nSUMMARY (counts only, not quality pass rates):\n"
              + json.dumps(record["per_case_summary"], ensure_ascii=False, indent=2), flush=True)
        record["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="incomplete", error=str(error) or "interrupted")
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print("\nSaved:", output, flush=True)
    if record["status"] != "complete":
        raise SystemExit("CHECK_INCOMPLETE: " + record["error"])
    print(f"DIAGNOSTIC_COMPLETE: {expected_count} requests recorded; navigation quality requires manual review.")


if __name__ == "__main__":
    main()
