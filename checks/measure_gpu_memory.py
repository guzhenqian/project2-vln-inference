"""项目2：以200ms请求间隔采样GPU总显存，覆盖一次既有的预热+五图请求。"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class GPUMemorySampler:
    def __init__(self, executable):
        self.command = [executable, "-i", "0", "--query-gpu=index,uuid,name,memory.used,memory.total",
                        "--format=csv,noheader,nounits", "--loop-ms=200"]
        self.rows = []
        self.errors = []
        self.phase = "loaded_idle_before"
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.process = None
        self.thread = None

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase

    def start(self):
        self.process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=8):
            raise RuntimeError("No valid GPU memory sample arrived; share this output")

    def _read(self):
        for line in self.process.stdout:
            try:
                values = [value.strip() for value in next(csv.reader([line]))]
                if len(values) != 5:
                    raise ValueError(f"Unexpected NVIDIA output: {line.strip()}")
                used, total = float(values[3]), float(values[4])
                if not math.isfinite(used) or not math.isfinite(total) or not 0 <= used <= total or total <= 0:
                    raise ValueError("NVIDIA memory fields are not valid numbers")
                with self.lock:
                    self.rows.append({"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                                      "observed_perf_counter_s": time.perf_counter(), "phase": self.phase,
                                      "gpu_index": int(values[0]), "gpu_uuid": values[1], "gpu_name": values[2],
                                      "used_mib": used, "total_mib": total})
                self.ready.set()
            except (ValueError, IndexError) as error:
                with self.lock:
                    self.errors.append(str(error))

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            if self.thread is not None:
                self.thread.join(timeout=3)
            stderr = self.process.stderr.read().strip()
            if stderr:
                self.errors.append(stderr)


def phase_summary(rows, phase):
    selected = [row for row in rows if row["phase"] == phase]
    if len(selected) < 2:
        raise ValueError(f"Too few memory samples during {phase}: {len(selected)}")
    values = [row["used_mib"] for row in selected]
    gaps = [b["observed_perf_counter_s"] - a["observed_perf_counter_s"] for a, b in zip(selected, selected[1:])]
    return {"sample_count": len(selected), "used_mib_median": statistics.median(values),
            "used_mib_min": min(values), "used_mib_sampled_max": max(values),
            "observed_sample_gap_s_median": statistics.median(gaps), "observed_sample_gap_s_max": max(gaps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampler-check-only", action="store_true", help="Read GPU samples for two seconds without model requests.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    executable = shutil.which("nvidia-smi")
    if not executable and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if not executable:
        raise SystemExit("FAIL: NVIDIA query executable is missing")
    sampler = GPUMemorySampler(executable)
    if args.sampler_check_only:
        try:
            sampler.start()
            time.sleep(2)
        finally:
            sampler.stop()
        if sampler.errors:
            raise RuntimeError(str(sampler.errors))
        print(json.dumps(phase_summary(sampler.rows, "loaded_idle_before"), indent=2))
        print("SAMPLER_OK: GPU readings only; no model request or memory workload measurement was performed.")
        return

    started = datetime.now(timezone.utc)
    record = {
        "purpose": "sampled_total_gpu_memory_during_initial_serial_navigation_requests",
        "started_at_utc": started.isoformat(), "requested_sample_interval_ms": 200,
        "definition": "GPU 0 total used memory reported by nvidia-smi; includes the model service, caches, desktop and other GPU processes",
        "limitations": "sampled maximum is not an exact transient peak, per-model allocation, or torch max_memory_allocated",
        "workload_window": "one warm-up plus five serial fresh-task frame requests, including client setup/cleanup overhead",
        "timing_comparison": "this run includes concurrent NVIDIA telemetry; do not claim a speed change from the prior run",
        "sampler_command": sampler.command,
    }
    output = root / "logs" / f"gpu_memory_{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output.parent.mkdir(exist_ok=True)
    child = None
    latency_path = None
    problem = None
    try:
        print("GPU memory: device-total values, sampled at a requested 200ms interval.", flush=True)
        print("Loaded-idle readings before requests: 2 seconds...", flush=True)
        sampler.start()
        time.sleep(2)
        sampler.set_phase("request_run_including_warmup_and_overhead")
        child = subprocess.Popen([sys.executable, str(root / "checks/measure_initial_latency.py")],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in child.stdout:
            print(line, end="", flush=True)
            if line.startswith("Saved measurement: "):
                latency_path = Path(line.split(": ", 1)[1].strip())
        record["latency_process_exit_code"] = child.wait()
        sampler.set_phase("loaded_idle_after")
        print("Loaded-idle readings after requests: 2 seconds...", flush=True)
        time.sleep(2)
        if record["latency_process_exit_code"] != 0:
            raise RuntimeError("The latency request run did not complete; partial memory data was preserved")
        if latency_path is None or not latency_path.is_file():
            raise ValueError("Could not locate the associated latency record")
        record["latency_run_file"] = str(latency_path)
        record["latency_run_sha256"] = hashlib.sha256(latency_path.read_bytes()).hexdigest()
        latency = json.loads(latency_path.read_text(encoding="utf-8"))
        if latency.get("error") or latency["summary"]["measured_count"] != 5:
            raise ValueError("Associated request record is incomplete")
    except (Exception, KeyboardInterrupt) as error:
        problem = str(error) or "interrupted"
        record["error"] = problem
        print("Memory measurement error:", problem, flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=3)
        sampler.stop()
        record["samples"] = sampler.rows
        record["sampler_errors"] = sampler.errors
        if not problem:
            try:
                if sampler.errors:
                    raise ValueError(str(sampler.errors))
                if len({row["gpu_uuid"] for row in sampler.rows}) != 1:
                    raise ValueError("GPU identity changed or samples are missing")
                record["gpu_name"] = sampler.rows[0]["gpu_name"]
                record["gpu_total_mib"] = sampler.rows[0]["total_mib"]
                record["summary"] = {phase: phase_summary(sampler.rows, phase) for phase in [
                    "loaded_idle_before", "request_run_including_warmup_and_overhead", "loaded_idle_after"]}
                print("Memory summary (MiB):\n" + json.dumps(record["summary"], indent=2), flush=True)
            except Exception as error:
                problem = str(error)
                record["error"] = problem
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved GPU memory measurement:", output, flush=True)
    if problem:
        raise SystemExit("MEMORY_CHECK_INCOMPLETE: share this output; partial data was preserved.")
    print("GPU_MEMORY_SAMPLED: total-device sampled maximum; exact peak and navigation quality are not established.")


if __name__ == "__main__":
    main()
