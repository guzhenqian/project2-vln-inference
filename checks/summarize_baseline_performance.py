"""Offline summary of existing baseline logs. No HTTP, GPU sampling or inference."""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "initial_v1": ("initial_latency_20261019T085254530037Z.json", 5),
    "repeat_v1": ("repeated_latency_20261019T105629469754Z.json", 50),
    "repeat_v2": ("repeated_latency_grounded_stop_v2_20261019T111120186377Z.json", 50),
    "staged_v2": ("initial_latency_grounded_stop_v2_20261020T053809229693Z.json", 5),
}


def read_log(name):
    path = ROOT / "logs" / name
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def stats(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Missing, negative or nonfinite measurement")
    return dict(count=len(values), median=statistics.median(values),
                min=min(values), max=max(values))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_completion(log, name):
    status = log.get("measurement_status")
    if status is None:
        require(name in {"initial_latency_20261019T085254530037Z.json",
                         "initial_latency_20261019T091137283029Z.json"},
                name + ": unexpected missing completion field")
        require(log["summary"]["measured_count"] == 5
                and log["summary"]["warmup_count"] == 1
                and len(log["attempts"]) == 6
                and all(a["structural_check"] is True for a in log["attempts"])
                and log["stats_delta"]["processed"] == 6
                and log["stats_delta"]["total_requests"] == 6
                and log["stats_delta"]["errors"] == 0
                and log["stats_delta"]["dropped"] == 0,
                name + ": legacy completion evidence missing")
        return "legacy_schema_checked_by_counts_summary_and_structural_results"
    require(status == "complete", name + ": incomplete")
    return "explicit_complete"


def summarize_run(name, count):
    log, source = read_log(name)
    completion_evidence = check_completion(log, name)
    attempts = log["attempts"]
    measured = [a for a in attempts if a["warmup"] is False]
    require(len(measured) == count and len(attempts) == count + 1,
            name + ": incorrect measured/warmup counts")
    require(len({a["task_id"] for a in attempts}) == len(attempts),
            name + ": repeated task identity")
    delta = log["stats_delta"]
    require(delta["processed"] == len(attempts) and delta["total_requests"] == len(attempts)
            and delta["dropped"] == 0 and delta["errors"] == 0, name + ": stats mismatch")
    traces = []
    for attempt in measured:
        matches = attempt["diagnostic_matches"]
        require(len(matches) == 1 and matches[0]["trace"]["task_id"] == attempt["task_id"],
                name + ": diagnostic identity mismatch")
        trace = matches[0]["trace"]
        require(trace["generation"] == log["configuration"]["generation"],
                name + ": generation mismatch")
        traces.append(trace)
    client = stats([a["client_elapsed_s"] for a in measured])
    service = stats([a["service_processing_s"] for a in measured])
    require(math.isclose(client["median"], log["summary"]["client_elapsed_s"]["median"],
                         abs_tol=1e-9), name + ": stored client summary mismatch")
    require(math.isclose(service["median"], log["summary"]["service_processing_s"]["median"],
                         abs_tol=1e-9), name + ": stored service summary mismatch")
    grouped = defaultdict(list)
    for attempt in measured:
        grouped[attempt["case_id"]].append(attempt)
    result = dict(source=source, completion_evidence=completion_evidence,
                  formal_count=count, warmup_count=1,
                  model=log["manifest"]["model"], packages=log.get("packages", {}),
                  configuration=log["configuration"],
                  manifest_sha256=log["manifest_sha256"],
                  system_prompt_sha256=log.get("system_prompt_sha256"),
                  client_elapsed_s=client, service_processing_s=service,
                  completion_tokens=stats([a["completion_tokens"] for a in measured]),
                  prompt_token_values=sorted({t["model_response"]["prompt_tokens"] for t in traces}),
                  normal_finish_count=sum(a["finish_reason"] == "stop" for a in measured),
                  valid_json_count=sum(a["original_json_valid"] is True for a in measured),
                  stats_delta_including_warmup=delta,
                  per_case_client_s={case: stats([a["client_elapsed_s"] for a in rows])
                                     for case, rows in grouped.items()})
    if all("timings_s" in t for t in traces):
        fields = ("prepare_s", "model_call_s", "parse_s", "total_processing_s")
        segments = [t["timings_s"] for t in traces]
        for attempt, segment in zip(measured, segments):
            total = sum(segment[k] for k in fields[:3])
            require(math.isclose(total, segment["total_processing_s"], abs_tol=1e-9)
                    and math.isclose(total, attempt["service_processing_s"], abs_tol=1e-9),
                    name + ": segment total mismatch")
        result["segments_s"] = {k: stats([s[k] for s in segments]) for k in fields}
        result["model_call_fraction_of_summed_service"] = (
            sum(s["model_call_s"] for s in segments) /
            sum(s["total_processing_s"] for s in segments))
    return result


def main():
    runs = {key: summarize_run(*spec) for key, spec in RUNS.items()}
    require(runs["repeat_v1"]["manifest_sha256"] == runs["repeat_v2"]["manifest_sha256"],
            "Prompt comparison did not use the same manifest")
    memory, source = read_log("gpu_memory_20261019T091135132377Z.json")
    require(memory["latency_process_exit_code"] == 0 and not memory["sampler_errors"],
            "GPU sampler or linked request run failed")
    linked, linked_source = read_log(Path(memory["latency_run_file"]).name)
    check_completion(linked, Path(memory["latency_run_file"]).name)
    require(linked_source["sha256"] == memory["latency_run_sha256"], "Linked memory workload mismatch")
    phases = defaultdict(list)
    for sample in memory["samples"]:
        phases[sample["phase"]].append(sample["used_mib"])
    memory_stats = {phase: stats(values) for phase, values in phases.items()}
    for phase, values in memory_stats.items():
        old = memory["summary"][phase]
        require(values["count"] == old["sample_count"]
                and values["median"] == old["used_mib_median"]
                and values["min"] == old["used_mib_min"]
                and values["max"] == old["used_mib_sampled_max"], "GPU summary mismatch")
    output = dict(date="2026-10-24", method="offline_recompute_from_existing_logs",
                  no_new_model_requests=True, runs=runs,
                  gpu_memory=dict(source=source, linked_workload=linked_source,
                                  device=memory["gpu_name"], total_mib=memory["gpu_total_mib"],
                                  requested_interval_ms=memory["requested_sample_interval_ms"],
                                  phases_used_mib=memory_stats,
                                  definition=memory["definition"], limitations=memory["limitations"]))
    report = ["# baseline性能汇总（已有日志离线重算）", "",
              "日期：2026-10-24。已核对完整状态、预热排除、样本数、任务与诊断关联、生成参数、计数及已存中位数。未新增推理、显存采样或服务重启。", "",
              "最早两份初测日志尚无measurement_status字段，依据正式/预热数量、结构校验、汇总与流水线计数核验完成证据；JSON中单列该旧格式处理方式，不将缺失状态直接视为complete。", "",
              "固定输入为5张640×480桌面图，每次新建独立任务、串行同步请求；每轮1次预热不计入下表。Qwen3-VL-2B-Instruct、RTX 4070 Ti单卡；历史部署记录与启动脚本为LMDeploy PyTorch后端、BF16、tp=1。生成上限512 token、temperature=0.3，未固定随机种子。", "",
              "| 测量 | 正式次数 | 客户端中位/秒 | 客户端最短～最长/秒 | 业务中位/秒 | 输出token中位 |",
              "| --- | ---: | ---: | --- | ---: | ---: |"]
    labels = dict(initial_v1="10/19初测 baseline_v1", repeat_v1="10/19重复 baseline_v1",
                  repeat_v2="10/19提示词对照 grounded_stop_v2", staged_v2="10/20分段 grounded_stop_v2")
    for key, run in runs.items():
        c = run["client_elapsed_s"]
        report.append(f"| {labels[key]} | {run['formal_count']} | {c['median']:.6f} | "
                      f"{c['min']:.6f}～{c['max']:.6f} | {run['service_processing_s']['median']:.6f} | "
                      f"{run['completion_tokens']['median']:g} |")
    report += ["", "客户端口径为同步帧HTTP往返，排除图片编码、任务创建及清理；业务口径为准备、后端模型调用及解析，不含流水线排队等开销。两轮50次均零错误/丢帧，正式输出均正常结束且JSON可解析；这是链路完整性，不是质量验收。", "",
               "两轮50次的输入清单相同，但系统提示词改变，输入token为646→854，输出token中位142→126.5，运行时刻也不同。中位耗时差不能宣称推理引擎加速，不能计算加速比或将10/20的5次当作50次之后的新优化。不同运行分别列示，不混合计算。", "",
               "## 5条正式样本的分段", "",
               "| 分段 | 中位/秒 |", "| --- | ---: |"]
    staged = runs["staged_v2"]
    for field, label in (("prepare_s", "准备"), ("model_call_s", "模型调用"),
                         ("parse_s", "解析"), ("total_processing_s", "业务总处理")):
        report.append(f"| {label} | {staged['segments_s'][field]['median']:.9f} |")
    report += ["", f"逐条分段之和与业务耗时一致。模型调用占这5条累计业务耗时的"
               f"{100 * staged['model_call_fraction_of_summed_service']:.3f}%。"
               "占比用累计耗时计算，各分段中位数不要求相加等于总耗时中位数。", "",
               "模型调用包含线程池调度、后端HTTP和推理，不能称纯GPU时间；尚未拆分视觉编码、prefill、decode。当前优先调查模型调用范围，解析优化对总延迟的改善空间很小。5次不足以代表稳定尾延迟。", "",
               "## 整卡显存采样", "",
               "| 窗口 | 样本数 | 中位/MiB | 最小/MiB | 采样最大/MiB |", "| --- | ---: | ---: | ---: | ---: |"]
    phase_labels = dict(loaded_idle_before="模型已加载、请求前空闲",
                        request_run_including_warmup_and_overhead="请求窗口（含预热与辅助开销）",
                        loaded_idle_after="请求后空闲")
    for phase, values in memory_stats.items():
        report.append(f"| {phase_labels[phase]} | {values['count']} | {values['median']:g} | "
                      f"{values['min']:g} | {values['max']:g} |")
    report += ["", "显卡总容量12282 MiB，目标采样间隔200毫秒。请求窗口采样最大7710 MiB，约7.53 GiB；包含桌面、缓存和其他GPU进程，可能漏过短暂峰值。它不是模型专属显存，也不是精确瞬时峰值。显存与计时属于另一次同时运行监测的窗口，不作加速对照。", "",
               "## 当前质量与视频边界", "",
               "10/19提示词对照中非零航点2→0、自报到达9→0，但至少5条新版理由仍声称桌面可移动。10/24沿当前朝向探索任务下，两张桌面图又各3/3前进。任务指令不同，不能把旧50次全停等同当前质量通过。逐图结论见baseline_image_review_20261024.md。", "",
               "10/24的12图及6条复测不并入固定5图性能分布；该轮无预热，任务指令及输入组成不同，耗时仅保留为诊断请求记录。", "",
               "视频基础功能保留10/19阶段记录的既有验证，10/24用户截图再次确认加载与首帧；不重复整套测试。连续视频新模型结果频率及当日完整连续复测没有新证据。文件标称31.198 FPS、页面送帧、API返回缓存结果、真正完成的新模型建议是不同事件；本报告不宣称视频决策频率、稳定P95、并发吞吐或实机闭环。", "",
               "## 复现与证据", "", "在项目根目录执行，仅读取既有文件：", "",
               "```bash", "~/.venvs/project2/bin/python checks/summarize_baseline_performance.py", "```", "",
               "完整精度、逐图中位数、配置、软件版本和来源SHA256见baseline_performance_summary_20261024.json。", ""]
    for run in runs.values():
        report.append(f"- {run['source']['path']}；SHA256 `{run['source']['sha256']}`")
    report.append(f"- {source['path']}；关联请求日志哈希已核对，详见JSON。")
    (ROOT / "reports/baseline_performance_summary_20261024.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "reports/baseline_performance_summary_20261024.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print("OFFLINE_SUMMARY_COMPLETE: 110 formal requests from 4 historical runs; no new inference.")
    print("reports/baseline_performance_summary_20261024.md")


if __name__ == "__main__":
    main()
