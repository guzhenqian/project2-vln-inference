# 量化时延研究：独立固定输入审计

状态：`audited`。保留每轮原始测量；不平均不同轮的分位数。

| 运行/配置 | 格子 | HTTP模式 | 执行器/MP引擎/间隔 | IPC补丁 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| system_full_v1/fp16_mp | 0_18 | nonstream | mp/True/16 | False | 10 | 0.5285 | 0.5314 | 未测 | 79.47 | 31.48 | 17.243 |
| system_full_v1/fp16_mp | 0_64 | nonstream | mp/True/16 | False | 10 | 1.4202 | 1.8486 | 未测 | 79.29 | 31.43 | 17.264 |
| system_full_v1/fp16_mp | 1_18 | nonstream | mp/True/16 | False | 10 | 1.8289 | 1.9799 | 未测 | 323.28 | 143.89 | 17.857 |
| system_full_v1/fp16_mp | 1_64 | nonstream | mp/True/16 | False | 10 | 2.7844 | 2.8435 | 未测 | 322.67 | 139.96 | 17.856 |
| system_full_v1/awq_w4a16_mp | 0_18 | nonstream | mp/True/16 | False | 10 | 0.3566 | 0.3646 | 未测 | 83.65 | 31.32 | 7.512 |
| system_full_v1/awq_w4a16_mp | 0_64 | nonstream | mp/True/16 | False | 10 | 0.7894 | 0.8042 | 未测 | 83.38 | 31.51 | 7.504 |
| system_full_v1/awq_w4a16_mp | 1_18 | nonstream | mp/True/16 | False | 10 | 1.5622 | 1.6369 | 未测 | 349.62 | 143.88 | 8.189 |
| system_full_v1/awq_w4a16_mp | 1_64 | nonstream | mp/True/16 | False | 10 | 2.0180 | 2.0554 | 未测 | 349.05 | 138.17 | 8.176 |
| system_full_v1/gptq_w4a16_mp | 0_18 | nonstream | mp/True/16 | False | 10 | 0.3537 | 0.3697 | 未测 | 83.51 | 31.43 | 7.514 |
| system_full_v1/gptq_w4a16_mp | 0_64 | nonstream | mp/True/16 | False | 10 | 0.7965 | 0.8026 | 未测 | 83.55 | 31.35 | 7.504 |
| system_full_v1/gptq_w4a16_mp | 1_18 | nonstream | mp/True/16 | False | 10 | 1.5759 | 1.9648 | 未测 | 349.19 | 143.80 | 8.196 |
| system_full_v1/gptq_w4a16_mp | 1_64 | nonstream | mp/True/16 | False | 10 | 2.0283 | 2.0433 | 未测 | 348.90 | 137.87 | 8.174 |
| system_full_v1/gptq_w4a16_uni | 0_18 | nonstream | uni/True/16 | False | 10 | 0.4692 | 0.4757 | 未测 | 83.59 | 31.51 | 7.508 |
| system_full_v1/gptq_w4a16_uni | 0_64 | nonstream | uni/True/16 | False | 10 | 0.7726 | 0.7824 | 未测 | 83.58 | 31.41 | 7.514 |
| system_full_v1/gptq_w4a16_uni | 1_18 | nonstream | uni/True/16 | False | 10 | 0.9821 | 1.0428 | 未测 | 348.90 | 137.87 | 8.175 |
| system_full_v1/gptq_w4a16_uni | 1_64 | nonstream | uni/True/16 | False | 10 | 1.3186 | 1.3287 | 未测 | 348.82 | 137.71 | 8.187 |
| system_full_v1/awq_w4a16_uni | 0_18 | nonstream | uni/True/16 | False | 10 | 0.4688 | 0.4787 | 未测 | 83.57 | 31.42 | 7.523 |
| system_full_v1/awq_w4a16_uni | 0_64 | nonstream | uni/True/16 | False | 10 | 0.7697 | 0.7860 | 未测 | 83.55 | 31.42 | 7.520 |
| system_full_v1/awq_w4a16_uni | 1_18 | nonstream | uni/True/16 | False | 10 | 0.9806 | 1.0312 | 未测 | 348.90 | 137.86 | 8.195 |
| system_full_v1/awq_w4a16_uni | 1_64 | nonstream | uni/True/16 | False | 10 | 1.3027 | 1.3286 | 未测 | 349.01 | 138.01 | 8.199 |
| system_full_v1/fp16_uni | 0_18 | nonstream | uni/True/16 | False | 10 | 0.7758 | 0.7848 | 未测 | 79.57 | 31.39 | 17.254 |
| system_full_v1/fp16_uni | 0_64 | nonstream | uni/True/16 | False | 10 | 1.3915 | 1.4053 | 未测 | 79.50 | 31.39 | 17.245 |
| system_full_v1/fp16_uni | 1_18 | nonstream | uni/True/16 | False | 10 | 1.2695 | 1.3201 | 未测 | 321.94 | 138.06 | 17.873 |
| system_full_v1/fp16_uni | 1_64 | nonstream | uni/True/16 | False | 10 | 1.9113 | 1.9287 | 未测 | 322.32 | 138.01 | 17.872 |

格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。

## 匹配比较

| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |
|---|---|---|---|---:|---:|---|
| matched_quantization | 00_fp16_mp → 01_awq_w4a16_mp | 0_18 | 是 | 32.53% | 0.6860 | 不适用 |
| matched_quantization | 00_fp16_mp → 01_awq_w4a16_mp | 0_64 | 是 | 44.42% | 0.4350 | 不适用 |
| matched_quantization | 00_fp16_mp → 01_awq_w4a16_mp | 1_18 | 是 | 14.58% | 0.8268 | 不适用 |
| matched_quantization | 00_fp16_mp → 01_awq_w4a16_mp | 1_64 | 是 | 27.52% | 0.7228 | 不适用 |
| matched_quantization | 00_fp16_mp → 02_gptq_w4a16_mp | 0_18 | 是 | 33.07% | 0.6957 | 不适用 |
| matched_quantization | 00_fp16_mp → 02_gptq_w4a16_mp | 0_64 | 是 | 43.91% | 0.4342 | 不适用 |
| matched_quantization | 00_fp16_mp → 02_gptq_w4a16_mp | 1_18 | 是 | 13.83% | 0.9924 | 不适用 |
| matched_quantization | 00_fp16_mp → 02_gptq_w4a16_mp | 1_64 | 是 | 27.15% | 0.7186 | 不适用 |
| matched_quantization | 00_fp16_mp → 03_gptq_w4a16_uni | 0_18 | 否 | 不作归因 | 0.8952 | 不适用 |
| matched_quantization | 00_fp16_mp → 03_gptq_w4a16_uni | 0_64 | 否 | 不作归因 | 0.4232 | 不适用 |
| matched_quantization | 00_fp16_mp → 03_gptq_w4a16_uni | 1_18 | 否 | 不作归因 | 0.5267 | 不适用 |
| matched_quantization | 00_fp16_mp → 03_gptq_w4a16_uni | 1_64 | 否 | 不作归因 | 0.4673 | 不适用 |
| matched_quantization | 00_fp16_mp → 04_awq_w4a16_uni | 0_18 | 否 | 不作归因 | 0.9009 | 不适用 |
| matched_quantization | 00_fp16_mp → 04_awq_w4a16_uni | 0_64 | 否 | 不作归因 | 0.4252 | 不适用 |
| matched_quantization | 00_fp16_mp → 04_awq_w4a16_uni | 1_18 | 否 | 不作归因 | 0.5209 | 不适用 |
| matched_quantization | 00_fp16_mp → 04_awq_w4a16_uni | 1_64 | 否 | 不作归因 | 0.4672 | 不适用 |
| same_model_executor_change | 00_fp16_mp → 05_fp16_uni | 0_18 | 是 | -46.80% | 1.4768 | True |
| same_model_executor_change | 00_fp16_mp → 05_fp16_uni | 0_64 | 是 | 2.02% | 0.7602 | True |
| same_model_executor_change | 00_fp16_mp → 05_fp16_uni | 1_18 | 是 | 30.58% | 0.6668 | True |
| same_model_executor_change | 00_fp16_mp → 05_fp16_uni | 1_64 | 是 | 31.36% | 0.6783 | True |
| same_model_executor_change | 01_awq_w4a16_mp → 04_awq_w4a16_uni | 0_18 | 是 | -31.47% | 1.3132 | True |
| same_model_executor_change | 01_awq_w4a16_mp → 04_awq_w4a16_uni | 0_64 | 是 | 2.48% | 0.9774 | True |
| same_model_executor_change | 01_awq_w4a16_mp → 04_awq_w4a16_uni | 1_18 | 是 | 37.23% | 0.6300 | True |
| same_model_executor_change | 01_awq_w4a16_mp → 04_awq_w4a16_uni | 1_64 | 是 | 35.45% | 0.6464 | True |
| matched_quantization | 05_fp16_uni → 01_awq_w4a16_mp | 0_18 | 否 | 不作归因 | 0.4645 | 不适用 |
| matched_quantization | 05_fp16_uni → 01_awq_w4a16_mp | 0_64 | 否 | 不作归因 | 0.5722 | 不适用 |
| matched_quantization | 05_fp16_uni → 01_awq_w4a16_mp | 1_18 | 否 | 不作归因 | 1.2400 | 不适用 |
| matched_quantization | 05_fp16_uni → 01_awq_w4a16_mp | 1_64 | 否 | 不作归因 | 1.0657 | 不适用 |
| same_model_executor_change | 02_gptq_w4a16_mp → 03_gptq_w4a16_uni | 0_18 | 是 | -32.64% | 1.2868 | True |
| same_model_executor_change | 02_gptq_w4a16_mp → 03_gptq_w4a16_uni | 0_64 | 是 | 3.01% | 0.9748 | True |
| same_model_executor_change | 02_gptq_w4a16_mp → 03_gptq_w4a16_uni | 1_18 | 是 | 37.68% | 0.5307 | True |
| same_model_executor_change | 02_gptq_w4a16_mp → 03_gptq_w4a16_uni | 1_64 | 是 | 34.99% | 0.6503 | True |
| matched_quantization | 05_fp16_uni → 02_gptq_w4a16_mp | 0_18 | 否 | 不作归因 | 0.4711 | 不适用 |
| matched_quantization | 05_fp16_uni → 02_gptq_w4a16_mp | 0_64 | 否 | 不作归因 | 0.5711 | 不适用 |
| matched_quantization | 05_fp16_uni → 02_gptq_w4a16_mp | 1_18 | 否 | 不作归因 | 1.4884 | 不适用 |
| matched_quantization | 05_fp16_uni → 02_gptq_w4a16_mp | 1_64 | 否 | 不作归因 | 1.0594 | 不适用 |
| matched_quantization | 05_fp16_uni → 03_gptq_w4a16_uni | 0_18 | 是 | 39.53% | 0.6062 | 不适用 |
| matched_quantization | 05_fp16_uni → 03_gptq_w4a16_uni | 0_64 | 是 | 44.48% | 0.5567 | 不适用 |
| matched_quantization | 05_fp16_uni → 03_gptq_w4a16_uni | 1_18 | 是 | 22.64% | 0.7899 | 不适用 |
| matched_quantization | 05_fp16_uni → 03_gptq_w4a16_uni | 1_64 | 是 | 31.01% | 0.6889 | 不适用 |
| matched_quantization | 05_fp16_uni → 04_awq_w4a16_uni | 0_18 | 是 | 39.58% | 0.6100 | 不适用 |
| matched_quantization | 05_fp16_uni → 04_awq_w4a16_uni | 0_64 | 是 | 44.68% | 0.5593 | 不适用 |
| matched_quantization | 05_fp16_uni → 04_awq_w4a16_uni | 1_18 | 是 | 22.76% | 0.7812 | 不适用 |
| matched_quantization | 05_fp16_uni → 04_awq_w4a16_uni | 1_64 | 是 | 31.84% | 0.6889 | 不适用 |

不可直接比较的原因：

- 00_fp16_mp → 03_gptq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 03_gptq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 03_gptq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 03_gptq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 04_awq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 04_awq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 04_awq_w4a16_uni: engine configuration differs；observed execution identity differs
- 00_fp16_mp → 04_awq_w4a16_uni: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 01_awq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 01_awq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 01_awq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 01_awq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 02_gptq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 02_gptq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 02_gptq_w4a16_mp: engine configuration differs；observed execution identity differs
- 05_fp16_uni → 02_gptq_w4a16_mp: engine configuration differs；observed execution identity differs

## 前置路径诊断

| 运行/配置 | 格子 | Frontend prompt/ms | Prompt结束→worker准备开始/ms | Worker准备/ms | 输入像素MiB |
|---|---|---:|---:|---:|---:|

这些CPU区间可能交叠；MP接收包括等待，不可相加成互斥总耗时。完整事件分段保存在JSON。

## 限制、失败与未完成项

- Fixed-input tests do not establish closed-loop navigation quality, success rate or generalization.
- All timing comparisons keep their HTTP mode; nonstream has no TTFT.
- Quantization comparisons require FP16 and quantized configurations with the same executor and engine settings.
- MP-to-uni improvements are execution-path changes, not solely quantization gains.
- Requested and effective engine configurations are separate. In-process HTTP uses a guarded private bridge and removes process isolation; concurrency responsiveness is untested.
- Linear IPC requires actual successful large-message send/receive receipts. Wide W4 kernels remain unsupported by this auditor pending dedicated numerical and per-projection coverage checks.
- CUDA event intervals can include dispatch gaps; decode uses actual forwards, including post-response tail work.
- Profile intervals overlap and include waiting; no exclusive overhead decomposition is claimed.
- Checkpoint identities are validated against prior SHA receipts; this CPU-only audit does not rehash model weights.
- Small repeated fixed inputs and raw P50/P95 are descriptive; repetitions are not pooled or treated as significance tests.

本报告不宣称完整导航质量通过。各组实际输入tensor SHA、源码/launch SHA、原始请求和逐前向计时已在JSON逐项保留。
