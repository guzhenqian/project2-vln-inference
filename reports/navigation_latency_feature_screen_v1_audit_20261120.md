# 量化时延研究：独立固定输入审计

状态：`audited`。保留每轮原始测量；不平均不同轮的分位数。

| 运行/配置 | 格子 | HTTP模式 | 执行器/MP引擎/间隔 | IPC补丁 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| feature_screen_v1/awq_mp16 | 0_18 | nonstream | mp/True/16 | False | 10 | 0.3536 | 0.3603 | 未测 | 83.52 | 31.42 | 7.523 |
| feature_screen_v1/awq_mp16 | 1_18 | nonstream | mp/True/16 | False | 10 | 1.4969 | 1.5808 | 未测 | 349.06 | 138.01 | 8.223 |
| feature_screen_v1/awq_mp16 | 1_64 | nonstream | mp/True/16 | False | 10 | 1.9938 | 2.0482 | 未测 | 349.25 | 138.07 | 8.204 |
| feature_screen_v1/awq_linear16 | 0_18 | nonstream | mp/True/16 | True | 10 | 0.3555 | 0.3625 | 未测 | 83.44 | 31.41 | 7.526 |
| feature_screen_v1/awq_linear16 | 1_18 | nonstream | mp/True/16 | True | 10 | 0.9190 | 0.9609 | 未测 | 348.97 | 137.80 | 8.177 |
| feature_screen_v1/awq_linear16 | 1_64 | nonstream | mp/True/16 | True | 10 | 1.3963 | 1.4225 | 未测 | 349.17 | 137.85 | 8.186 |
| feature_screen_v1/awq_uni16 | 0_18 | nonstream | uni/True/16 | False | 10 | 0.4653 | 0.4692 | 未测 | 83.47 | 31.40 | 7.521 |
| feature_screen_v1/awq_uni16 | 1_18 | nonstream | uni/True/16 | False | 10 | 0.9835 | 0.9955 | 未测 | 348.81 | 137.96 | 8.190 |
| feature_screen_v1/awq_uni16 | 1_64 | nonstream | uni/True/16 | False | 10 | 1.3072 | 1.3518 | 未测 | 348.97 | 138.08 | 8.181 |
| feature_screen_v1/awq_uni4 | 0_18 | nonstream | uni/True/4 | False | 10 | 0.3590 | 0.3664 | 未测 | 83.40 | 31.29 | 7.492 |
| feature_screen_v1/awq_uni4 | 1_18 | nonstream | uni/True/4 | False | 10 | 0.8588 | 0.8847 | 未测 | 348.98 | 137.80 | 8.162 |
| feature_screen_v1/awq_uni4 | 1_64 | nonstream | uni/True/4 | False | 10 | 1.3309 | 1.3799 | 未测 | 349.03 | 137.93 | 8.159 |
| feature_screen_v1/awq_uni1 | 0_18 | nonstream | uni/True/1 | False | 10 | 0.3493 | 0.3562 | 未测 | 83.53 | 31.30 | 7.510 |
| feature_screen_v1/awq_uni1 | 1_18 | nonstream | uni/True/1 | False | 10 | 0.8457 | 0.8653 | 未测 | 348.71 | 137.82 | 8.187 |
| feature_screen_v1/awq_uni1 | 1_64 | nonstream | uni/True/1 | False | 10 | 1.3714 | 1.3886 | 未测 | 348.83 | 137.79 | 8.178 |
| feature_screen_v1/awq_inprocess1 | 0_18 | nonstream | uni/False/1 | False | 10 | 0.3495 | 0.3584 | 未测 | 83.53 | 31.32 | 7.517 |
| feature_screen_v1/awq_inprocess1 | 1_18 | nonstream | uni/False/1 | False | 10 | 0.7598 | 0.7837 | 未测 | 348.83 | 137.82 | 8.178 |
| feature_screen_v1/awq_inprocess1 | 1_64 | nonstream | uni/False/1 | False | 10 | 1.3473 | 1.3657 | 未测 | 348.71 | 137.90 | 8.176 |

格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。

## 匹配比较

| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |
|---|---|---|---|---:|---:|---|
| same_model_linear_ipc_change | 00_awq_mp16 → 01_awq_linear16 | 0_18 | 是 | -0.53% | 1.0059 | True |
| same_model_linear_ipc_change | 00_awq_mp16 → 01_awq_linear16 | 1_18 | 是 | 38.61% | 0.6078 | True |
| same_model_linear_ipc_change | 00_awq_mp16 → 01_awq_linear16 | 1_64 | 是 | 29.96% | 0.6945 | True |
| same_model_executor_change | 00_awq_mp16 → 02_awq_uni16 | 0_18 | 是 | -31.59% | 1.3022 | True |
| same_model_executor_change | 00_awq_mp16 → 02_awq_uni16 | 1_18 | 是 | 34.30% | 0.6298 | True |
| same_model_executor_change | 00_awq_mp16 → 02_awq_uni16 | 1_64 | 是 | 34.44% | 0.6600 | True |
| same_model_executor_change | 00_awq_mp16 → 03_awq_uni4 | 0_18 | 否 | 不作归因 | 1.0167 | True |
| same_model_executor_change | 00_awq_mp16 → 03_awq_uni4 | 1_18 | 否 | 不作归因 | 0.5596 | True |
| same_model_executor_change | 00_awq_mp16 → 03_awq_uni4 | 1_64 | 否 | 不作归因 | 0.6737 | True |
| same_model_executor_change | 00_awq_mp16 → 04_awq_uni1 | 0_18 | 否 | 不作归因 | 0.9885 | True |
| same_model_executor_change | 00_awq_mp16 → 04_awq_uni1 | 1_18 | 否 | 不作归因 | 0.5474 | True |
| same_model_executor_change | 00_awq_mp16 → 04_awq_uni1 | 1_64 | 否 | 不作归因 | 0.6779 | True |
| same_model_executor_change | 00_awq_mp16 → 05_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.9946 | True |
| same_model_executor_change | 00_awq_mp16 → 05_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.4957 | True |
| same_model_executor_change | 00_awq_mp16 → 05_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.6668 | True |
| same_model_executor_change | 01_awq_linear16 → 02_awq_uni16 | 0_18 | 否 | 不作归因 | 1.2946 | True |
| same_model_executor_change | 01_awq_linear16 → 02_awq_uni16 | 1_18 | 否 | 不作归因 | 1.0361 | True |
| same_model_executor_change | 01_awq_linear16 → 02_awq_uni16 | 1_64 | 否 | 不作归因 | 0.9502 | True |
| same_model_executor_change | 01_awq_linear16 → 03_awq_uni4 | 0_18 | 否 | 不作归因 | 1.0108 | True |
| same_model_executor_change | 01_awq_linear16 → 03_awq_uni4 | 1_18 | 否 | 不作归因 | 0.9207 | True |
| same_model_executor_change | 01_awq_linear16 → 03_awq_uni4 | 1_64 | 否 | 不作归因 | 0.9700 | True |
| same_model_executor_change | 01_awq_linear16 → 04_awq_uni1 | 0_18 | 否 | 不作归因 | 0.9828 | True |
| same_model_executor_change | 01_awq_linear16 → 04_awq_uni1 | 1_18 | 否 | 不作归因 | 0.9005 | True |
| same_model_executor_change | 01_awq_linear16 → 04_awq_uni1 | 1_64 | 否 | 不作归因 | 0.9761 | True |
| same_model_executor_change | 01_awq_linear16 → 05_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.9888 | True |
| same_model_executor_change | 01_awq_linear16 → 05_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.8156 | True |
| same_model_executor_change | 01_awq_linear16 → 05_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.9600 | True |
| same_model_repetition | 02_awq_uni16 → 03_awq_uni4 | 0_18 | 否 | 不作归因 | 0.7808 | True |
| same_model_repetition | 02_awq_uni16 → 03_awq_uni4 | 1_18 | 否 | 不作归因 | 0.8886 | True |
| same_model_repetition | 02_awq_uni16 → 03_awq_uni4 | 1_64 | 否 | 不作归因 | 1.0208 | True |
| same_model_repetition | 02_awq_uni16 → 04_awq_uni1 | 0_18 | 否 | 不作归因 | 0.7591 | True |
| same_model_repetition | 02_awq_uni16 → 04_awq_uni1 | 1_18 | 否 | 不作归因 | 0.8692 | True |
| same_model_repetition | 02_awq_uni16 → 04_awq_uni1 | 1_64 | 否 | 不作归因 | 1.0272 | True |
| same_model_inprocess_change | 02_awq_uni16 → 05_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.7638 | True |
| same_model_inprocess_change | 02_awq_uni16 → 05_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.7872 | True |
| same_model_inprocess_change | 02_awq_uni16 → 05_awq_inprocess1 | 1_64 | 否 | 不作归因 | 1.0103 | True |
| same_model_repetition | 03_awq_uni4 → 04_awq_uni1 | 0_18 | 否 | 不作归因 | 0.9723 | True |
| same_model_repetition | 03_awq_uni4 → 04_awq_uni1 | 1_18 | 否 | 不作归因 | 0.9781 | True |
| same_model_repetition | 03_awq_uni4 → 04_awq_uni1 | 1_64 | 否 | 不作归因 | 1.0063 | True |
| same_model_inprocess_change | 03_awq_uni4 → 05_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.9782 | True |
| same_model_inprocess_change | 03_awq_uni4 → 05_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.8858 | True |
| same_model_inprocess_change | 03_awq_uni4 → 05_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.9897 | True |
| same_model_inprocess_change | 04_awq_uni1 → 05_awq_inprocess1 | 0_18 | 是 | -0.06% | 1.0062 | True |
| same_model_inprocess_change | 04_awq_uni1 → 05_awq_inprocess1 | 1_18 | 是 | 10.16% | 0.9057 | True |
| same_model_inprocess_change | 04_awq_uni1 → 05_awq_inprocess1 | 1_64 | 是 | 1.76% | 0.9835 | True |

不可直接比较的原因：

- 00_awq_mp16 → 03_awq_uni4: engine configuration differs；observed execution identity differs
- 00_awq_mp16 → 03_awq_uni4: engine configuration differs；observed execution identity differs
- 00_awq_mp16 → 03_awq_uni4: engine configuration differs；observed execution identity differs
- 00_awq_mp16 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 00_awq_mp16 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 00_awq_mp16 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 00_awq_mp16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_awq_mp16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_awq_mp16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 02_awq_uni16: observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 02_awq_uni16: observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 02_awq_uni16: observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 03_awq_uni4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 03_awq_uni4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 03_awq_uni4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 04_awq_uni1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 04_awq_uni1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 04_awq_uni1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_linear16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_awq_uni16 → 03_awq_uni4: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 03_awq_uni4: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 03_awq_uni4: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs
- 02_awq_uni16 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_awq_uni4 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 03_awq_uni4 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 03_awq_uni4 → 04_awq_uni1: engine configuration differs；observed execution identity differs
- 03_awq_uni4 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_awq_uni4 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_awq_uni4 → 05_awq_inprocess1: engine configuration differs；observed execution identity differs

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
