# 量化时延研究：独立固定输入审计

状态：`audited`。保留每轮原始测量；不平均不同轮的分位数。

| 运行/配置 | 格子 | HTTP模式 | 执行器/间隔 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ipc_screen_v1/awq_mp | 1_18 | nonstream | mp/16 | 10 | 1.5189 | 1.7917 | 未测 | 349.16 | 140.04 | 8.184 |
| ipc_screen_v1/awq_uni | 1_18 | nonstream | uni/16 | 10 | 0.9904 | 1.0276 | 未测 | 348.72 | 137.43 | 8.174 |

格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。

## 匹配比较

| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |
|---|---|---|---|---:|---:|---|
| same_model_executor_change | 00_awq_mp → 01_awq_uni | 1_18 | 是 | 34.79% | 0.5735 | True |

## 前置路径诊断

| 运行/配置 | 格子 | Frontend prompt/ms | Prompt结束→worker准备开始/ms | Worker准备/ms | 输入像素MiB |
|---|---|---:|---:|---:|---:|
| ipc_screen_v1/awq_mp | 1_18 | 28.31 | 783.83 | 4.65 | 63.28 |
| ipc_screen_v1/awq_uni | 1_18 | 30.55 | 126.01 | 6.17 | 63.28 |

这些CPU区间可能交叠；MP接收包括等待，不可相加成互斥总耗时。完整事件分段保存在JSON。

## 限制、失败与未完成项

- Fixed-input tests do not establish closed-loop navigation quality, success rate or generalization.
- All timing comparisons keep their HTTP mode; nonstream has no TTFT.
- Quantization comparisons require FP16 and quantized configurations with the same executor and engine settings.
- MP-to-uni improvements are execution-path changes, not solely quantization gains.
- CUDA event intervals can include dispatch gaps; decode uses actual forwards, including post-response tail work.
- Profile intervals overlap and include waiting; no exclusive overhead decomposition is claimed.
- Checkpoint identities are validated against prior SHA receipts; this CPU-only audit does not rehash model weights.
- Small repeated fixed inputs and raw P50/P95 are descriptive; repetitions are not pooled or treated as significance tests.

本报告不宣称完整导航质量通过。各组实际输入tensor SHA、源码/launch SHA、原始请求和逐前向计时已在JSON逐项保留。
