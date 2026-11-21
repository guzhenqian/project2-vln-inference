# 量化时延研究：独立固定输入审计

状态：`audited_with_failures_or_incomplete_attempts`。保留每轮原始测量；不平均不同轮的分位数。

| 运行/配置 | 格子 | HTTP模式 | 执行器/MP引擎/间隔 | IPC补丁 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| kernel_vision_screen_v1/fp16_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.5019 | 0.5244 | 未测 | 79.74 | 31.41 | 17.237 |
| kernel_vision_screen_v1/fp16_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 1.4524 | 1.4910 | 未测 | 79.53 | 31.40 | 17.226 |
| kernel_vision_screen_v1/fp16_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.8832 | 0.9152 | 未测 | 322.59 | 138.37 | 17.838 |
| kernel_vision_screen_v1/fp16_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.8701 | 1.8987 | 未测 | 322.43 | 138.14 | 17.842 |
| kernel_vision_screen_v1/awq_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3334 | 0.3417 | 未测 | 83.61 | 31.52 | 7.500 |
| kernel_vision_screen_v1/awq_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8206 | 0.8627 | 未测 | 83.50 | 31.44 | 7.486 |
| kernel_vision_screen_v1/awq_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7394 | 0.7663 | 未测 | 349.41 | 137.83 | 8.168 |
| kernel_vision_screen_v1/awq_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2764 | 2.4675 | 未测 | 349.43 | 137.66 | 8.168 |
| kernel_vision_screen_v1/awq_wide_all | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3175 | 0.3315 | 未测 | 80.47 | 31.45 | 6.502 |
| kernel_vision_screen_v1/awq_wide_all | 0_64 | nonstream | uni/False/2 | False | 10 | 0.7650 | 0.8463 | 未测 | 80.54 | 31.32 | 6.502 |
| kernel_vision_screen_v1/awq_wide_all | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7269 | 0.7412 | 未测 | 348.99 | 137.81 | 7.189 |
| kernel_vision_screen_v1/awq_wide_all | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2078 | 1.2922 | 未测 | 348.96 | 137.75 | 7.174 |
| kernel_vision_screen_v1/awq_wide_prefill | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3343 | 0.3526 | 未测 | 80.55 | 31.35 | 7.508 |
| kernel_vision_screen_v1/awq_wide_prefill | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8330 | 0.8776 | 未测 | 80.56 | 31.46 | 7.500 |
| kernel_vision_screen_v1/awq_wide_prefill | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7426 | 0.7600 | 未测 | 349.20 | 138.09 | 8.194 |
| kernel_vision_screen_v1/awq_wide_prefill | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2753 | 1.3071 | 未测 | 349.08 | 138.02 | 8.166 |
| kernel_vision_screen_v1/awq_qmax | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3338 | 0.3452 | 未测 | 83.52 | 30.95 | 7.517 |
| kernel_vision_screen_v1/awq_qmax | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8353 | 0.8687 | 未测 | 83.69 | 30.97 | 7.507 |
| kernel_vision_screen_v1/awq_qmax | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7394 | 0.7561 | 未测 | 348.72 | 137.70 | 8.197 |
| kernel_vision_screen_v1/awq_qmax | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2695 | 1.2922 | 未测 | 349.03 | 137.91 | 8.182 |
| kernel_vision_screen_v1/fp16_qmax | 0_18 | nonstream | uni/False/2 | False | 10 | 0.4989 | 0.5183 | 未测 | 79.32 | 31.51 | 17.250 |
| kernel_vision_screen_v1/fp16_qmax | 0_64 | nonstream | uni/False/2 | False | 10 | 1.4643 | 1.4777 | 未测 | 79.46 | 31.48 | 17.249 |
| kernel_vision_screen_v1/fp16_qmax | 1_18 | nonstream | uni/False/2 | False | 10 | 0.8838 | 0.8919 | 未测 | 322.32 | 137.91 | 17.874 |
| kernel_vision_screen_v1/fp16_qmax | 1_64 | nonstream | uni/False/2 | False | 10 | 1.8764 | 1.8948 | 未测 | 322.34 | 138.07 | 17.836 |

格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。

## 匹配比较

| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |
|---|---|---|---|---:|---:|---|
| matched_quantization | 00_fp16_inprocess2 → 01_awq_inprocess2 | 0_18 | 是 | 33.57% | 0.6516 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 01_awq_inprocess2 | 0_64 | 是 | 43.50% | 0.5786 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 01_awq_inprocess2 | 1_18 | 是 | 16.28% | 0.8373 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 01_awq_inprocess2 | 1_64 | 是 | 31.75% | 1.2996 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 02_awq_wide_all | 0_18 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 02_awq_wide_all | 0_64 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 02_awq_wide_all | 1_18 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 02_awq_wide_all | 1_64 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 03_awq_wide_prefill | 0_18 | 是 | 33.40% | 0.6723 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 03_awq_wide_prefill | 0_64 | 是 | 42.65% | 0.5886 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 03_awq_wide_prefill | 1_18 | 是 | 15.92% | 0.8305 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 03_awq_wide_prefill | 1_64 | 是 | 31.80% | 0.6884 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 04_awq_qmax | 0_18 | 否 | 不作归因 | 0.6582 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 04_awq_qmax | 0_64 | 否 | 不作归因 | 0.5826 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 04_awq_qmax | 1_18 | 否 | 不作归因 | 0.8261 | 不适用 |
| matched_quantization | 00_fp16_inprocess2 → 04_awq_qmax | 1_64 | 否 | 不作归因 | 0.6806 | 不适用 |
| same_model_vision_qmax_change | 00_fp16_inprocess2 → 05_fp16_qmax | 0_18 | 是 | 0.60% | 0.9882 | True |
| same_model_vision_qmax_change | 00_fp16_inprocess2 → 05_fp16_qmax | 0_64 | 是 | -0.82% | 0.9911 | True |
| same_model_vision_qmax_change | 00_fp16_inprocess2 → 05_fp16_qmax | 1_18 | 是 | -0.07% | 0.9746 | True |
| same_model_vision_qmax_change | 00_fp16_inprocess2 → 05_fp16_qmax | 1_64 | 是 | -0.33% | 0.9980 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 02_awq_wide_all | 0_18 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 02_awq_wide_all | 0_64 | 否 | 不作归因 | 未通过门槛 | False |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 02_awq_wide_all | 1_18 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 02_awq_wide_all | 1_64 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 03_awq_wide_prefill | 0_18 | 是 | -0.26% | 1.0317 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 03_awq_wide_prefill | 0_64 | 是 | -1.52% | 1.0173 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 03_awq_wide_prefill | 1_18 | 是 | -0.43% | 0.9918 | True |
| same_model_kernel_implementation_change | 01_awq_inprocess2 → 03_awq_wide_prefill | 1_64 | 是 | 0.08% | 0.5297 | True |
| same_model_vision_qmax_change | 01_awq_inprocess2 → 04_awq_qmax | 0_18 | 是 | -0.13% | 1.0100 | True |
| same_model_vision_qmax_change | 01_awq_inprocess2 → 04_awq_qmax | 0_64 | 是 | -1.80% | 1.0070 | True |
| same_model_vision_qmax_change | 01_awq_inprocess2 → 04_awq_qmax | 1_18 | 是 | -0.00% | 0.9866 | True |
| same_model_vision_qmax_change | 01_awq_inprocess2 → 04_awq_qmax | 1_64 | 是 | 0.54% | 0.5237 | True |
| matched_quantization | 05_fp16_qmax → 01_awq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6594 | 不适用 |
| matched_quantization | 05_fp16_qmax → 01_awq_inprocess2 | 0_64 | 否 | 不作归因 | 0.5838 | 不适用 |
| matched_quantization | 05_fp16_qmax → 01_awq_inprocess2 | 1_18 | 否 | 不作归因 | 0.8592 | 不适用 |
| matched_quantization | 05_fp16_qmax → 01_awq_inprocess2 | 1_64 | 否 | 不作归因 | 1.3022 | 不适用 |
| same_model_kernel_implementation_change | 02_awq_wide_all → 03_awq_wide_prefill | 0_18 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_kernel_implementation_change | 02_awq_wide_all → 03_awq_wide_prefill | 0_64 | 否 | 不作归因 | 未通过门槛 | False |
| same_model_kernel_implementation_change | 02_awq_wide_all → 03_awq_wide_prefill | 1_18 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_kernel_implementation_change | 02_awq_wide_all → 03_awq_wide_prefill | 1_64 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_system_and_kernel_bundle | 02_awq_wide_all → 04_awq_qmax | 0_18 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_system_and_kernel_bundle | 02_awq_wide_all → 04_awq_qmax | 0_64 | 否 | 不作归因 | 未通过门槛 | False |
| same_model_system_and_kernel_bundle | 02_awq_wide_all → 04_awq_qmax | 1_18 | 否 | 不作归因 | 未通过门槛 | True |
| same_model_system_and_kernel_bundle | 02_awq_wide_all → 04_awq_qmax | 1_64 | 否 | 不作归因 | 未通过门槛 | True |
| matched_quantization | 05_fp16_qmax → 02_awq_wide_all | 0_18 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 05_fp16_qmax → 02_awq_wide_all | 0_64 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 05_fp16_qmax → 02_awq_wide_all | 1_18 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| matched_quantization | 05_fp16_qmax → 02_awq_wide_all | 1_64 | 否 | 不作归因 | 未通过门槛 | 不适用 |
| same_model_system_and_kernel_bundle | 03_awq_wide_prefill → 04_awq_qmax | 0_18 | 是 | 0.13% | 0.9790 | True |
| same_model_system_and_kernel_bundle | 03_awq_wide_prefill → 04_awq_qmax | 0_64 | 是 | -0.27% | 0.9899 | True |
| same_model_system_and_kernel_bundle | 03_awq_wide_prefill → 04_awq_qmax | 1_18 | 是 | 0.43% | 0.9948 | True |
| same_model_system_and_kernel_bundle | 03_awq_wide_prefill → 04_awq_qmax | 1_64 | 是 | 0.46% | 0.9886 | True |
| matched_quantization | 05_fp16_qmax → 03_awq_wide_prefill | 0_18 | 否 | 不作归因 | 0.6803 | 不适用 |
| matched_quantization | 05_fp16_qmax → 03_awq_wide_prefill | 0_64 | 否 | 不作归因 | 0.5939 | 不适用 |
| matched_quantization | 05_fp16_qmax → 03_awq_wide_prefill | 1_18 | 否 | 不作归因 | 0.8522 | 不适用 |
| matched_quantization | 05_fp16_qmax → 03_awq_wide_prefill | 1_64 | 否 | 不作归因 | 0.6898 | 不适用 |
| matched_quantization | 05_fp16_qmax → 04_awq_qmax | 0_18 | 是 | 33.09% | 0.6660 | 不适用 |
| matched_quantization | 05_fp16_qmax → 04_awq_qmax | 0_64 | 是 | 42.95% | 0.5879 | 不适用 |
| matched_quantization | 05_fp16_qmax → 04_awq_qmax | 1_18 | 是 | 16.34% | 0.8477 | 不适用 |
| matched_quantization | 05_fp16_qmax → 04_awq_qmax | 1_64 | 是 | 32.34% | 0.6819 | 不适用 |

不可直接比较的原因：

- 00_fp16_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 00_fp16_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 00_fp16_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 00_fp16_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 00_fp16_inprocess2 → 04_awq_qmax: observed execution identity differs；other candidate switches differ
- 00_fp16_inprocess2 → 04_awq_qmax: observed execution identity differs；other candidate switches differ
- 00_fp16_inprocess2 → 04_awq_qmax: observed execution identity differs；other candidate switches differ
- 00_fp16_inprocess2 → 04_awq_qmax: observed execution identity differs；other candidate switches differ
- 01_awq_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 01_awq_inprocess2 → 02_awq_wide_all: same-weight output equality not established；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 01_awq_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 01_awq_inprocess2 → 02_awq_wide_all: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 05_fp16_qmax → 01_awq_inprocess2: observed execution identity differs；other candidate switches differ
- 05_fp16_qmax → 01_awq_inprocess2: observed execution identity differs；other candidate switches differ
- 05_fp16_qmax → 01_awq_inprocess2: observed execution identity differs；other candidate switches differ
- 05_fp16_qmax → 01_awq_inprocess2: observed execution identity differs；other candidate switches differ
- 02_awq_wide_all → 03_awq_wide_prefill: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 03_awq_wide_prefill: same-weight output equality not established；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 03_awq_wide_prefill: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 03_awq_wide_prefill: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 04_awq_qmax: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 04_awq_qmax: same-weight output equality not established；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 04_awq_qmax: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 02_awq_wide_all → 04_awq_qmax: Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 05_fp16_qmax → 02_awq_wide_all: observed execution identity differs；other candidate switches differ；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 05_fp16_qmax → 02_awq_wide_all: observed execution identity differs；other candidate switches differ；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 05_fp16_qmax → 02_awq_wide_all: observed execution identity differs；other candidate switches differ；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 05_fp16_qmax → 02_awq_wide_all: observed execution identity differs；other candidate switches differ；Wide native output gate not verified: logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all; status=unverified_missing_matching_native_cells; missing cells=0_64
- 05_fp16_qmax → 03_awq_wide_prefill: observed execution identity differs；other candidate switches differ
- 05_fp16_qmax → 03_awq_wide_prefill: observed execution identity differs；other candidate switches differ
- 05_fp16_qmax → 03_awq_wide_prefill: observed execution identity differs；other candidate switches differ
- 05_fp16_qmax → 03_awq_wide_prefill: observed execution identity differs；other candidate switches differ

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
- Linear IPC requires actual successful large-message send/receive receipts. Wide kernels require independent real packed-tensor hashes, numerical gates and all 144 projection routes per request.
- Wide kernels use saved exact-M configurations only; other navigation lengths retain native fallback. Microbenchmark numerical coverage uses layers0/18/35, not all layers.
- Vision QMAX requires bitwise main/three-DeepStack checks in excluded 2/9-image warmups. First-grid host receipts are not per-request GPU counters; sampled peak memory may include the diagnostic double-forward.
- Quantized/FP16 comparisons require the same vision QMAX state and numerical-validation flag; QMAX gains are reported separately as vision/system changes.
- CUDA event intervals can include dispatch gaps; decode uses actual forwards, including post-response tail work.
- Profile intervals overlap and include waiting; no exclusive overhead decomposition is claimed.
- Full checkpoint identities use prior SHA receipts; wide numerical tensors are additionally verified by independent mmap byte hashes. Audit only outside timed GPU runs.
- Small repeated fixed inputs and raw P50/P95 are descriptive; repetitions are not pooled or treated as significance tests.
- {"missing_cells":["0_64"],"reason":"Wide candidate lacks same-engine native fixed-output equivalence","session":"logs/navigation_latency_20261120/kernel_vision_screen_v1/02_awq_wide_all"}

本报告不宣称完整导航质量通过。各组实际输入tensor SHA、源码/launch SHA、原始请求和逐前向计时已在JSON逐项保留。
