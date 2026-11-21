# 量化时延研究：独立固定输入审计

状态：`audited`。保留每轮原始测量；不平均不同轮的分位数。

| 运行/配置 | 格子 | HTTP模式 | 执行器/MP引擎/间隔 | IPC补丁 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| system_confirmation_reverse_v1/gptq_w4a16_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3321 | 0.3683 | 未测 | 83.53 | 31.41 | 7.516 |
| system_confirmation_reverse_v1/gptq_w4a16_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8331 | 0.9116 | 未测 | 83.75 | 31.44 | 7.519 |
| system_confirmation_reverse_v1/gptq_w4a16_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7405 | 0.7663 | 未测 | 349.05 | 137.95 | 8.211 |
| system_confirmation_reverse_v1/gptq_w4a16_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2694 | 1.3041 | 未测 | 348.84 | 137.98 | 8.197 |
| system_confirmation_reverse_v1/awq_w4a16_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3310 | 0.3547 | 未测 | 83.49 | 31.41 | 7.529 |
| system_confirmation_reverse_v1/awq_w4a16_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8234 | 0.8575 | 未测 | 83.54 | 31.41 | 7.514 |
| system_confirmation_reverse_v1/awq_w4a16_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7387 | 0.7741 | 未测 | 348.79 | 137.81 | 8.186 |
| system_confirmation_reverse_v1/awq_w4a16_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2629 | 1.2854 | 未测 | 348.84 | 137.96 | 8.182 |
| system_confirmation_reverse_v1/fp16_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.5046 | 0.5280 | 未测 | 79.58 | 31.31 | 17.233 |
| system_confirmation_reverse_v1/fp16_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 1.4489 | 1.4782 | 未测 | 79.45 | 31.43 | 17.228 |
| system_confirmation_reverse_v1/fp16_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.8804 | 0.8986 | 未测 | 322.12 | 138.20 | 17.844 |
| system_confirmation_reverse_v1/fp16_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.8455 | 1.8916 | 未测 | 322.14 | 138.25 | 17.833 |
| system_confirmation_reverse_v1/gptq_w4a16_mp16 | 0_18 | nonstream | mp/True/16 | False | 10 | 0.3521 | 0.3563 | 未测 | 83.54 | 31.32 | 7.510 |
| system_confirmation_reverse_v1/gptq_w4a16_mp16 | 0_64 | nonstream | mp/True/16 | False | 10 | 0.7912 | 0.7984 | 未测 | 83.42 | 31.40 | 7.509 |
| system_confirmation_reverse_v1/gptq_w4a16_mp16 | 1_18 | nonstream | mp/True/16 | False | 10 | 1.5568 | 1.6283 | 未测 | 349.68 | 143.96 | 8.189 |
| system_confirmation_reverse_v1/gptq_w4a16_mp16 | 1_64 | nonstream | mp/True/16 | False | 10 | 2.0237 | 2.0470 | 未测 | 348.95 | 137.90 | 8.185 |
| system_confirmation_reverse_v1/awq_w4a16_mp16 | 0_18 | nonstream | mp/True/16 | False | 10 | 0.3572 | 0.7741 | 未测 | 83.43 | 31.33 | 7.525 |
| system_confirmation_reverse_v1/awq_w4a16_mp16 | 0_64 | nonstream | mp/True/16 | False | 10 | 0.7947 | 0.7999 | 未测 | 83.51 | 31.32 | 7.504 |
| system_confirmation_reverse_v1/awq_w4a16_mp16 | 1_18 | nonstream | mp/True/16 | False | 10 | 1.5748 | 1.6461 | 未测 | 349.18 | 144.02 | 8.198 |
| system_confirmation_reverse_v1/awq_w4a16_mp16 | 1_64 | nonstream | mp/True/16 | False | 10 | 2.0267 | 2.3965 | 未测 | 349.35 | 138.15 | 8.187 |
| system_confirmation_reverse_v1/fp16_mp16 | 0_18 | nonstream | mp/True/16 | False | 10 | 0.5247 | 0.5321 | 未测 | 79.40 | 31.36 | 17.239 |
| system_confirmation_reverse_v1/fp16_mp16 | 0_64 | nonstream | mp/True/16 | False | 10 | 1.4115 | 1.4186 | 未测 | 79.34 | 31.38 | 17.221 |
| system_confirmation_reverse_v1/fp16_mp16 | 1_18 | nonstream | mp/True/16 | False | 10 | 1.7151 | 1.7980 | 未测 | 324.38 | 143.73 | 17.867 |
| system_confirmation_reverse_v1/fp16_mp16 | 1_64 | nonstream | mp/True/16 | False | 10 | 2.6374 | 2.6552 | 未测 | 322.10 | 138.20 | 17.860 |
| system_confirmation_reverse_v1/fp16_inprocess4_long_control | 0_18 | nonstream | uni/False/4 | False | 10 | 0.5443 | 0.5713 | 未测 | 79.47 | 31.40 | 17.225 |
| system_confirmation_reverse_v1/fp16_inprocess4_long_control | 0_64 | nonstream | uni/False/4 | False | 10 | 1.4206 | 1.4273 | 未测 | 79.46 | 31.40 | 17.227 |
| system_confirmation_reverse_v1/fp16_inprocess4_long_control | 1_18 | nonstream | uni/False/4 | False | 10 | 0.9267 | 0.9489 | 未测 | 322.11 | 138.11 | 17.837 |
| system_confirmation_reverse_v1/fp16_inprocess4_long_control | 1_64 | nonstream | uni/False/4 | False | 10 | 1.8398 | 1.8627 | 未测 | 322.04 | 138.06 | 17.840 |

格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。

## 匹配比较

| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |
|---|---|---|---|---:|---:|---|
| matched_quantization | 02_fp16_inprocess2 → 00_gptq_w4a16_inprocess2 | 0_18 | 是 | 34.19% | 0.6975 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 00_gptq_w4a16_inprocess2 | 0_64 | 是 | 42.51% | 0.6167 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 00_gptq_w4a16_inprocess2 | 1_18 | 是 | 15.89% | 0.8528 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 00_gptq_w4a16_inprocess2 | 1_64 | 是 | 31.21% | 0.6894 | 不适用 |
| same_model_system_bundle | 03_gptq_w4a16_mp16 → 00_gptq_w4a16_inprocess2 | 0_18 | 是 | 5.67% | 1.0335 | True |
| same_model_system_bundle | 03_gptq_w4a16_mp16 → 00_gptq_w4a16_inprocess2 | 0_64 | 是 | -5.29% | 1.1417 | True |
| same_model_system_bundle | 03_gptq_w4a16_mp16 → 00_gptq_w4a16_inprocess2 | 1_18 | 是 | 52.43% | 0.4706 | True |
| same_model_system_bundle | 03_gptq_w4a16_mp16 → 00_gptq_w4a16_inprocess2 | 1_64 | 是 | 37.27% | 0.6371 | True |
| matched_quantization | 05_fp16_mp16 → 00_gptq_w4a16_inprocess2 | 0_18 | 否 | 不作归因 | 0.6922 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 00_gptq_w4a16_inprocess2 | 0_64 | 否 | 不作归因 | 0.6426 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 00_gptq_w4a16_inprocess2 | 1_18 | 否 | 不作归因 | 0.4262 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 00_gptq_w4a16_inprocess2 | 1_64 | 否 | 不作归因 | 0.4911 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2 | 0_18 | 否 | 不作归因 | 0.6446 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2 | 0_64 | 否 | 不作归因 | 0.6386 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2 | 1_18 | 否 | 不作归因 | 0.8076 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2 | 1_64 | 否 | 不作归因 | 0.7001 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 01_awq_w4a16_inprocess2 | 0_18 | 是 | 34.40% | 0.6719 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 01_awq_w4a16_inprocess2 | 0_64 | 是 | 43.17% | 0.5801 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 01_awq_w4a16_inprocess2 | 1_18 | 是 | 16.10% | 0.8614 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 01_awq_w4a16_inprocess2 | 1_64 | 是 | 31.57% | 0.6795 | 不适用 |
| same_model_system_bundle | 04_awq_w4a16_mp16 → 01_awq_w4a16_inprocess2 | 0_18 | 是 | 7.34% | 0.4582 | True |
| same_model_system_bundle | 04_awq_w4a16_mp16 → 01_awq_w4a16_inprocess2 | 0_64 | 是 | -3.61% | 1.0719 | True |
| same_model_system_bundle | 04_awq_w4a16_mp16 → 01_awq_w4a16_inprocess2 | 1_18 | 是 | 53.09% | 0.4703 | True |
| same_model_system_bundle | 04_awq_w4a16_mp16 → 01_awq_w4a16_inprocess2 | 1_64 | 是 | 37.69% | 0.5363 | True |
| matched_quantization | 05_fp16_mp16 → 01_awq_w4a16_inprocess2 | 0_18 | 否 | 不作归因 | 0.6667 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 01_awq_w4a16_inprocess2 | 0_64 | 否 | 不作归因 | 0.6045 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 01_awq_w4a16_inprocess2 | 1_18 | 否 | 不作归因 | 0.4305 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 01_awq_w4a16_inprocess2 | 1_64 | 否 | 不作归因 | 0.4841 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2 | 0_18 | 否 | 不作归因 | 0.6209 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2 | 0_64 | 否 | 不作归因 | 0.6008 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2 | 1_18 | 否 | 不作归因 | 0.8157 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2 | 1_64 | 否 | 不作归因 | 0.6900 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 03_gptq_w4a16_mp16 | 0_18 | 否 | 不作归因 | 0.6749 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 03_gptq_w4a16_mp16 | 0_64 | 否 | 不作归因 | 0.5401 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 03_gptq_w4a16_mp16 | 1_18 | 否 | 不作归因 | 1.8121 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 03_gptq_w4a16_mp16 | 1_64 | 否 | 不作归因 | 1.0822 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 04_awq_w4a16_mp16 | 0_18 | 否 | 不作归因 | 1.4663 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 04_awq_w4a16_mp16 | 0_64 | 否 | 不作归因 | 0.5411 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 04_awq_w4a16_mp16 | 1_18 | 否 | 不作归因 | 1.8319 | 不适用 |
| matched_quantization | 02_fp16_inprocess2 → 04_awq_w4a16_mp16 | 1_64 | 否 | 不作归因 | 1.2669 | 不适用 |
| same_model_system_bundle | 05_fp16_mp16 → 02_fp16_inprocess2 | 0_18 | 是 | 3.83% | 0.9923 | True |
| same_model_system_bundle | 05_fp16_mp16 → 02_fp16_inprocess2 | 0_64 | 是 | -2.65% | 1.0420 | True |
| same_model_system_bundle | 05_fp16_mp16 → 02_fp16_inprocess2 | 1_18 | 是 | 48.67% | 0.4998 | True |
| same_model_system_bundle | 05_fp16_mp16 → 02_fp16_inprocess2 | 1_64 | 是 | 30.03% | 0.7124 | True |
| same_model_interval_change | 02_fp16_inprocess2 → 06_fp16_inprocess4_long_control | 0_18 | 是 | -7.87% | 1.0821 | True |
| same_model_interval_change | 02_fp16_inprocess2 → 06_fp16_inprocess4_long_control | 0_64 | 是 | 1.95% | 0.9656 | True |
| same_model_interval_change | 02_fp16_inprocess2 → 06_fp16_inprocess4_long_control | 1_18 | 是 | -5.26% | 1.0560 | True |
| same_model_interval_change | 02_fp16_inprocess2 → 06_fp16_inprocess4_long_control | 1_64 | 是 | 0.31% | 0.9847 | True |
| matched_quantization | 05_fp16_mp16 → 03_gptq_w4a16_mp16 | 0_18 | 是 | 32.90% | 0.6697 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 03_gptq_w4a16_mp16 | 0_64 | 是 | 43.95% | 0.5628 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 03_gptq_w4a16_mp16 | 1_18 | 是 | 9.23% | 0.9056 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 03_gptq_w4a16_mp16 | 1_64 | 是 | 23.27% | 0.7709 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16 | 0_18 | 否 | 不作归因 | 0.6237 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16 | 0_64 | 否 | 不作归因 | 0.5594 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16 | 1_18 | 否 | 不作归因 | 1.7160 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16 | 1_64 | 否 | 不作归因 | 1.0990 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 04_awq_w4a16_mp16 | 0_18 | 是 | 31.92% | 1.4550 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 04_awq_w4a16_mp16 | 0_64 | 是 | 43.70% | 0.5639 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 04_awq_w4a16_mp16 | 1_18 | 是 | 8.18% | 0.9155 | 不适用 |
| matched_quantization | 05_fp16_mp16 → 04_awq_w4a16_mp16 | 1_64 | 是 | 23.15% | 0.9026 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16 | 0_18 | 否 | 不作归因 | 1.3551 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16 | 0_64 | 否 | 不作归因 | 0.5605 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16 | 1_18 | 否 | 不作归因 | 1.7347 | 不适用 |
| matched_quantization | 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16 | 1_64 | 否 | 不作归因 | 1.2866 | 不适用 |
| same_model_system_bundle | 05_fp16_mp16 → 06_fp16_inprocess4_long_control | 0_18 | 是 | -3.74% | 1.0737 | True |
| same_model_system_bundle | 05_fp16_mp16 → 06_fp16_inprocess4_long_control | 0_64 | 是 | -0.65% | 1.0061 | True |
| same_model_system_bundle | 05_fp16_mp16 → 06_fp16_inprocess4_long_control | 1_18 | 是 | 45.97% | 0.5278 | True |
| same_model_system_bundle | 05_fp16_mp16 → 06_fp16_inprocess4_long_control | 1_64 | 是 | 30.24% | 0.7015 | True |

不可直接比较的原因：

- 05_fp16_mp16 → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 06_fp16_inprocess4_long_control → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 05_fp16_mp16 → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 06_fp16_inprocess4_long_control → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess2 → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 06_fp16_inprocess4_long_control → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ

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

本报告不宣称完整导航质量通过。各组实际输入tensor SHA、源码/launch SHA、原始请求和逐前向计时已在JSON逐项保留。
