# 量化时延研究：独立固定输入审计

状态：`audited`。保留每轮原始测量；不平均不同轮的分位数。

| 运行/配置 | 格子 | HTTP模式 | 执行器/MP引擎/间隔 | IPC补丁 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| system_candidates_v1/fp16_linear16 | 0_18 | nonstream | mp/True/16 | True | 10 | 0.5251 | 0.5325 | 未测 | 79.57 | 31.30 | 17.241 |
| system_candidates_v1/fp16_linear16 | 0_64 | nonstream | mp/True/16 | True | 10 | 1.4151 | 1.4254 | 未测 | 79.43 | 31.45 | 17.239 |
| system_candidates_v1/fp16_linear16 | 1_18 | nonstream | mp/True/16 | True | 10 | 1.0883 | 1.1369 | 未测 | 322.50 | 137.84 | 17.869 |
| system_candidates_v1/fp16_linear16 | 1_64 | nonstream | mp/True/16 | True | 10 | 1.9859 | 2.0231 | 未测 | 322.13 | 138.12 | 17.853 |
| system_candidates_v1/fp16_uni4 | 0_18 | nonstream | uni/True/4 | False | 10 | 0.5486 | 0.5570 | 未测 | 79.42 | 31.46 | 17.239 |
| system_candidates_v1/fp16_uni4 | 0_64 | nonstream | uni/True/4 | False | 10 | 1.4134 | 1.4275 | 未测 | 79.42 | 31.48 | 17.232 |
| system_candidates_v1/fp16_uni4 | 1_18 | nonstream | uni/True/4 | False | 10 | 1.0446 | 1.0890 | 未测 | 322.06 | 138.05 | 17.852 |
| system_candidates_v1/fp16_uni4 | 1_64 | nonstream | uni/True/4 | False | 10 | 1.9193 | 1.9366 | 未测 | 322.21 | 137.90 | 17.850 |
| system_candidates_v1/fp16_inprocess1 | 0_18 | nonstream | uni/False/1 | False | 10 | 0.5195 | 0.5563 | 未测 | 79.51 | 31.48 | 17.236 |
| system_candidates_v1/fp16_inprocess1 | 0_64 | nonstream | uni/False/1 | False | 10 | 1.5392 | 1.5625 | 未测 | 79.51 | 31.45 | 17.244 |
| system_candidates_v1/fp16_inprocess1 | 1_18 | nonstream | uni/False/1 | False | 10 | 0.9087 | 0.9446 | 未测 | 322.31 | 137.87 | 17.852 |
| system_candidates_v1/fp16_inprocess1 | 1_64 | nonstream | uni/False/1 | False | 10 | 1.9722 | 2.0016 | 未测 | 322.10 | 137.84 | 17.841 |
| system_candidates_v1/fp16_inprocess4 | 0_18 | nonstream | uni/False/4 | False | 10 | 0.5453 | 0.5586 | 未测 | 79.48 | 31.40 | 17.257 |
| system_candidates_v1/fp16_inprocess4 | 0_64 | nonstream | uni/False/4 | False | 10 | 1.4227 | 1.4435 | 未测 | 79.58 | 31.38 | 17.253 |
| system_candidates_v1/fp16_inprocess4 | 1_18 | nonstream | uni/False/4 | False | 10 | 0.9317 | 0.9435 | 未测 | 322.33 | 138.10 | 17.883 |
| system_candidates_v1/fp16_inprocess4 | 1_64 | nonstream | uni/False/4 | False | 10 | 1.8404 | 1.8616 | 未测 | 322.39 | 138.00 | 17.858 |
| system_candidates_v1/fp16_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.4994 | 0.5128 | 未测 | 79.48 | 31.44 | 17.238 |
| system_candidates_v1/fp16_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 1.4755 | 1.5006 | 未测 | 79.39 | 31.45 | 17.246 |
| system_candidates_v1/fp16_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.8803 | 0.8969 | 未测 | 322.38 | 137.89 | 17.849 |
| system_candidates_v1/fp16_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.8787 | 1.8985 | 未测 | 322.16 | 137.94 | 17.859 |
| system_candidates_v1/awq_inprocess4 | 0_18 | nonstream | uni/False/4 | False | 10 | 0.3489 | 0.3791 | 未测 | 83.69 | 31.48 | 7.511 |
| system_candidates_v1/awq_inprocess4 | 0_64 | nonstream | uni/False/4 | False | 10 | 0.7968 | 0.8342 | 未测 | 83.71 | 31.43 | 7.515 |
| system_candidates_v1/awq_inprocess4 | 1_18 | nonstream | uni/False/4 | False | 10 | 0.7639 | 0.7747 | 未测 | 348.89 | 137.76 | 8.190 |
| system_candidates_v1/awq_inprocess4 | 1_64 | nonstream | uni/False/4 | False | 10 | 1.2380 | 1.2719 | 未测 | 349.04 | 137.83 | 8.189 |
| system_candidates_v1/awq_inprocess1 | 0_18 | nonstream | uni/False/1 | False | 10 | 0.3474 | 0.3621 | 未测 | 83.67 | 31.37 | 7.504 |
| system_candidates_v1/awq_inprocess1 | 0_64 | nonstream | uni/False/1 | False | 10 | 0.9037 | 0.9656 | 未测 | 83.49 | 31.40 | 7.500 |
| system_candidates_v1/awq_inprocess1 | 1_18 | nonstream | uni/False/1 | False | 10 | 0.7868 | 0.8713 | 未测 | 348.95 | 137.84 | 8.204 |
| system_candidates_v1/awq_inprocess1 | 1_64 | nonstream | uni/False/1 | False | 10 | 1.3618 | 1.4032 | 未测 | 348.89 | 137.76 | 8.176 |
| system_candidates_v1/awq_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3346 | 0.3624 | 未测 | 83.67 | 31.37 | 7.521 |
| system_candidates_v1/awq_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8262 | 0.9240 | 未测 | 83.57 | 31.37 | 7.505 |
| system_candidates_v1/awq_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7425 | 0.8002 | 未测 | 348.90 | 137.73 | 8.206 |
| system_candidates_v1/awq_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2707 | 1.3374 | 未测 | 348.78 | 137.90 | 8.178 |
| system_candidates_v1/awq_linear16 | 0_18 | nonstream | mp/True/16 | True | 10 | 0.3581 | 0.3627 | 未测 | 83.42 | 31.38 | 7.534 |
| system_candidates_v1/awq_linear16 | 0_64 | nonstream | mp/True/16 | True | 10 | 0.7939 | 0.8044 | 未测 | 83.56 | 31.36 | 7.524 |
| system_candidates_v1/awq_linear16 | 1_18 | nonstream | mp/True/16 | True | 10 | 0.9245 | 0.9627 | 未测 | 349.20 | 137.96 | 8.202 |
| system_candidates_v1/awq_linear16 | 1_64 | nonstream | mp/True/16 | True | 10 | 1.3878 | 1.4032 | 未测 | 349.54 | 137.88 | 8.195 |
| system_candidates_v1/gptq_linear16 | 0_18 | nonstream | mp/True/16 | True | 10 | 0.3529 | 0.3582 | 未测 | 83.47 | 31.37 | 7.527 |
| system_candidates_v1/gptq_linear16 | 0_64 | nonstream | mp/True/16 | True | 10 | 0.7914 | 0.7988 | 未测 | 83.51 | 31.43 | 7.516 |
| system_candidates_v1/gptq_linear16 | 1_18 | nonstream | mp/True/16 | True | 10 | 0.9193 | 0.9619 | 未测 | 348.99 | 138.11 | 8.202 |
| system_candidates_v1/gptq_linear16 | 1_64 | nonstream | mp/True/16 | True | 10 | 1.3873 | 1.4075 | 未测 | 349.06 | 137.89 | 8.198 |
| system_candidates_v1/gptq_inprocess4 | 0_18 | nonstream | uni/False/4 | False | 10 | 0.3526 | 0.3751 | 未测 | 83.49 | 31.36 | 7.528 |
| system_candidates_v1/gptq_inprocess4 | 0_64 | nonstream | uni/False/4 | False | 10 | 0.7962 | 0.8223 | 未测 | 83.44 | 31.36 | 7.515 |
| system_candidates_v1/gptq_inprocess4 | 1_18 | nonstream | uni/False/4 | False | 10 | 0.7638 | 0.7807 | 未测 | 349.30 | 137.82 | 8.222 |
| system_candidates_v1/gptq_inprocess4 | 1_64 | nonstream | uni/False/4 | False | 10 | 1.2379 | 1.2583 | 未测 | 349.25 | 137.88 | 8.190 |
| system_candidates_v1/gptq_inprocess1 | 0_18 | nonstream | uni/False/1 | False | 10 | 0.3519 | 0.3640 | 未测 | 83.34 | 31.51 | 7.519 |
| system_candidates_v1/gptq_inprocess1 | 0_64 | nonstream | uni/False/1 | False | 10 | 0.8820 | 0.9707 | 未测 | 83.51 | 31.40 | 7.498 |
| system_candidates_v1/gptq_inprocess1 | 1_18 | nonstream | uni/False/1 | False | 10 | 0.7604 | 0.7764 | 未测 | 349.39 | 138.17 | 8.165 |
| system_candidates_v1/gptq_inprocess1 | 1_64 | nonstream | uni/False/1 | False | 10 | 1.3451 | 1.4017 | 未测 | 349.23 | 138.00 | 8.171 |
| system_candidates_v1/gptq_inprocess2 | 0_18 | nonstream | uni/False/2 | False | 10 | 0.3335 | 0.3508 | 未测 | 83.70 | 31.35 | 7.517 |
| system_candidates_v1/gptq_inprocess2 | 0_64 | nonstream | uni/False/2 | False | 10 | 0.8331 | 0.8742 | 未测 | 83.52 | 31.36 | 7.533 |
| system_candidates_v1/gptq_inprocess2 | 1_18 | nonstream | uni/False/2 | False | 10 | 0.7414 | 0.7517 | 未测 | 349.29 | 137.92 | 8.223 |
| system_candidates_v1/gptq_inprocess2 | 1_64 | nonstream | uni/False/2 | False | 10 | 1.2738 | 1.3042 | 未测 | 349.42 | 138.12 | 8.209 |

格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。

## 匹配比较

| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |
|---|---|---|---|---:|---:|---|
| same_model_system_bundle | 00_fp16_linear16 → 01_fp16_uni4 | 0_18 | 是 | -4.48% | 1.0459 | True |
| same_model_system_bundle | 00_fp16_linear16 → 01_fp16_uni4 | 0_64 | 是 | 0.12% | 1.0015 | True |
| same_model_system_bundle | 00_fp16_linear16 → 01_fp16_uni4 | 1_18 | 是 | 4.02% | 0.9579 | True |
| same_model_system_bundle | 00_fp16_linear16 → 01_fp16_uni4 | 1_64 | 是 | 3.35% | 0.9573 | True |
| same_model_system_bundle | 00_fp16_linear16 → 02_fp16_inprocess1 | 0_18 | 是 | 1.06% | 1.0446 | True |
| same_model_system_bundle | 00_fp16_linear16 → 02_fp16_inprocess1 | 0_64 | 是 | -8.76% | 1.0962 | True |
| same_model_system_bundle | 00_fp16_linear16 → 02_fp16_inprocess1 | 1_18 | 是 | 16.50% | 0.8309 | True |
| same_model_system_bundle | 00_fp16_linear16 → 02_fp16_inprocess1 | 1_64 | 是 | 0.69% | 0.9894 | True |
| same_model_system_bundle | 00_fp16_linear16 → 03_fp16_inprocess4 | 0_18 | 是 | -3.84% | 1.0489 | True |
| same_model_system_bundle | 00_fp16_linear16 → 03_fp16_inprocess4 | 0_64 | 是 | -0.53% | 1.0127 | True |
| same_model_system_bundle | 00_fp16_linear16 → 03_fp16_inprocess4 | 1_18 | 是 | 14.39% | 0.8299 | True |
| same_model_system_bundle | 00_fp16_linear16 → 03_fp16_inprocess4 | 1_64 | 是 | 7.33% | 0.9202 | True |
| same_model_system_bundle | 00_fp16_linear16 → 04_fp16_inprocess2 | 0_18 | 是 | 4.90% | 0.9629 | True |
| same_model_system_bundle | 00_fp16_linear16 → 04_fp16_inprocess2 | 0_64 | 是 | -4.27% | 1.0528 | True |
| same_model_system_bundle | 00_fp16_linear16 → 04_fp16_inprocess2 | 1_18 | 是 | 19.11% | 0.7889 | True |
| same_model_system_bundle | 00_fp16_linear16 → 04_fp16_inprocess2 | 1_64 | 是 | 5.40% | 0.9384 | True |
| matched_quantization | 00_fp16_linear16 → 05_awq_inprocess4 | 0_18 | 否 | 不作归因 | 0.7119 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 05_awq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5853 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 05_awq_inprocess4 | 1_18 | 否 | 不作归因 | 0.6815 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 05_awq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6287 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 06_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.6799 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 06_awq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6774 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 06_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.7664 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 06_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.6936 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 07_awq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6806 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 07_awq_inprocess2 | 0_64 | 否 | 不作归因 | 0.6482 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 07_awq_inprocess2 | 1_18 | 否 | 不作归因 | 0.7039 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 07_awq_inprocess2 | 1_64 | 否 | 不作归因 | 0.6611 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 08_awq_linear16 | 0_18 | 是 | 31.81% | 0.6811 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 08_awq_linear16 | 0_64 | 是 | 43.90% | 0.5643 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 08_awq_linear16 | 1_18 | 是 | 15.06% | 0.8468 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 08_awq_linear16 | 1_64 | 是 | 30.11% | 0.6936 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 09_gptq_linear16 | 0_18 | 是 | 32.79% | 0.6725 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 09_gptq_linear16 | 0_64 | 是 | 44.08% | 0.5604 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 09_gptq_linear16 | 1_18 | 是 | 15.53% | 0.8460 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 09_gptq_linear16 | 1_64 | 是 | 30.14% | 0.6957 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 10_gptq_inprocess4 | 0_18 | 否 | 不作归因 | 0.7044 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 10_gptq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5769 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 10_gptq_inprocess4 | 1_18 | 否 | 不作归因 | 0.6867 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 10_gptq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6220 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 11_gptq_inprocess1 | 0_18 | 否 | 不作归因 | 0.6836 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 11_gptq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6810 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 11_gptq_inprocess1 | 1_18 | 否 | 不作归因 | 0.6829 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 11_gptq_inprocess1 | 1_64 | 否 | 不作归因 | 0.6929 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 12_gptq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6586 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 12_gptq_inprocess2 | 0_64 | 否 | 不作归因 | 0.6133 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 12_gptq_inprocess2 | 1_18 | 否 | 不作归因 | 0.6612 | 不适用 |
| matched_quantization | 00_fp16_linear16 → 12_gptq_inprocess2 | 1_64 | 否 | 不作归因 | 0.6446 | 不适用 |
| same_model_system_bundle | 01_fp16_uni4 → 02_fp16_inprocess1 | 0_18 | 是 | 5.30% | 0.9987 | True |
| same_model_system_bundle | 01_fp16_uni4 → 02_fp16_inprocess1 | 0_64 | 是 | -8.90% | 1.0945 | True |
| same_model_system_bundle | 01_fp16_uni4 → 02_fp16_inprocess1 | 1_18 | 是 | 13.01% | 0.8674 | True |
| same_model_system_bundle | 01_fp16_uni4 → 02_fp16_inprocess1 | 1_64 | 是 | -2.75% | 1.0336 | True |
| same_model_inprocess_change | 01_fp16_uni4 → 03_fp16_inprocess4 | 0_18 | 是 | 0.61% | 1.0028 | True |
| same_model_inprocess_change | 01_fp16_uni4 → 03_fp16_inprocess4 | 0_64 | 是 | -0.66% | 1.0112 | True |
| same_model_inprocess_change | 01_fp16_uni4 → 03_fp16_inprocess4 | 1_18 | 是 | 10.81% | 0.8664 | True |
| same_model_inprocess_change | 01_fp16_uni4 → 03_fp16_inprocess4 | 1_64 | 是 | 4.11% | 0.9613 | True |
| same_model_system_bundle | 01_fp16_uni4 → 04_fp16_inprocess2 | 0_18 | 是 | 8.97% | 0.9206 | True |
| same_model_system_bundle | 01_fp16_uni4 → 04_fp16_inprocess2 | 0_64 | 是 | -4.40% | 1.0512 | True |
| same_model_system_bundle | 01_fp16_uni4 → 04_fp16_inprocess2 | 1_18 | 是 | 15.72% | 0.8236 | True |
| same_model_system_bundle | 01_fp16_uni4 → 04_fp16_inprocess2 | 1_64 | 是 | 2.12% | 0.9803 | True |
| matched_quantization | 01_fp16_uni4 → 05_awq_inprocess4 | 0_18 | 否 | 不作归因 | 0.6806 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 05_awq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5844 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 05_awq_inprocess4 | 1_18 | 否 | 不作归因 | 0.7114 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 05_awq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6568 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 06_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.6501 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 06_awq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6764 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 06_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.8001 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 06_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.7245 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 07_awq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6507 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 07_awq_inprocess2 | 0_64 | 否 | 不作归因 | 0.6473 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 07_awq_inprocess2 | 1_18 | 否 | 不作归因 | 0.7348 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 07_awq_inprocess2 | 1_64 | 否 | 不作归因 | 0.6906 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 08_awq_linear16 | 0_18 | 否 | 不作归因 | 0.6512 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 08_awq_linear16 | 0_64 | 否 | 不作归因 | 0.5635 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 08_awq_linear16 | 1_18 | 否 | 不作归因 | 0.8840 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 08_awq_linear16 | 1_64 | 否 | 不作归因 | 0.7246 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 09_gptq_linear16 | 0_18 | 否 | 不作归因 | 0.6430 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 09_gptq_linear16 | 0_64 | 否 | 不作归因 | 0.5596 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 09_gptq_linear16 | 1_18 | 否 | 不作归因 | 0.8833 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 09_gptq_linear16 | 1_64 | 否 | 不作归因 | 0.7268 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 10_gptq_inprocess4 | 0_18 | 否 | 不作归因 | 0.6735 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 10_gptq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5760 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 10_gptq_inprocess4 | 1_18 | 否 | 不作归因 | 0.7169 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 10_gptq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6497 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 11_gptq_inprocess1 | 0_18 | 否 | 不作归因 | 0.6535 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 11_gptq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6800 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 11_gptq_inprocess1 | 1_18 | 否 | 不作归因 | 0.7130 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 11_gptq_inprocess1 | 1_64 | 否 | 不作归因 | 0.7238 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 12_gptq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6297 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 12_gptq_inprocess2 | 0_64 | 否 | 不作归因 | 0.6124 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 12_gptq_inprocess2 | 1_18 | 否 | 不作归因 | 0.6902 | 不适用 |
| matched_quantization | 01_fp16_uni4 → 12_gptq_inprocess2 | 1_64 | 否 | 不作归因 | 0.6734 | 不适用 |
| same_model_interval_change | 02_fp16_inprocess1 → 03_fp16_inprocess4 | 0_18 | 是 | -4.95% | 1.0041 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 03_fp16_inprocess4 | 0_64 | 是 | 7.57% | 0.9239 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 03_fp16_inprocess4 | 1_18 | 是 | -2.52% | 0.9988 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 03_fp16_inprocess4 | 1_64 | 是 | 6.68% | 0.9301 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 04_fp16_inprocess2 | 0_18 | 是 | 3.88% | 0.9218 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 04_fp16_inprocess2 | 0_64 | 是 | 4.14% | 0.9604 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 04_fp16_inprocess2 | 1_18 | 是 | 3.12% | 0.9495 | True |
| same_model_interval_change | 02_fp16_inprocess1 → 04_fp16_inprocess2 | 1_64 | 是 | 4.74% | 0.9485 | True |
| matched_quantization | 02_fp16_inprocess1 → 05_awq_inprocess4 | 0_18 | 否 | 不作归因 | 0.6815 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 05_awq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5339 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 05_awq_inprocess4 | 1_18 | 否 | 不作归因 | 0.8202 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 05_awq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6354 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 06_awq_inprocess1 | 0_18 | 是 | 33.14% | 0.6509 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 06_awq_inprocess1 | 0_64 | 是 | 41.29% | 0.6180 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 06_awq_inprocess1 | 1_18 | 是 | 13.42% | 0.9224 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 06_awq_inprocess1 | 1_64 | 是 | 30.95% | 0.7010 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 07_awq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6515 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 07_awq_inprocess2 | 0_64 | 否 | 不作归因 | 0.5914 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 07_awq_inprocess2 | 1_18 | 否 | 不作归因 | 0.8471 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 07_awq_inprocess2 | 1_64 | 否 | 不作归因 | 0.6681 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 08_awq_linear16 | 0_18 | 否 | 不作归因 | 0.6520 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 08_awq_linear16 | 0_64 | 否 | 不作归因 | 0.5148 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 08_awq_linear16 | 1_18 | 否 | 不作归因 | 1.0192 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 08_awq_linear16 | 1_64 | 否 | 不作归因 | 0.7010 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 09_gptq_linear16 | 0_18 | 否 | 不作归因 | 0.6438 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 09_gptq_linear16 | 0_64 | 否 | 不作归因 | 0.5113 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 09_gptq_linear16 | 1_18 | 否 | 不作归因 | 1.0183 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 09_gptq_linear16 | 1_64 | 否 | 不作归因 | 0.7032 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 10_gptq_inprocess4 | 0_18 | 否 | 不作归因 | 0.6743 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 10_gptq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5263 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 10_gptq_inprocess4 | 1_18 | 否 | 不作归因 | 0.8265 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 10_gptq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6286 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 11_gptq_inprocess1 | 0_18 | 是 | 32.27% | 0.6544 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 11_gptq_inprocess1 | 0_64 | 是 | 42.70% | 0.6212 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 11_gptq_inprocess1 | 1_18 | 是 | 16.32% | 0.8219 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 11_gptq_inprocess1 | 1_64 | 是 | 31.80% | 0.7003 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 12_gptq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6305 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 12_gptq_inprocess2 | 0_64 | 否 | 不作归因 | 0.5595 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 12_gptq_inprocess2 | 1_18 | 否 | 不作归因 | 0.7957 | 不适用 |
| matched_quantization | 02_fp16_inprocess1 → 12_gptq_inprocess2 | 1_64 | 否 | 不作归因 | 0.6516 | 不适用 |
| same_model_interval_change | 03_fp16_inprocess4 → 04_fp16_inprocess2 | 0_18 | 是 | 8.42% | 0.9180 | True |
| same_model_interval_change | 03_fp16_inprocess4 → 04_fp16_inprocess2 | 0_64 | 是 | -3.71% | 1.0396 | True |
| same_model_interval_change | 03_fp16_inprocess4 → 04_fp16_inprocess2 | 1_18 | 是 | 5.51% | 0.9507 | True |
| same_model_interval_change | 03_fp16_inprocess4 → 04_fp16_inprocess2 | 1_64 | 是 | -2.08% | 1.0198 | True |
| matched_quantization | 03_fp16_inprocess4 → 05_awq_inprocess4 | 0_18 | 是 | 36.00% | 0.6787 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 05_awq_inprocess4 | 0_64 | 是 | 43.99% | 0.5779 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 05_awq_inprocess4 | 1_18 | 是 | 18.01% | 0.8212 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 05_awq_inprocess4 | 1_64 | 是 | 32.73% | 0.6832 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 06_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.6483 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 06_awq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6689 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 06_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.9235 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 06_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.7537 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 07_awq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6489 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 07_awq_inprocess2 | 0_64 | 否 | 不作归因 | 0.6401 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 07_awq_inprocess2 | 1_18 | 否 | 不作归因 | 0.8482 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 07_awq_inprocess2 | 1_64 | 否 | 不作归因 | 0.7184 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 08_awq_linear16 | 0_18 | 否 | 不作归因 | 0.6494 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 08_awq_linear16 | 0_64 | 否 | 不作归因 | 0.5572 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 08_awq_linear16 | 1_18 | 否 | 不作归因 | 1.0204 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 08_awq_linear16 | 1_64 | 否 | 不作归因 | 0.7537 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 09_gptq_linear16 | 0_18 | 否 | 不作归因 | 0.6412 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 09_gptq_linear16 | 0_64 | 否 | 不作归因 | 0.5534 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 09_gptq_linear16 | 1_18 | 否 | 不作归因 | 1.0195 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 09_gptq_linear16 | 1_64 | 否 | 不作归因 | 0.7561 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 10_gptq_inprocess4 | 0_18 | 是 | 35.34% | 0.6716 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 10_gptq_inprocess4 | 0_64 | 是 | 44.04% | 0.5696 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 10_gptq_inprocess4 | 1_18 | 是 | 18.02% | 0.8275 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 10_gptq_inprocess4 | 1_64 | 是 | 32.74% | 0.6759 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 11_gptq_inprocess1 | 0_18 | 否 | 不作归因 | 0.6517 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 11_gptq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6724 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 11_gptq_inprocess1 | 1_18 | 否 | 不作归因 | 0.8229 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 11_gptq_inprocess1 | 1_64 | 否 | 不作归因 | 0.7529 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 12_gptq_inprocess2 | 0_18 | 否 | 不作归因 | 0.6280 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 12_gptq_inprocess2 | 0_64 | 否 | 不作归因 | 0.6056 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 12_gptq_inprocess2 | 1_18 | 否 | 不作归因 | 0.7967 | 不适用 |
| matched_quantization | 03_fp16_inprocess4 → 12_gptq_inprocess2 | 1_64 | 否 | 不作归因 | 0.7005 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 05_awq_inprocess4 | 0_18 | 否 | 不作归因 | 0.7393 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 05_awq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5559 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 05_awq_inprocess4 | 1_18 | 否 | 不作归因 | 0.8638 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 05_awq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6699 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 06_awq_inprocess1 | 0_18 | 否 | 不作归因 | 0.7061 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 06_awq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6434 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 06_awq_inprocess1 | 1_18 | 否 | 不作归因 | 0.9714 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 06_awq_inprocess1 | 1_64 | 否 | 不作归因 | 0.7391 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 07_awq_inprocess2 | 0_18 | 是 | 33.00% | 0.7068 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 07_awq_inprocess2 | 0_64 | 是 | 44.01% | 0.6157 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 07_awq_inprocess2 | 1_18 | 是 | 15.66% | 0.8921 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 07_awq_inprocess2 | 1_64 | 是 | 32.36% | 0.7044 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 08_awq_linear16 | 0_18 | 否 | 不作归因 | 0.7074 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 08_awq_linear16 | 0_64 | 否 | 不作归因 | 0.5360 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 08_awq_linear16 | 1_18 | 否 | 不作归因 | 1.0733 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 08_awq_linear16 | 1_64 | 否 | 不作归因 | 0.7391 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 09_gptq_linear16 | 0_18 | 否 | 不作归因 | 0.6985 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 09_gptq_linear16 | 0_64 | 否 | 不作归因 | 0.5323 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 09_gptq_linear16 | 1_18 | 否 | 不作归因 | 1.0724 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 09_gptq_linear16 | 1_64 | 否 | 不作归因 | 0.7414 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 10_gptq_inprocess4 | 0_18 | 否 | 不作归因 | 0.7315 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 10_gptq_inprocess4 | 0_64 | 否 | 不作归因 | 0.5479 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 10_gptq_inprocess4 | 1_18 | 否 | 不作归因 | 0.8704 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 10_gptq_inprocess4 | 1_64 | 否 | 不作归因 | 0.6628 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 11_gptq_inprocess1 | 0_18 | 否 | 不作归因 | 0.7099 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 11_gptq_inprocess1 | 0_64 | 否 | 不作归因 | 0.6468 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 11_gptq_inprocess1 | 1_18 | 否 | 不作归因 | 0.8656 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 11_gptq_inprocess1 | 1_64 | 否 | 不作归因 | 0.7383 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 12_gptq_inprocess2 | 0_18 | 是 | 33.22% | 0.6840 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 12_gptq_inprocess2 | 0_64 | 是 | 43.54% | 0.5825 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 12_gptq_inprocess2 | 1_18 | 是 | 15.79% | 0.8380 | 不适用 |
| matched_quantization | 04_fp16_inprocess2 → 12_gptq_inprocess2 | 1_64 | 是 | 32.20% | 0.6869 | 不适用 |
| same_model_interval_change | 05_awq_inprocess4 → 06_awq_inprocess1 | 0_18 | 是 | 0.45% | 0.9551 | True |
| same_model_interval_change | 05_awq_inprocess4 → 06_awq_inprocess1 | 0_64 | 是 | -13.41% | 1.1574 | True |
| same_model_interval_change | 05_awq_inprocess4 → 06_awq_inprocess1 | 1_18 | 是 | -3.00% | 1.1246 | True |
| same_model_interval_change | 05_awq_inprocess4 → 06_awq_inprocess1 | 1_64 | 是 | -10.00% | 1.1032 | True |
| same_model_interval_change | 05_awq_inprocess4 → 07_awq_inprocess2 | 0_18 | 是 | 4.11% | 0.9560 | True |
| same_model_interval_change | 05_awq_inprocess4 → 07_awq_inprocess2 | 0_64 | 是 | -3.68% | 1.1076 | True |
| same_model_interval_change | 05_awq_inprocess4 → 07_awq_inprocess2 | 1_18 | 是 | 2.80% | 1.0329 | True |
| same_model_interval_change | 05_awq_inprocess4 → 07_awq_inprocess2 | 1_64 | 是 | -2.64% | 1.0515 | True |
| same_model_system_bundle | 08_awq_linear16 → 05_awq_inprocess4 | 0_18 | 是 | 2.55% | 1.0452 | True |
| same_model_system_bundle | 08_awq_linear16 → 05_awq_inprocess4 | 0_64 | 是 | -0.37% | 1.0371 | True |
| same_model_system_bundle | 08_awq_linear16 → 05_awq_inprocess4 | 1_18 | 是 | 17.37% | 0.8047 | True |
| same_model_system_bundle | 08_awq_linear16 → 05_awq_inprocess4 | 1_64 | 是 | 10.80% | 0.9064 | True |
| same_model_interval_change | 06_awq_inprocess1 → 07_awq_inprocess2 | 0_18 | 是 | 3.68% | 1.0009 | True |
| same_model_interval_change | 06_awq_inprocess1 → 07_awq_inprocess2 | 0_64 | 是 | 8.58% | 0.9569 | True |
| same_model_interval_change | 06_awq_inprocess1 → 07_awq_inprocess2 | 1_18 | 是 | 5.63% | 0.9184 | True |
| same_model_interval_change | 06_awq_inprocess1 → 07_awq_inprocess2 | 1_64 | 是 | 6.69% | 0.9531 | True |
| same_model_system_bundle | 08_awq_linear16 → 06_awq_inprocess1 | 0_18 | 是 | 2.98% | 0.9982 | True |
| same_model_system_bundle | 08_awq_linear16 → 06_awq_inprocess1 | 0_64 | 是 | -13.83% | 1.2004 | True |
| same_model_system_bundle | 08_awq_linear16 → 06_awq_inprocess1 | 1_18 | 是 | 14.89% | 0.9050 | True |
| same_model_system_bundle | 08_awq_linear16 → 06_awq_inprocess1 | 1_64 | 是 | 1.88% | 1.0000 | True |
| same_model_system_bundle | 08_awq_linear16 → 07_awq_inprocess2 | 0_18 | 是 | 6.55% | 0.9992 | True |
| same_model_system_bundle | 08_awq_linear16 → 07_awq_inprocess2 | 0_64 | 是 | -4.07% | 1.1487 | True |
| same_model_system_bundle | 08_awq_linear16 → 07_awq_inprocess2 | 1_18 | 是 | 19.68% | 0.8312 | True |
| same_model_system_bundle | 08_awq_linear16 → 07_awq_inprocess2 | 1_64 | 是 | 8.44% | 0.9531 | True |
| same_model_system_bundle | 09_gptq_linear16 → 10_gptq_inprocess4 | 0_18 | 是 | 0.10% | 1.0474 | True |
| same_model_system_bundle | 09_gptq_linear16 → 10_gptq_inprocess4 | 0_64 | 是 | -0.60% | 1.0294 | True |
| same_model_system_bundle | 09_gptq_linear16 → 10_gptq_inprocess4 | 1_18 | 是 | 16.91% | 0.8117 | True |
| same_model_system_bundle | 09_gptq_linear16 → 10_gptq_inprocess4 | 1_64 | 是 | 10.77% | 0.8940 | True |
| same_model_system_bundle | 09_gptq_linear16 → 11_gptq_inprocess1 | 0_18 | 是 | 0.29% | 1.0164 | True |
| same_model_system_bundle | 09_gptq_linear16 → 11_gptq_inprocess1 | 0_64 | 是 | -11.45% | 1.2151 | True |
| same_model_system_bundle | 09_gptq_linear16 → 11_gptq_inprocess1 | 1_18 | 是 | 17.28% | 0.8072 | True |
| same_model_system_bundle | 09_gptq_linear16 → 11_gptq_inprocess1 | 1_64 | 是 | 3.04% | 0.9959 | True |
| same_model_system_bundle | 09_gptq_linear16 → 12_gptq_inprocess2 | 0_18 | 是 | 5.50% | 0.9793 | True |
| same_model_system_bundle | 09_gptq_linear16 → 12_gptq_inprocess2 | 0_64 | 是 | -5.27% | 1.0943 | True |
| same_model_system_bundle | 09_gptq_linear16 → 12_gptq_inprocess2 | 1_18 | 是 | 19.35% | 0.7815 | True |
| same_model_system_bundle | 09_gptq_linear16 → 12_gptq_inprocess2 | 1_64 | 是 | 8.18% | 0.9266 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 11_gptq_inprocess1 | 0_18 | 是 | 0.20% | 0.9704 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 11_gptq_inprocess1 | 0_64 | 是 | -10.78% | 1.1804 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 11_gptq_inprocess1 | 1_18 | 是 | 0.44% | 0.9945 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 11_gptq_inprocess1 | 1_64 | 是 | -8.66% | 1.1140 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 12_gptq_inprocess2 | 0_18 | 是 | 5.41% | 0.9351 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 12_gptq_inprocess2 | 0_64 | 是 | -4.63% | 1.0631 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 12_gptq_inprocess2 | 1_18 | 是 | 2.94% | 0.9628 | True |
| same_model_interval_change | 10_gptq_inprocess4 → 12_gptq_inprocess2 | 1_64 | 是 | -2.90% | 1.0365 | True |
| same_model_interval_change | 11_gptq_inprocess1 → 12_gptq_inprocess2 | 0_18 | 是 | 5.22% | 0.9635 | True |
| same_model_interval_change | 11_gptq_inprocess1 → 12_gptq_inprocess2 | 0_64 | 是 | 5.55% | 0.9006 | True |
| same_model_interval_change | 11_gptq_inprocess1 → 12_gptq_inprocess2 | 1_18 | 是 | 2.51% | 0.9681 | True |
| same_model_interval_change | 11_gptq_inprocess1 → 12_gptq_inprocess2 | 1_64 | 是 | 5.30% | 0.9304 | True |

不可直接比较的原因：

- 00_fp16_linear16 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 00_fp16_linear16 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_fp16_uni4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess1 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 02_fp16_inprocess1 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 07_awq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 03_fp16_inprocess4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 03_fp16_inprocess4 → 12_gptq_inprocess2: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 05_awq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 06_awq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 08_awq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 09_gptq_linear16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 04_fp16_inprocess2 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 10_gptq_inprocess4: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs
- 04_fp16_inprocess2 → 11_gptq_inprocess1: engine configuration differs；observed execution identity differs

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
