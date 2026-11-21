# 343条真实历史输入：独立配对时延审计

状态：`audited`。每配置完整单次回放，未反馈输出，不是导航成功率评测。

| 配置 | HTTP总计/s | Mean/s | P50/s | P95/s | 实际decode次数 | Client gap总计/s |
|---|---:|---:|---:|---:|---:|---:|
| gptq_w4a16_inprocess2 | 255.513 | 0.7449 | 0.7476 | 0.7763 | 6174 | 0.092 |
| awq_w4a16_inprocess2 | 256.757 | 0.7486 | 0.7480 | 0.7789 | 6174 | 0.091 |
| fp16_inprocess2 | 307.024 | 0.8951 | 0.8934 | 0.9204 | 6174 | 0.087 |
| gptq_w4a16_mp16 | 529.717 | 1.5444 | 1.5879 | 1.6199 | 10976 | 0.089 |
| awq_w4a16_mp16 | 526.525 | 1.5351 | 1.5848 | 1.6204 | 10976 | 0.087 |
| fp16_mp16 | 629.793 | 1.8361 | 1.8676 | 1.9052 | 10976 | 0.080 |

## 配对比较

| 类型/配置 | 可归因 | 总HTTP降低 | 同输入配对平均减少/ms | 配对P50减少/ms | 变快/变慢请求 | 同权重输出 |
|---|---|---:|---:|---:|---:|---|
| matched_quantization_methods: 01_awq_w4a16_inprocess2 → 00_gptq_w4a16_inprocess2 | 是 | 0.48% | 3.63 | 1.66 | 184/159 | 跨权重不作同等要求 |
| matched_quantization: 02_fp16_inprocess2 → 00_gptq_w4a16_inprocess2 | 是 | 16.78% | 150.18 | 143.78 | 340/3 | 跨权重不作同等要求 |
| same_weight_system_change: 03_gptq_w4a16_mp16 → 00_gptq_w4a16_inprocess2 | 是 | 51.76% | 799.43 | 838.75 | 342/1 | True |
| matched_quantization_methods: 04_awq_w4a16_mp16 → 00_gptq_w4a16_inprocess2 | 否 | 禁用收益声明 | 保留观测 | 保留观测 | 342/1 | 跨权重不作同等要求 |
| matched_quantization: 05_fp16_mp16 → 00_gptq_w4a16_inprocess2 | 否 | 禁用收益声明 | 保留观测 | 保留观测 | 342/1 | 跨权重不作同等要求 |
| matched_quantization: 02_fp16_inprocess2 → 01_awq_w4a16_inprocess2 | 是 | 16.37% | 146.55 | 143.42 | 341/2 | 跨权重不作同等要求 |
| matched_quantization_methods: 01_awq_w4a16_inprocess2 → 03_gptq_w4a16_mp16 | 否 | 禁用收益声明 | 保留观测 | 保留观测 | 2/341 | 跨权重不作同等要求 |
| same_weight_system_change: 04_awq_w4a16_mp16 → 01_awq_w4a16_inprocess2 | 是 | 51.24% | 786.49 | 836.00 | 341/2 | True |
| matched_quantization: 05_fp16_mp16 → 01_awq_w4a16_inprocess2 | 否 | 禁用收益声明 | 保留观测 | 保留观测 | 341/2 | 跨权重不作同等要求 |
| matched_quantization: 02_fp16_inprocess2 → 03_gptq_w4a16_mp16 | 否 | 禁用收益声明 | 保留观测 | 保留观测 | 11/332 | 跨权重不作同等要求 |
| matched_quantization: 02_fp16_inprocess2 → 04_awq_w4a16_mp16 | 否 | 禁用收益声明 | 保留观测 | 保留观测 | 11/332 | 跨权重不作同等要求 |
| same_weight_system_change: 05_fp16_mp16 → 02_fp16_inprocess2 | 是 | 51.25% | 941.02 | 976.65 | 341/2 | True |
| matched_quantization_methods: 04_awq_w4a16_mp16 → 03_gptq_w4a16_mp16 | 是 | -0.61% | -9.31 | -3.49 | 151/192 | 跨权重不作同等要求 |
| matched_quantization: 05_fp16_mp16 → 03_gptq_w4a16_mp16 | 是 | 15.89% | 291.77 | 280.42 | 341/2 | 跨权重不作同等要求 |
| matched_quantization: 05_fp16_mp16 → 04_awq_w4a16_mp16 | 是 | 16.40% | 301.07 | 282.98 | 341/2 | 跨权重不作同等要求 |

## 每个episode的HTTP统计

| 配置 | Episode | 请求 | Sum/s | Mean/s | P50/s | P95/s |
|---|---|---:|---:|---:|---:|---:|
| gptq_w4a16_inprocess2 | 1250 | 128 | 96.317 | 0.7525 | 0.7543 | 0.7789 |
| gptq_w4a16_inprocess2 | 159 | 14 | 9.766 | 0.6975 | 0.7485 | 0.7683 |
| gptq_w4a16_inprocess2 | 1413 | 16 | 13.177 | 0.8235 | 0.7472 | 1.6277 |
| gptq_w4a16_inprocess2 | 558 | 156 | 115.394 | 0.7397 | 0.7364 | 0.7650 |
| gptq_w4a16_inprocess2 | 1123 | 17 | 12.318 | 0.7246 | 0.7536 | 0.7753 |
| gptq_w4a16_inprocess2 | 384 | 12 | 8.542 | 0.7119 | 0.7538 | 0.7712 |
| awq_w4a16_inprocess2 | 1250 | 128 | 96.661 | 0.7552 | 0.7575 | 0.7918 |
| awq_w4a16_inprocess2 | 159 | 14 | 9.823 | 0.7016 | 0.7520 | 0.7897 |
| awq_w4a16_inprocess2 | 1413 | 16 | 11.193 | 0.6996 | 0.7495 | 0.7814 |
| awq_w4a16_inprocess2 | 558 | 156 | 118.330 | 0.7585 | 0.7373 | 0.7699 |
| awq_w4a16_inprocess2 | 1123 | 17 | 12.248 | 0.7204 | 0.7495 | 0.7558 |
| awq_w4a16_inprocess2 | 384 | 12 | 8.502 | 0.7085 | 0.7502 | 0.7689 |
| fp16_inprocess2 | 1250 | 128 | 114.896 | 0.8976 | 0.9005 | 0.9227 |
| fp16_inprocess2 | 159 | 14 | 11.937 | 0.8527 | 0.8966 | 0.9328 |
| fp16_inprocess2 | 1413 | 16 | 13.534 | 0.8459 | 0.8961 | 0.9182 |
| fp16_inprocess2 | 558 | 156 | 141.550 | 0.9074 | 0.8784 | 0.9107 |
| fp16_inprocess2 | 1123 | 17 | 14.835 | 0.8727 | 0.8972 | 0.9334 |
| fp16_inprocess2 | 384 | 12 | 10.272 | 0.8560 | 0.8984 | 0.9096 |
| gptq_w4a16_mp16 | 1250 | 128 | 200.337 | 1.5651 | 1.5809 | 1.6254 |
| gptq_w4a16_mp16 | 159 | 14 | 19.064 | 1.3617 | 1.4442 | 1.6080 |
| gptq_w4a16_mp16 | 1413 | 16 | 22.933 | 1.4333 | 1.6000 | 1.6135 |
| gptq_w4a16_mp16 | 558 | 156 | 244.324 | 1.5662 | 1.5888 | 1.6136 |
| gptq_w4a16_mp16 | 1123 | 17 | 25.634 | 1.5079 | 1.6032 | 1.6192 |
| gptq_w4a16_mp16 | 384 | 12 | 17.425 | 1.4521 | 1.5789 | 1.6015 |
| awq_w4a16_mp16 | 1250 | 128 | 197.100 | 1.5398 | 1.5849 | 1.6192 |
| awq_w4a16_mp16 | 159 | 14 | 18.720 | 1.3372 | 1.4483 | 1.6080 |
| awq_w4a16_mp16 | 1413 | 16 | 22.518 | 1.4073 | 1.5932 | 1.6178 |
| awq_w4a16_mp16 | 558 | 156 | 244.722 | 1.5687 | 1.5826 | 1.6162 |
| awq_w4a16_mp16 | 1123 | 17 | 25.793 | 1.5172 | 1.6136 | 1.6249 |
| awq_w4a16_mp16 | 384 | 12 | 17.672 | 1.4727 | 1.6026 | 1.6305 |
| fp16_mp16 | 1250 | 128 | 236.005 | 1.8438 | 1.8734 | 1.9075 |
| fp16_mp16 | 159 | 14 | 23.940 | 1.7100 | 1.8500 | 1.8972 |
| fp16_mp16 | 1413 | 16 | 27.438 | 1.7148 | 1.8786 | 1.8991 |
| fp16_mp16 | 558 | 156 | 291.880 | 1.8710 | 1.8620 | 1.8878 |
| fp16_mp16 | 1123 | 17 | 30.229 | 1.7782 | 1.8751 | 1.8919 |
| fp16_mp16 | 384 | 12 | 20.302 | 1.6918 | 1.7766 | 1.8942 |

逐请求配对差、各episode配对差、prefill/vision/decode阶段和client gap完整数据保存在JSON。正配对差表示候选更快。

## 限制与未通过项

- All measurements are one complete 343-request frozen historical sequence per configuration, without feedback; they do not measure navigation SR or quality.
- Only this same-manifest no-idle replay is paired. Fixed-input study latencies with one-second idle gaps are never pooled or substituted.
- Same-weight MP16/in-process2 changes require exact token equality for every request. Cross-precision token differences are not treated as quality failures or successes.
- Per-request HTTP intervals include serial service effects. CUDA-event stage intervals include dispatch gaps and post-response decode work; nested stages must not be added as exclusive wall time.
- Only the two excluded smoke requests have actual tensor SHA receipts. Every historical request is bound to frozen payload/JPEG/text hashes and observed prompt tokens, not a newly collected tensor SHA.
- Client gaps are reported separately from HTTP intervals and retained in sequence wall time. Payload decoding, audit checks, and stage-file reads occur outside the measured request loop.
- P50/P95 over different historical requests describe this sequence; one pass supplies no repeated-run uncertainty, significance, or independence claim.
- The original full-model quantization quality evidence remains separate. Forced 18-token replay output is not fed back or parsed into a navigation success metric.
- 04_awq_w4a16_mp16 → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 00_gptq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 01_awq_w4a16_inprocess2 → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 05_fp16_mp16 → 01_awq_w4a16_inprocess2: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 03_gptq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
- 02_fp16_inprocess2 → 04_awq_w4a16_mp16: engine configuration differs；observed execution identity differs；other candidate switches differ
