# 新修复配置同100条累计复核

配置 `gptq_w4a16`：通过验收的新20条＋相同冻结条件的剩余80条；全部100条已逐步重算。

| 配置 | 成功/100 | SR | SPL | 真实请求 | HTTP P50/s | HTTP P95/s | 整卡峰值/MiB | 质量错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gptq_w4a16 | 68 | 0.680 | 0.5798 | 3733 | 1.479 | 1.790 | 6498 | 0 |

输入token 10266546；返回token 67110；实际prefill 3733次、decode 118112次。

20条来源：`logs/navigation_repair_20261112/initial_sequence_v1/w4_screens/gptq_w4a16`；80条来源：`logs/navigation_repair_20261112/continuation_20261114_v1/remaining80_gptq_w4a16/gptq_w4a16`。执行顺序为20后80，完整案例顺序、配置、审计记录和全部已用证据SHA保存在[navigation_repair_20261114_continuation_gptq_w4a16_100.json](navigation_repair_20261114_continuation_gptq_w4a16_100.json)。

P50/P95直接合并全部真实HTTP请求重算，未平均两批分位。排队动作、4次导航预热、固定输入测量和尾请求均排除。质量失败保留在100分母。显存为200ms整卡采样峰值。

该结果仅代表这一组的同100条复测；不能据此宣称八组全部完成，也不能把20条验收当作统计等价证明。
