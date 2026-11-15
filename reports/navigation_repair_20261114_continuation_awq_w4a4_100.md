# 新修复配置同100条累计复核

配置 `awq_w4a4`：通过验收的新20条＋相同冻结条件的剩余80条；全部100条已逐步重算。

| 配置 | 成功/100 | SR | SPL | 真实请求 | HTTP P50/s | HTTP P95/s | 整卡峰值/MiB | 质量错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| awq_w4a4 | 53 | 0.530 | 0.4472 | 4844 | 1.705 | 2.027 | 6519 | 0 |

输入token 13422706；返回token 88660；实际prefill 4844次、decode 153792次。

20条来源：`logs/navigation_repair_20261112/screen_six_balanced_v1/awq_w4a4`；80条来源：`logs/navigation_repair_20261112/continuation_20261114_v1/remaining80_awq_w4a4/awq_w4a4`。执行顺序为20后80，完整案例顺序、配置、审计记录和全部已用证据SHA保存在[navigation_repair_20261114_continuation_awq_w4a4_100.json](navigation_repair_20261114_continuation_awq_w4a4_100.json)。

P50/P95直接合并全部真实HTTP请求重算，未平均两批分位。排队动作、4次导航预热、固定输入测量和尾请求均排除。质量失败保留在100分母。显存为200ms整卡采样峰值。

该结果仅代表这一组的同100条复测；不能据此宣称八组全部完成，也不能把20条验收当作统计等价证明。
