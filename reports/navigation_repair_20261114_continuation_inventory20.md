# 量化修复同100条：FP16＋八组状态

100条完成状态与20条验收资格分开记录；空白结果表示尚无完整100条证据。

| 配置 | 100/执行状态 | 最新20资格 | 比较状态 | 成功/100 | SR | SPL | ΔSR | ΔSPL | HTTP P50/s | HTTP P95/s | 质量错误 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| awq_w4a16 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| awq_w4a4 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| awq_w4a8 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| awq_w8a16 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| gptq_w4a16 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| gptq_w4a4 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| gptq_w4a8 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |
| gptq_w8a16 | 仅20条通过，待80条 | passed | pending_comparison | — | — | — | — | — | — | — | — |

100条没有额外“质量达标”阈值；只展示实际SR/SPL及同100 FP16差值。缺少FP16100时比较状态为 `pending_comparison`。初始输入不一致会阻止比较。

HTTP来自各自闭环轨迹的真实请求分布，不能纯归因于量化算法。后续轨迹不同是允许的；逐案例输入核对只针对第一次请求。

完整状态、逐字段来源、初始请求核对、失败信息及证据SHA见[navigation_repair_20261114_continuation_inventory20.json](navigation_repair_20261114_continuation_inventory20.json)。模型权重未在汇总步骤重新读取，引用此前累计器的校验结果。
