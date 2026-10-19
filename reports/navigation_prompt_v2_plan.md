# 项目2：停止与到达条件的最小修正及对照方案

日期：2026-10-19。当前状态：候选代码和离线验证完成，运行中的业务服务尚未切换，v2真实质量尚未测量。

## 已定位的链路

- `app/vln/prompts.py`的原系统提示词要求生成航点，但没有充分定义“无法确认地面路径必须停止”，也没有区分“目标可见”和“机器人已到达”。用户指令中的停止要求确实存在于实际诊断中；不是指令丢失。
- `app/vln/waypoint_parser.py`提取模型JSON中的航点和到达标志；已测50条的对应字段原始值和解析值一致。
- `app/vln/inference_pipeline.py`根据`result.reached_goal`直接把任务标记为completed；这是模型判断被业务采纳，不是独立的到达验证。
- 两次原始`dx=0.5`经首航点/0.2秒及限幅得到`linear_vel=1.5`。当前测试没有机器人执行器或实测运动。

## 本轮实施：只切换系统提示词

新增两个版本，默认仍为`baseline_v1`。原版系统提示词SHA-256保持为`5568b46e82d4fef204afa2c0160333f97a381f4749deabd0ba520f56e930c70c`。

`grounded_stop_v2`明确要求：

1. 先判断图像是否支持从机器人当前位置出发的可通行地面路径；目标可见、桌面平坦或没看见障碍物都不等于存在路径。
2. 无法确认路径时，输出`action="停止"`、`waypoints=[]`、`reached_goal=false`，理由说明缺少什么证据。
3. 实际到达与看见目标分开；没有到达证据时，不应宣布物理导航完成。
4. 动作、理由、航点、进度和到达标志应一致；保留原JSON字段。

v2系统提示词SHA-256：`cc15062cc1e796901332ea2aee43d038918992d23175b5973aa5f2ca75b22d71`。全文保存在`app/vln/prompts.py`，由`PROJECT2_NAV_PROMPT_PROFILE`选择。

本轮不修改解析规则、到达状态转换或生成参数，也不在后处理里强制把所有回答改成停止。因而原始输出是否改善可直接观察；提示词规则本身不构成可靠的执行层保障。

## 启用与测量

先停止页面播放，在业务API终端停止旧业务进程，再从WSL项目根目录启动：

```bash
PROJECT2_NAV_PROMPT_PROFILE=grounded_stop_v2 bash scripts/start_business_api.sh
```

启动日志应显示`Navigation prompt profile: grounded_stop_v2`和`Application startup complete`。模型服务和页面服务无需重启。

确认启动后，下一步才在操作终端测同一组图片：

```bash
~/.venvs/project2/bin/python checks/measure_initial_latency.py --repeats 10 --prompt-profile grounded_stop_v2
```

1次预热+50次正式请求，保持图片、用户指令、max_tokens=512、temperature=0.3不变。诊断记录实际系统提示词哈希；测量先用预热诊断核对版本，切错版本时停止，避免继续测50次。结果使用独立的`repeated_latency_grounded_stop_v2_*.json`文件名。

本地验收关注：实际输出是否停止、非零航点/到达标志是否仍出现、理由与动作是否一致、是否有截断或格式错误。可以比较该小样本中的事件计数，但不得称正式导航成功率。修改提示词会改变输入长度及生成内容，耗时差异不作为推理加速比例。

回到原版时使用`PROJECT2_NAV_PROMPT_PROFILE=baseline_v1 bash scripts/start_business_api.sh`重启业务服务。历史测量文件保留。

## 验证证据与后续边界

- WSL真实Python环境中，`checks/check_prompt_profiles.py`用保存的两条模型回答进行离线检查：原版文本不变、实际发送的系统提示词与版本相符、哈希一致、用户内容与生成参数相同、原始回答不被强制覆盖、拼错版本时不调用引擎。没有连接模型。
- 隔离HTTP用例通过：原版默认6次与重复51次、新版51次、失败时保留数据、错误版本或缺少新版哈希时预热后停止。
- 本地固定输入检查、Python语法和启动脚本Bash语法检查通过。
- 源码检查还发现独立问题：纯文本解析可能以关键词判完成，无匹配动作时默认前进；`bool("false")`会成为True；异步接口首帧默认前进。这些路径不是本次50条有效JSON中已复现问题的原因。本轮保持它们不变，后续分别修复并验证，不能把提示词改善当成这些问题已解决。
- 即使桌面样例全部停止，也只能证明对当前样例的约束遵守改善；还需要合理的可移动正例，防止将“总是停止”当成完整导航能力。正式导航数据及验收指标仍待老师确认。
