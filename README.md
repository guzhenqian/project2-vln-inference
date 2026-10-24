# 项目2：资源受限的多模态推理部署与业务联调

在 RTX 4070 Ti 12 GB 上，以 Qwen3-VL-2B-Instruct 运行视觉语言导航业务，使用 LMDeploy、FastAPI 和 Gradio 完成图片问答及本地视频联调。

**当前状态（2026-10-24）：已完成本轮工程baseline：单卡部署与业务联调、两轮各50次提示词对照、12图18次诊断及逐项初审、已有性能与显存汇总。导航质量仍有已复现失败，稳定尾延迟、视频新结果频率及推理引擎优化未完成。** 结果与边界见[baseline报告](reports/baseline_report_20261024.md)。视频文件FPS不能用作模型决策频率，提示词对照的耗时变化不能用作推理加速成果。

## 业务功能与已完成工作

业务层提供图片问答与视觉语言导航接口：接收图像和自然语言指令，管理导航任务，调用模型并解析场景描述、动作和航点；Gradio页面支持本地视频送帧与结果展示。业务代码位于 `Qwen3VL-main/Qwen3VL-main/`，许可证见 [LICENSE](Qwen3VL-main/Qwen3VL-main/LICENSE)。

本项目已完成：

- WSL2 Ubuntu与独立Python环境配置、GPU计算验证。
- 固定版本2B模型下载、权重完整性校验及单卡运行。
- 模型直连与FastAPI业务层图片问答验证。
- 导航任务创建、同步单帧处理、离线视频送帧和页面展示。
- 修复页面在任务结束后继续送帧导致重复400报错的问题。
- 补充原始模型输出、终止原因、解析结果和速度的诊断日志。
- 修复WSL下中文叠加字体路径。
- 修复异步帧编号重复，并以真实请求复核结果关联。
- 固定5张输入完成初步串行计时，补充约200毫秒间隔的整卡显存采样。
- 增加可切换的`baseline_v1`/`grounded_stop_v2`系统提示词，并记录实际版本与哈希；同一组图片分别完成50次真实请求。
- 修复无效JSON/纯文本解析时默认前进、否定到达与字符串布尔误判，以及异步首帧尚无结果时默认前进的问题；26种异常用例和100条历史回答离线回放通过。
- 增加使用单调时钟的准备、模型调用、解析分段计时；已采集5条正式分段样本，核对各段与业务总耗时一致。
- 固定12张图像及探索任务，完成首轮12次和6次复测，保留原始回答、解析结果、来源及哈希；完成逐图初审并保留待定项。

## 当前baseline结果（2026-10-24）

| 维度 | 结果与范围 |
| --- | --- |
| 质量诊断 | 12张图来自2个TUM序列，首轮+复测共18次；18/18处理成功且JSON有效，不代表导航质量通过 |
| 明确失败 | 两张桌面图在探索任务下各3/3给出非零前进；缺少地面证据的问题已存在于模型原始回答中 |
| 固定输入性能 | 10/19 v2提示词50次正式请求，客户端中位1.348秒；与原版的耗时差不能宣称推理加速 |
| 分段线索 | 10/20的5条正式样本中，模型调用中位1.246063秒，占累计业务耗时约99.977%；包括后端HTTP及调度，不是纯GPU耗时 |
| 整卡显存 | 独立采样窗口最大7710 MiB，约7.53 GiB；不是模型专属显存或精确瞬时峰值 |
| 视频功能 | 历史已验证送帧、显示、中文叠加及结束处理；本轮确认加载首帧，未新增完整连续运行或新模型结果频率测量 |

公开的[逐条结果摘要](reports/baseline_results_20261024.json)包含18条原始回答、解析与耗时、图像哈希和分项审核说明，未包含原始服务日志、机器绝对路径或图片。图片审阅包单独提供；该诊断集不是官方导航标注或独立场景泛化测试。

## 初步实测（2026-10-19）

1次预热后，5张固定640×480图片各提交1次独立同步导航请求：客户端等待中位数 **1.611秒**（1.494–1.848秒），业务处理中位数 **1.604秒**。另一次相同请求流程中，整卡显存采样最大值 **7710 MiB**；包含预热、辅助开销、桌面和其他GPU进程，可能遗漏短时峰值。

这不是纯GPU耗时、稳定P95、视频决策频率或导航质量通过结论。范围、方法、复现命令和限制见[初步性能记录](项目2_初步性能记录.md)。

进一步对5张图各重复10次，两轮均另加1次预热；每轮51条请求各匹配一条真实业务诊断，无丢帧或流水线错误：

| 项目 | 原版提示词，50次 | grounded_stop_v2，50次 |
| --- | ---: | ---: |
| 客户端完整请求等待中位数 | 1.470秒 | 1.348秒 |
| 客户端等待范围 | 1.300～2.717秒 | 1.015～1.927秒 |
| 原始JSON含非零航点 | 2次 | 0次 |
| 原始JSON自报已到达 | 9次 | 0次 |
| 正常结束且JSON有效 | 50次 | 50次 |

本轮仅反映桌面样例中停止约束的改善；新版至少5条理由仍声称可在桌面移动，整体质量未通过。输入与生成长度均变化，耗时下降不构成推理引擎加速结论。完整证据见[提示词对照报告](reports/navigation_prompt_v2_comparison_20261019.md)。

原业务目录中的README、部署文档仍保留原方案（例如32B、多卡及响应频率描述）。这些描述不是本项目已经实测的成果；当前配置与验证范围以本页及[baseline报告](reports/baseline_report_20261024.md)为准。[10/19阶段汇报](项目2_阶段汇报_20261019.md)保留当时状态。

## 架构

```text
浏览器
  └─ Gradio 页面 :7860（独立页面环境）
       └─ FastAPI 业务服务 :20000（任务、帧采样、提示词与结果解析）
            └─ LMDeploy :23333（GPU模型推理）
```

FastAPI启动日志中的“模型加载完成”对应HTTP客户端初始化；模型权重由LMDeploy进程加载。异步帧接口可能返回历史结果或首帧默认值，接口响应次数不等于模型产生新结果的次数。

## 已验证环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti，12282 MiB |
| 系统 | Windows + WSL2 Ubuntu |
| Python | 3.12.14 |
| 推理环境 | PyTorch 2.5.1+cu121、Torchvision 0.20.1+cu121、Triton 3.1.0 |
| 模型框架 | LMDeploy 0.11.1、Transformers 4.57.6 |
| 页面环境 | Gradio 5.49.1，与推理环境分离 |
| 运行配置 | 单卡、BF16、session_len=2048、max_batch_size=1 |

## 从源码运行

以下命令在WSL Ubuntu中、仓库根目录下执行。前提是GPU驱动和WSL GPU访问可用，并已安装 `uv`。

### 1. 创建推理与页面环境

```bash
uv venv --python 3.12.14 ~/.venvs/project2
uv pip install --python ~/.venvs/project2/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python ~/.venvs/project2/bin/python -r setup/inference-requirements.txt -e ./Qwen3VL-main/Qwen3VL-main

uv venv --python ~/.venvs/project2/bin/python ~/.venvs/project2-webui
uv pip install --python ~/.venvs/project2-webui/bin/python -r setup/webui-requirements.txt
```

### 2. 下载固定版本模型并检查

```bash
source ~/.venvs/project2/bin/activate
hf download Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203 --local-dir models/Qwen3-VL-2B-Instruct
python checks/check_gpu.py
python checks/check_inference_imports.py
python checks/check_model_files.py
python checks/check_business_imports.py
```

模型文件与校验信息见 `setup/model-manifest.json`。权重不随仓库上传。

### 3. 启动三个服务

在三个终端分别进入仓库根目录并运行一条命令，启动后保留终端：

```bash
bash scripts/start_lmdeploy_2b.sh
```

```bash
PROJECT2_NAV_PROMPT_PROFILE=grounded_stop_v2 bash scripts/start_business_api.sh
```

```bash
bash scripts/start_webui.sh
```

在Windows浏览器打开 http://localhost:7860 ，页面应显示服务正常、模型为Qwen3-VL-2B-Instruct。

上面显式选择本次已测的新版提示词；不指定`PROJECT2_NAV_PROMPT_PROFILE`时，启动脚本默认使用保留的`baseline_v1`。不要把两个版本的测量混在一起。

### 4. 准备公开样例并检查业务

```bash
~/.venvs/project2/bin/python checks/prepare_tum_sample.py
~/.venvs/project2-webui/bin/python checks/prepare_tum_video.py
~/.venvs/project2/bin/python checks/check_image_api.py --base-url http://127.0.0.1:20000
~/.venvs/project2/bin/python checks/check_navigation_api.py
```

样例来自[TUM RGB-D官方数据](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)。视频准备脚本读取数百MB压缩流，选取最早150张RGB帧，按时间排序后编码MJPG AVI，记录来源、时间戳及哈希；不保存完整数据集压缩包。

页面“视频路径或URL”应填写WSL可读的绝对路径，例如仓库所在目录下的 `/mnt/e/.../data/tum_rgbd/desk_clip_150.avi`。点击“加载视频”，输入指令、发布任务，再点击“开始”；测试完点击“停止”。

当前样例指令：

```text
观察当前桌面场景，寻找键盘。如果画面不能确认机器人可通行的地面路径，请保持停止并说明原因。
```

## 检查与证据

- `checks/`：GPU、模型文件、接口、页面、中文字体与相关修复的检查脚本。
- `scripts/`：本项目三个服务的单卡启动脚本。
- `setup/`：依赖文件、模型清单及修改前的部分原文件备份。
- `项目2_阶段汇报_20261019.md`：阶段结果与尚未解决的问题。
- `项目2_初步性能记录.md`：初步请求耗时、显存采样方法及结果。
- `reports/initial_measurements_20261019.json`：可分享的实测摘要、5条原始模型回答及本地原始记录哈希，已去除机器绝对路径。
- [原版50次记录](reports/repeated_measurements_20261019.json)、[新版50次记录](reports/grounded_stop_v2_measurements_20261019.json)：保留逐次原始回答、计时与诊断关联，去除机器绝对路径。
- [无有效结果默认停止修复](reports/fallback_stop_fix_20261019.md)：修改范围、26种异常用例与100条历史回答回放证据。
- [10/24 baseline报告](reports/baseline_report_20261024.md)、[公开结果摘要](reports/baseline_results_20261024.json)：质量诊断、已有性能、分段线索与视频证据范围。
- `logs/`：在本地生成的真实请求、响应和诊断记录；不上传仓库。

业务启动脚本当前设置 `PROJECT2_DIAGNOSTICS=1`，诊断条目以 `VLN_DIAGNOSTIC` 标识。诊断开销需在后续性能测量中明确记录。部分检查使用模拟响应，并明确打印mocked；这类检查只验证代码行为，不能视为真实模型实验。

以下离线回归不连接模型服务：

```bash
~/.venvs/project2/bin/python checks/check_prompt_profiles.py
~/.venvs/project2/bin/python checks/check_fallback_stop.py
~/.venvs/project2/bin/python checks/check_navigation_diagnostics.py
```

真实重复测量应在页面停止播放、没有其他请求且服务已使用对应提示词版本时运行，例如：

```bash
~/.venvs/project2/bin/python checks/prepare_baseline_inputs.py
~/.venvs/project2/bin/python checks/measure_initial_latency.py --repeats 10 --prompt-profile grounded_stop_v2
```

脚本用预热诊断核对实际版本，新版记录使用独立文件名；不会将格式有效或零速度自动视为导航质量通过。当天两轮真实对照发生在默认停止修复之前；之后回放100条有效JSON结果不变，不能把离线回放视为新的模型测量。

12图诊断脚本及离线性能汇总的用法、所需本地数据见[复现说明](reports/baseline_report_20261024.md#复现与验证)。这些脚本不会自动获取整套评测数据；图片和完整运行日志保存在本地，公开JSON可直接审阅已有结果。

## 已知限制

1. 部分模型文字理由与航点及任务约束不一致，正式质量验收未完成。
2. 原版50次重复测量复现2次非零航点和9次模型自报到达，均存在于原始JSON；新版同任务50次未再出现，但10/24探索任务下桌面错误前进仍复现。已补充可见地面的移动候选，尚无独立场景泛化或真实到达验证。
3. 异步请求编号重复已修复，排队、采样跳过、丢帧及超时检查通过；服务已重启，首轮6次真实请求（含预热）各匹配一条诊断记录。编号可以有空缺，不能将最大编号当作处理数量。
4. TUM样例用于部署联调和轻量诊断，没有本项目正式语言导航成功标签；12图只有2个来源序列，部分样例已有历史模型输出。
5. 页面速度是航点换算值；没有实际机器人闭环导航验证。
6. 非流式聊天业务接口的 `finish_reason` 仍硬编码为stop，应查看诊断中的真实后端终止原因。
7. WSL字体修复读取本机 `/mnt/c/Windows/Fonts/`；其他系统需要可用的中文字体。
8. 启动时出现过Triton编译信息，现有输入可运行，不代表所有输入形状均已验证。

本轮baseline已形成可复查记录；后续基于固定版本进一步分析模型调用耗时，或开展一次有依据的质量修复对照。量化、视觉Token剪枝、投机解码等尚未实施，不作为已完成成果。
