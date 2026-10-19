# 项目2：资源受限的多模态推理部署与业务联调

在 RTX 4070 Ti 12 GB 上，以 Qwen3-VL-2B-Instruct 运行视觉语言导航业务，使用 LMDeploy、FastAPI 和 Gradio 完成图片问答及本地视频联调。

**当前状态（2026-10-19）：本地业务链路已跑通，完成5图各10次的原版/新版提示词对照及初步整卡显存采样，修复无有效结果时默认前进等业务问题。整体导航质量、稳定性能基线和推理引擎优化尚未完成。** 视频文件FPS不能用作模型决策频率，提示词对照的耗时变化不能用作推理加速成果。

## 来源与已完成工作

业务代码基于老师提供的 `Qwen3VL-main.zip`。原项目位于 `Qwen3VL-main/Qwen3VL-main/`，保留其 [MIT许可及原版权声明](Qwen3VL-main/Qwen3VL-main/LICENSE)。当前尚未确认该代码包对应的上游公开仓库地址。

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

原业务目录中的README、部署文档仍保留原方案（例如32B、多卡及响应频率描述）。这些描述不是本项目已经实测的成果；当前配置与验证范围以本页及[阶段汇报](项目2_阶段汇报_20261019.md)为准。

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

## 已知限制

1. 部分模型文字理由与航点及任务约束不一致，正式质量验收未完成。
2. 原版50次重复测量复现2次非零航点和9次模型自报到达，均存在于原始JSON；新版本地50次未再出现，但不代表模型在其他场景可靠。尚缺可移动正例及独立的实际到达验证。
3. 异步请求编号重复已修复，排队、采样跳过、丢帧及超时检查通过；服务已重启，首轮6次真实请求（含预热）各匹配一条诊断记录。编号可以有空缺，不能将最大编号当作处理数量。
4. TUM桌面样例用于部署联调，没有本项目正式语言导航成功标签。
5. 页面速度是航点换算值；没有实际机器人闭环导航验证。
6. 非流式聊天业务接口的 `finish_reason` 仍硬编码为stop，应查看诊断中的真实后端终止原因。
7. WSL字体修复读取本机 `/mnt/c/Windows/Fonts/`；其他系统需要可用的中文字体。
8. 启动时出现过Triton编译信息，现有输入可运行，不代表所有输入形状均已验证。

后续以固定输入建立可信基线，再依据测量提出优化方案。量化、视觉Token剪枝、投机解码等尚未实施，不作为已完成成果。
