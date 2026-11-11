# 导航实验源码交付与复现

本指南补齐 SimpleVLN 导航闭环、实验编排、第一阶段 C/D 入口、AWQ 量化与转换，以及校准输入和模型整理的依赖。原有运行源码和冻结配置按原字节交付；新增工具只负责恢复依赖、生成独立复现清单和静态检查。

只读源码检查通过不等于 GPU 运行通过。权重、场景、渲染图、校准张量和历史原始日志不随 Git 分发。静态验收不导入实验模块，不执行服务、仿真、量化、显存探测或进程控制。本文后面的环境安装和实验命令需要另行执行，不能在现有测量期间运行。

## 在干净副本检查代码

在新克隆目录中，用 Python 3.10 或更新版本运行：

```bash
python -B checks/check_navigation_repository.py --require-tracked
python -B checks/check_navigation_reproduction_unit.py
python -B checks/restore_navigation_sources.py --restore
python -B checks/check_navigation_repository.py --require-tracked --require-external
```

检查器只解析和编译 Python 语法，核对交付清单、静态本地依赖、源码字节和冻结引用。它不会运行被检查脚本，亦不会调用那些可能在导入时创建目录、加载框架或停止服务的旧脚本。输出中的 `data_and_model_artifacts_not_validated` 明确列出尚未验证的数据和模型资产。

`restore_navigation_sources.py` 默认只检查，缺文件返回非零；`--restore` 下载缺失文件。下载后先核对 SHA256，再写入指定位置；已存在但内容不符时直接拒绝覆盖。固定来源见 `setup/reproduction/external_sources.json`：

| 依赖 | 固定版本与用途 |
|---|---|
| SimpleVLN 的 `eval/vln/eval.py` | commit `a450c57cff323587f54b0654c8e20e530a7202ac`；原提示词、历史采样和动作解析 |
| VLN-CE 的 `task.py`、配置、许可证 | commit `729d141b2ee10628061ada74dd3a5b9f70faeba5`；Habitat 数据集模块 |
| RL/SFT 模型元信息 | 仓库内固定的模型 revision 和文件清单；恢复到旧下载器要求的位置 |
| Habitat-Lab 源码压缩包 | v0.1.7 标签加固定 SHA256；仅指定 `--habitat-lab` 时获取 |

运行时代码仍使用原来的 `tmp/habitat_setup/...` 路径，恢复工具填补该路径，不改冻结代码。外部源码由相应上游项目提供，保留其许可证和归属；本项目的代码许可不会替代第三方许可。

## 环境与授权数据

实际实验运行在 Linux x86_64 或 WSL2 Ubuntu。三个环境必须隔离：

| 环境 | 关键版本 | 路径约定 |
|---|---|---|
| 推理 | Python 3.12，torch 2.5.1+cu121，LMDeploy 0.11.1，Transformers 4.57.6，Triton 3.1.0 | `~/.venvs/project2` |
| 量化与张量预处理 | torch 2.8.0，Transformers 4.57.3，llmcompressor 0.9.0，compressed-tensors 0.13.0 | `~/.venvs/project2-quant312` |
| Habitat | Python 3.6.15，Habitat-Sim/Lab 0.1.7，NumPy 1.19.5，CPU torch 1.10.2 | `~/.local/share/project2-habitat/mamba/envs/vlnce-017` |

推理环境按根 README 的环境安装步骤配置，版本见 `setup/inference-requirements.txt`。量化依赖使用 `setup/navigation-quant-requirements.txt` 和已记录的 `setup/navigation-quant-lock.txt`，不要通过旧 `setup_navigation_quant.sh` 重新覆盖锁文件：

```bash
uv venv --python 3.12.14 ~/.venvs/project2-quant312
uv pip install --python ~/.venvs/project2-quant312/bin/python -r setup/navigation-quant-lock.txt
```

在独立复现机器上，Habitat 的准备顺序是：

```bash
python -B checks/restore_navigation_sources.py --restore --habitat-lab
bash scripts/setup_habitat_manager.sh
bash scripts/install_habitat_environment.sh
bash scripts/install_habitat_task_environment.sh
```

这些安装脚本会安装软件或检查依赖导入，不属于上述静态验收命令。渲染采用 `LIBGL_ALWAYS_SOFTWARE=1` 和 llvmpipe。旧二进制的系统兼容问题需单独诊断，不能通过更换版本后继续声称完全相同的执行环境。

按 [VLN-CE 官方数据说明](https://github.com/jacobkrantz/VLN-CE#data)获取 `R2R_VLNCE_v1-3.zip`，按官方授权渠道获取 Matterport3D 场景；许可和来源沿用 `datasets/navigation_20261110/README.md`。将场景放到 `data/scene_datasets/mp3d/<scene>/<scene>.glb` 和同目录 `.navmesh`。

```bash
python -B checks/restore_navigation_frozen_inputs.py --archive /path/to/R2R_VLNCE_v1-3.zip
```

此命令校验官方压缩包，恢复原校准 8 条训练轨迹、历史 200 条标注子集及所需清单，并核对原 gzip 文件哈希。历史计划含 200 条，实际已报告开发池只使用前三批 150 条，本次同 100 条为前两批；恢复第四批不代表执行过第四批。它不会恢复或复制场景、图像、张量、权重或历史结果。若 gzip 字节与冻结值不符，应核对 Python/zlib 环境，不能改旧哈希绕过检查。

## 模型整理与校准输入

以下步骤在新的复现目录执行，目标目录不得复用正在运行的实验目录。源模型 RL revision 固定为 `4078dab77f3c93a0711b57549ac54a59c1c1e1ea`：

```bash
python checks/download_simplevln_rl.py
python checks/prepare_simplevln_serving.py --variant RL
```

下载器校验模型文件并写 `source_manifest.json`；整理脚本为权重建立硬链接，转换 Transformers 配置字段，并写 `serving_adapter.json`。模型源目录和整理目录应位于支持硬链接的同一文件系统。不要手动对硬链接权重原地修改。

已恢复校准 manifest 和场景后，生成原作者格式的 RGB 输入；这一步执行仿真，应与当前测量错开：

```bash
env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 bash scripts/run_habitat_task_python.sh checks/capture_navigation_calibration.py
~/.venvs/project2-quant312/bin/python checks/prepare_navigation_awq_learning.py --output data/calibration/navigation_awq_learning_20261104/processed
```

第一条产生 `data/calibration/navigation_awq_v1/samples.json` 和 JPEG；第二条产生 `00.pt` 至 `31.pt`、manifest 和 provenance。每轨迹采集四个时刻，共 32 个输入，参考路径只用于训练集观测采集。图像/提示词清单见 `datasets/navigation_20261110/calibration_inputs32.json`，张量清单见 `prepared_tensor_manifest.json`。应逐项核对 ID、顺序、图像 SHA256 和张量语义；渲染或序列化环境变化可能导致字节不同，不能更改冻结哈希把新输入冒充旧输入。固定输入计时还会严格校验 `samples.json` 的原始字节。

新工具已从公开冻结清单恢复必需元信息，因此无需为了本轮重建再运行旧 `prepare_navigation_quant_data.py` 或 `prepare_navigation_stage2.py`。后者会试图写入已存在的冻结执行配置。

## 量化和格式转换

以下命令加载模型并使用 GPU，属于之后单独安排的实验步骤。AWQ 和 GPTQ 共用同一组 32 个输入、FP16 舍入后的源权重及冻结配方；目标模型和日志目录必须为空。

```bash
~/.venvs/project2-quant312/bin/python checks/quantize_navigation_awq_learning.py --config setup/navigation-awq-learning-20261104.json
~/.venvs/project2-quant312/bin/python checks/convert_navigation_awq_learning.py --source models/SimpleVLN_Qwen3VL_4B_RL_AWQ_CT_learning_20261104 --output models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104
~/.venvs/project2-quant312/bin/python checks/quantize_navigation_gptq_stage2.py --config setup/navigation-gptq-stage2-20261109.json
~/.venvs/project2-quant312/bin/python checks/convert_navigation_gptq_stage2.py --source models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_CT_stage2_20261109 --output models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_stage2_20261109
```

保留量化元数据、转换审计和 kernel 接入证据。转换成功不代表导航精度验收成功。已有目录不应删除后重跑来覆盖历史证据。

## 服务入口与新一轮闭环

入口对应关系如下；一次只启动一个服务。

| 路线 | 入口 |
|---|---|
| 第二阶段 FP16/AWQ/GPTQ/FastV/SparseVLM | `checks/serve_navigation_stage2.py --group <group> --run-dir <新日志目录>` |
| 第一阶段 C，DART75 | `bash scripts/start_simplevln_rl_dart_learning.sh 0.75 navigation` |
| 第一阶段 D，AWQ+DART75 | `bash scripts/start_simplevln_rl_awq_dart.sh 0.75` |
| 九组编排 | `checks/run_navigation_stage2_validation.py`、`checks/run_navigation_stage2_screen.py` |
| 中断恢复 | `checks/resume_overnight_batch.py`、`checks/resume_navigation_dart150.py`、`checks/audit_navigation_dart_recovery.py` |

第一阶段 D 有专用组合运行时和模块映射。第二阶段量化组保留 keep=1.0 的边界，不通过修改阶段二入口来替代 D 组。恢复脚本必须配套原运行目录、进程身份记录和审计证据；干净克隆不能凭汇总报告“恢复”一个不存在的历史进程。

从干净副本运行同 100 条的新评测，可先生成独立清单。例如 FP16（这一步仅写元数据）：

```bash
python -B checks/prepare_navigation_reproduction.py --group fp16 --count 100 --output data/reproduction/fp16_same100
```

在之后允许测量的时段，终端一启动服务：

```bash
~/.venvs/project2/bin/python checks/serve_navigation_stage2.py --group fp16 --run-dir logs/reproduction_fp16_service
```

终端二依次执行两批，使用原作者策略 `--policy memory`：

```bash
env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 bash scripts/run_habitat_task_python.sh checks/run_simplevln_diagnostic.py --policy memory --manifest data/reproduction/fp16_same100/batch_01.json --output logs/reproduction_fp16/batch_01
env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 bash scripts/run_habitat_task_python.sh checks/run_simplevln_diagnostic.py --policy memory --manifest data/reproduction/fp16_same100/batch_02.json --output logs/reproduction_fp16/batch_02
```

对 AWQ/GPTQ 使用相应 `--group` 和各自新输出目录；C/D 清单生成器支持 `dart75`、`awq_dart75`，服务使用上表专用入口。此独立入口保留案例成员和顺序，生成新结果，不伪造旧控制器的验收记录。第一阶段与第二阶段性能配置不同，不能混作纯算法速度对照。

固定输入计时入口为 `run_navigation_quant_timing.py`、`run_navigation_quant_attribution.py`、`probe_navigation_quant_bottlenecks.py`；三者调用的 `prepare_input` 已包含在仓库。它们需要校准 JPEG、原始样本 JSON、全部对应模型和指定环境，也会启动服务或访问 GPU，不属于静态检查。

SR/SPL 保留全部固定案例作为分母；延迟只统计真实模型请求，排除预热和队列动作的零秒记录。历史报告的完整重新审计还需要原始逐步日志、运行时 trace 和显存采样；源码交付不替代这些证据。
