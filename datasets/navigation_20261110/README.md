# 项目数据与固定案例清单

当前导航开发池是 **R2R_VLNCE_v1-3 val_unseen 中固定的150条任务**，本次精度比较使用原来的前两批、同100条。它们不是新建的独立测试集，也不是官方1839条完整评测。每条任务含一条指令与一条轨迹；多组重复执行不会增加独立任务数。

| 数据 | 数量 | 用途 | 公开内容 |
|---|---:|---|---|
| 导航开发池 | 150条，11个场景 | 第一阶段150条；第二阶段从中初筛/复筛 | 完整任务标注、ID、批次、来源和哈希 |
| 本次固定比较集 | 100条 | AWQ/GPTQ各精度比较 | `selection.json`的`same100`，精确等于前两批 |
| 校准训练轨迹 | 8条，4个场景 | AWQ/GPTQ权重量化 | 完整任务标注、采样协议；与开发池场景不重叠 |
| 校准输入 | 32个 | 每训练轨迹4个观测时刻 | 原提示词、图像相对路径/哈希、预处理张量清单 |
| 历史工程验证 | 见清单 | TUM桌面/移动机器人图像、早期R2R检查 | 来源、选帧清单、历史任务ID；不混入100条统计 |

`val_unseen150.json`和`calibration_train8.json`保留原始指令、起点、目标、参考路径和距离等标注。目标和参考路径仅供环境评估/训练集校准图像采集，不能作为评测时的模型输入。模型实际只接收历史RGB、当前RGB和任务指令。

`batches/`保留历史冻结配置，因此个别`scope`仍写着原200条计划；当前实际使用范围以`selection.json`和已完成报告为准，不能据旧计划宣称200条已完成。`historical_selection_inventory.json`列出曾准备过的集合，存在清单不代表全部案例都执行过。

## 来源、许可与复现

R2R由Peter Anderson等发布，VLN-CE连续环境版本由Jacob Krantz、Erik Wijmans、Arjun Majumdar、Dhruv Batra、Stefan Lee等提供。参见[VLN-CE数据说明](https://jacobkrantz.github.io/vlnce/data)、[官方仓库与下载方法](https://github.com/jacobkrantz/VLN-CE)、[R2R项目许可说明](https://github.com/peteanderson80/Matterport3DSimulator#license)。本目录的第三方数据及派生内容适用[**Matterport3D学术数据使用协议**](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf)，不因仓库代码许可而变成MIT数据。

协议限定非商业学术使用，并要求分发数据或派生信息时附协议链接；分发大量数据还需接收方明确接受并留存记录。因此此仓库提供小规模任务标注与复现清单，**不镜像Matterport场景压缩包、网格/纹理、渲染图或校准张量**。场景通过官方授权渠道获取；该边界不表示所有小规模派生资料都禁止发布。

1. 按[官方说明](https://github.com/jacobkrantz/VLN-CE#data)获取`R2R_VLNCE_v1-3.zip`以及授权的MP3D Habitat场景。
2. 在仓库根目录运行下面命令。脚本核对官方压缩包SHA256、逐条核对公开标注，并保留官方词表生成可供Habitat加载的子集。

```bash
python checks/restore_navigation_dataset_release.py \
  --archive data/datasets/R2R_VLNCE_v1-3.zip \
  --output data/reproduced_navigation_20261110
```

3. 依照`selection.json`使用同100条，并保持原图像、历史、随机种子、动作上限和统计口径。场景文件放在`data/scene_datasets/mp3d/<scene>/<scene>.glb`；校准采集步骤及图像哈希见`calibration_protocol.json`、`calibration_inputs32.json`。

历史TUM RGB-D数据由Jürgen Sturm、Nikolas Engelhard、Felix Endres、Wolfram Burgard、Daniel Cremers提供，参见[官方数据与CC BY 4.0许可](https://cvg.cit.tum.de/data/datasets/rgbd-dataset#license)。本项目曾从视频解码/转码后选帧，变换方式和原文件哈希保存在`historical_tum/`，不冒充原始传感器PNG或官方导航动作标注。本目录没有镜像完整TUM视频。

文件完整性与场景隔离检查见`release_audit.json`；历史模型结果见仓库`reports/`。
