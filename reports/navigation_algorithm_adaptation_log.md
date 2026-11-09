# 导航推理项目：算法适配问题记录

最后更新：2026-11-10。按用户要求持续记录各算法的问题、原因、处理办法、验证证据和未解决项，随项目进展同步 GitHub。

状态说明：“已验证”必须注明验证范围；“代码检查”不代表实际模型跑通；“待验证”不得作为实验结论。发现失败或回退也保留记录。

## 共同配置与边界

第二阶段实际运行以 [执行配置](../setup/navigation-stage2-execution-20261109.json) 为准；[早期统一配置](../setup/navigation-stage2-common-20261109.json)保留为历史方案，其中日志和样本安排由执行配置覆盖。使用Qwen3-VL 4B Dense、LMDeploy 0.11.1 PyTorch、FP16、eager、最多3072输入token一次完整prefill。AWQ/GPTQ共用W4A16、非对称group128、252个语言投影层和同一批32个校准模型输入；视觉塔、merger、DeepStack、嵌入及输出头保持原精度。

七组真实接入验收于11月9日19:51全部通过；11月10日07:15完成七组各50条新运行，并复用两组DART相同子集，九组初筛齐全、均无请求/解析错误。450条结果及新运行13,857对执行记录的独立复核通过。原冻结代码和第一阶段结果保持原样；基线/AWQ已按eager/3072重跑，DART75/50沿用原案例和统计方式复用，未切换轻量日志。已记录[累计100条复筛决定](navigation_stage2_rescreen_plan_20261110.json)，保留全部稳定且有取舍或不确定性的配置，追加原batch02。

## 问题台账

| 编号 / 算法 | 适配问题及影响 | 处理办法 / 下一步 | 证据与当前状态 |
|---|---|---|---|
| COMMON-01 | 新服务首次启动时，LMDeploy EngineRole 枚举不能直接写JSON，导致加载模型前退出 | 按枚举value显式序列化，并将此检查移到配置检查分支之前；保留失败目录和控制台日志 | 已修复；七组后续真实服务均成功启动并完成验收 |
| COMMON-02 | FP16验收后切换AWQ，旧HTTP连接导致保守端口绑定检查短暂失败 | 控制器在新服务启动前等待端口可重新绑定，最多120秒；不终止无关进程，不修改模型运行时 | 已修复并完成后续全部组切换；首次AWQ失败发生在加载模型前，证据留存 |
| REPORT-01 | 离线Token诊断导出把字符串文件名传给只接受Path的哈希函数，汇总结束时退出 | 修正为Path后仅重做离线导出；未改运行时，也未重跑任何案例 | 失败在写报告前发生；初筛结果及独立复核不受影响 |
| AWQ-01 | compressed-tensors 的 INT4 存储布局与 LMDeploy PyTorch 原生 AWQ GEMM 不同 | 已转换整数布局、零点和 scale；区分位布局保值与 FP16 scale 舍入 | 旧转换脚本 `checks/convert_navigation_awq_learning.py`；[原生算子验证](navigation_awq_kernel_learning_20261104.json)。第一阶段真实模型已验证 |
| AWQ-02 | GQA 下不能直接套用 v_proj→o_proj 的 AWQ 平滑映射；校准算术精度也影响可比性 | 既有实现不使用该平滑映射；固定源权重先舍入 FP16、再用 FP32 校准，两量化算法统一 | 第二阶段同50条已完成：AWQ35/50、0请求/解析错误；100条复筛待完成 |
| GPTQ-01 | 当前 LMDeploy 0.11.1 PyTorch 的 Linear 构造器只识别 awq / smooth_quant / fp8，直接使用 gptq 会报 Unsupported quant method | 保持当前后端，采用 GPTQ 量化 → compressed-tensors → AWQ GEMM 存储布局；实验算法仍标注 GPTQ，不把格式名称当算法名称 | 真实转换、18项CUDA算子、两/九图及单1250通过；同50条完成36/50、0请求/解析错误，100条待完成 |
| GPTQ-02 | dynamic/group 激活排序可能引入非连续 weight_g_idx；旧 AWQ 转换脚本会丢弃该张量，不能直接用于通用 GPTQ | 新转换器只接受 static/weight；g_idx 必须缺省或逐列等于 floor(column/128)，否则拒绝。旧冻结脚本未改 | [CPU 检查](navigation_gptq_format_cpu_20261109.json)：2个小矩阵和5类拒绝检查通过；`checks/convert_navigation_gptq_stage2.py`已对真实252层完成核验 |
| GPTQ-03 | GPTQ Hessian 额外占内存；推理可放入 12GB 不代表校准也能放入。Cholesky 失败时库可能警告后回退 RTN | 采用逐层调度、CPU offload及Hessian卸载；捕捉回退并拒绝以回退产物冒充成功GPTQ | 本机4070Ti 12GB完成32输入/252层真实校准，无OOM、RTN回退0次；正式请求峰值仍以对应50/100批次统计 |
| DART-01 | Qwen3-VL 多图、DeepStack、mRoPE 与分层 KV 需要同步裁剪，不能直接复制 Qwen2.5-VL 实现 | 已按实际多图索引适配；低层保持原长度，高层使用压缩长度，并审计 prefill / 首次 decode | 第一阶段两图、九图、完整案例和 150 条运行已验证；[75%/50%复核](navigation_dart75_vs50_delivery_audit_20261109.json)、[组合复核](navigation_abcd_delivery_audit_20261109.json) |
| DART-02 | 当前选择器使用 GPU→CPU 搬运和 CPU FP32 评分；详细审计也有开销。固定 KV 池不因裁剪自动减少分配 | 记录执行位置和实际开销；若改 GPU 选择器或轻量日志则作为新实现重新验证，不沿用旧耗时 | `checks/navigation_dart_runtime.py` 中 `.cpu().float()`；统一配置。轻量性能模式尚未验证 |
| DART-03 | 历史DART没有单独记录与新选择器一致的分项耗时，不能从HTTP延迟中反推 | 复用时保留原HTTP指标，单独计时标为未记录；不补造值，也不为补分项而破坏本轮复用约定 | [Token与计时范围](navigation_stage2_diagnostics_50_20261109.md)；新方法分项也只是prepare+selection区间，非完整压缩开销 |
| FastV-01 | PriorTR 的 Qwen3-VL 参考依赖 Transformers 开发版本，图像区间按单连续区间推断；不适用于导航中夹杂分隔符的 2–9 张图输入 | 移植评分逻辑，使用真实多图 mask / span，接入现有 DeepStack、位置和分层缓存逻辑 | 两比例均通过CPU数值/控制流、真实两/九图和完整1250审计；见[七组验收](navigation_stage2_integration_20261109.md) |
| FastV-02 | mask-only 版本不会实际缩短后续层序列；参考实现的低层 KV 裁剪方式也与当前 DART 不同 | 实现物理删除，第2层前裁剪；低层保留原KV，高层序列缩短，评分开销计入请求耗时 | 75%与50%的真实prefill/first-decode审计均通过；不是mask-only加速宣称 |
| FastV-03 | 保留更少token不保证当前CPU评分/详细审计实现的端到端延迟下降 | 用真实HTTP结果判断收益，不用理论Token比例替代计时 | 同50条：75% P50=2.037秒，高于基线1.945秒；50% P50=1.697秒，但成功30/50低于75%的32/50，需复筛确认取舍 |
| SparseVLM-01 | PriorTR 标注 v1.0，不能等同 SparseVLM+；官方多层版本也不能与单层参考版混称 | 本轮固定PriorTR v1.0单层无合并配置，第3层前裁剪；明确作为这一适配变体比较 | 两比例真实两/九图及完整1250审计通过；原论文多层+回收及SparseVLM+未实现，不作为本轮结论 |
| SparseVLM-02 | keep_ratio 与 keep_tokens 的合并预算处理不同；合并还涉及新坐标/DeepStack/KV，多图区间问题同FastV | 本轮明确不启用合并，merged_tokens=0；直接按最终225N/150N预算选取全局视觉token，保留全部非视觉token，记录逐图分配 | 本轮无合并路径已验证；若以后启用合并，须另做坐标、特征和缓存验收，不能沿用本次通过结论 |
| SparseVLM-03 | 50%版本本批碰撞事件偏多，零解析错误不等同导航稳定性优秀 | 在100条复筛继续检查碰撞、终止方式与质量，保留全部失败案例 | 同50条：50%为1862次/7531动作（24.72/100动作），75%为1142次/6856动作（16.66/100动作）；不能只凭低P95判为部署优选 |

PriorTR参考已固定提交 `e442a4c14af1b333067a6b8310c9111cad922033`，7个参考文件的SHA保存在本地来源清单。首轮 FastV：零基第2层前裁剪，末token查询、跨头均值、全体多图视觉token全局Top-K。SparseVLM：明确标为 **PriorTR v1.0单层无合并适配版**，采用参考默认第3层前裁剪、初始视觉嵌入与末图后文本的相似度筛选重要文本、跨重要文本和头求均值，不宣称原论文多层回收算法或SparseVLM+。两者均保留原语义坐标、低层原KV长度、高层压缩KV长度；最终预算为225N/150N，逐图实际分配进入审计。

## 已开始的适配验证（2026-11-09 晚）

- GPTQ真实32输入、252投影量化已完成，耗时916.36秒、RTN回退0次；复用FP16舍入后的源权重和FP32校准算术，static分组、CPU卸载Hessian。转换后252层整数/零点保值，未量化张量与源FP16一致；部署文件3496772056字节，SHA256为`2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f`。
- GPTQ原生CUDA算子18项检查通过（0/18/35层的q_proj、o_proj、down_proj，各1/16输入行），最大相对L2误差0.00036312，低于既有0.01阈值。证据：[GPU数值检查](navigation_gptq_kernel_stage2_20261109.json)。该检查对照已量化矩阵，不衡量相对原模型的质量；完整模型请求/导航验收随后通过，见下条。
- 19:28 GPTQ完整接入验收通过：两图701token、九图2815token原长执行，144个原生INT4模块和756个加载张量审计通过；完整1250为81步、0错误、模型STOP、终点距离0.8648m（本单例成功）。单例只用于技术验收，不能据此推断150条成功率或算法优劣。FP16/AWQ也已通过技术验收；各组真实单例结果均保留。
- 独立GPTQ转换器已补static分组及逐列g_idx检查，核对全部未量化张量与源FP16一致；旧AWQ转换器未改。
- FastV/Sparse评分通过3项CPU测试：所需查询行与完整因果GQA注意力的排序一致、多图分隔文本保留、精确保留比例，以及空重要文本的显式均匀分数分支和非法输入拒绝。
- 适配运行时通过2项CPU合成模型测试，覆盖两算法×两比例、第2/3层裁剪、3次DeepStack注入、原mRoPE坐标、输出散回、连续两次decode与KV前缀保持；100%控制与原生forward数值一致。这些不代替真实GPU验收。
- 初筛直接复用原分层第1批50条，复筛追加原第2批至100条，每批11场景；未根据既有成败重选案例。DART两个比例已提取相同50/100子集，并从完整150条重新计算SR/SPL、真实请求总量和显存峰值对照历史报告通过。
- 初次派生导出发现DART50累计100条显存少收第2批采样，已修正并追加检查；旧草稿留存，正式使用本地`reused_dart_v2`。原始日志和实验结果未修改。DART75原恢复采样范围和计时限制继续保留。

## GPTQ 本次检查证据（2026-11-09）

脚本：[check_navigation_gptq_format_cpu.py](../checks/check_navigation_gptq_format_cpu.py)。结果：[navigation_gptq_format_cpu_20261109.json](navigation_gptq_format_cpu_20261109.json)。使用已锁定量化环境：llmcompressor 0.9.0、compressed-tensors 0.13.0、PyTorch 2.8.0、Transformers 4.57.3。

- 对 32×256 和 16×512 的合成 Linear 权重执行真实 GPTQ 函数，使用相关合成输入积累 Hessian，static 排序、group128、非对称 INT4。
- GPTQ → compressed-tensors 往返误差为 0；转 AWQ GEMM 布局后的整数权重、零点和 FP32 反量化值完全一致。
- FP16 scale 舍入引起的最大权重差为 2.8484e-5 / 1.4242e-5，均在计算的舍入误差界内；这不是相对原模型的量化误差或导航效果结果。
- dynamic/group、非连续 g_idx、错误 g_idx 类型和形状均被检查脚本的防护逻辑拒绝；此时尚未验证真实 checkpoint 转换器，后续实际接入情况见上文。这次CPU检查未初始化 CUDA，未读取真实模型或校准集。
- 初次检查脚本把 static 与字符串 `weight` 比较，断言失败；改为库内部实际使用的 `ActivationOrdering.WEIGHT` 枚举比较后通过。此项属于检查脚本的别名处理问题，未修改第三方库。
- 导入时出现 requests 依赖版本警告；本次离线 CPU 数值检查不调用网络，未因此修改锁定环境。结果 JSON 中 warnings 字段只记录 GPTQ 的 Loguru 警告，不代表整个进程没有 Python 警告。

该次CPU检查仅确认格式可行性；后续真实量化、转换、CUDA与完整Qwen3-VL请求已按上文补齐，并通过[七组接入验收](navigation_stage2_integration_20261109.md)。不能仅修改 `quant_method` 字段便宣称完成接入，也不能把CPU或单案例验收当作50/100条性能实验。

## 后续维护与 GitHub 同步

每次适配追加问题编号、版本、解决办法、失败/成功证据与剩余事项。后续同步本台账、必要代码和小型检查结果；原始模型、数据、完整日志和个人续接记录不上传。当前已授权完整适配与50/100实验，但只有真实接入验收通过后才能启动正式初筛。
