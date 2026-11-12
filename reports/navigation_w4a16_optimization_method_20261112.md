# 原W4A16执行优化：方法、验证与适配记录

本轮只优化现有AWQ/GPTQ W4A16的执行路径，并以未量化FP16作对照。没有重新量化、改动冻结权重或恢复原W4A4/W4A8/W8A16全量实验，也没有启动Token压缩组合实验。性能结论与确认实验结果另见[结果报告](navigation_w4a16_optimization_20261112.md)；本文记录方法与验证边界。

## 固定条件与验证范围

| 项目 | 固定设置 |
|---|---|
| 模型与环境 | 项目SimpleVLN Qwen3-VL 4B Dense；RTX 4070 Ti 12GB；LMDeploy 0.11.1、PyTorch 2.5.1+cu121 |
| 请求规模 | batch/TP/DP均为1；上下文4096；输入上限3072，一次完整prefill |
| 实际输入 | 同一轨迹的2图、9图请求，分别为701、2815输入token；不是输入恰好3072 token |
| 输出控制 | 贪心、seed=7，强制18或64个返回token；每组四种输入/输出长度组合 |
| 缓存 | 原生FP16 KV，64块×每块64 token；关闭prefix cache、KV量化及microbatch |
| 图像与位置 | 原生视觉编码、DeepStack、mRoPE和KV更新；视觉token全部保留 |
| 权重身份 | 控制器先核对三份完整checkpoint SHA；服务保存配置SHA、文件大小及加载/执行审计 |

所有新配置使用独立服务和日志目录。原冻结适配器、checkpoint及安装包不修改。新运行类保留旧Stage2的构造、输入检查和权重加载审计，直接调用原生Qwen3-VL的forward与元数据更新，绕过旧Token压缩状态和逐步审计逻辑。FP16、AWQ、GPTQ使用同一套新计时框架，因此本轮stock对照也需要重新测量，不能直接与旧详细审计环境拼表。

## 两项执行改动

**原生decode CUDA Graph。** 旧Stage2类明确返回`support_cuda_graph=False`，只改`eager_mode`不能启用图。新类允许原生Qwen3-VL的decode图路径；图像prefill仍为eager。原生图缓冲复制本次input_ids、mRoPE位置与KV元数据，使用原来的权重和KV张量。

启动顺序为合成prefill预热、decode图外预热、capture；候选算子的M=1形状先完成编译和调优。计时事件放在图调用外，语言/视觉细分计时只用于prefill，避免将逐token日志和审计hook捕获进图。此优化改变执行调度，不应全部归功于量化；评估量化收益须比较相同graph模式的FP16与W4A16。已验证范围仅为原生完整视觉token路径；尚未验证CUDA Graph与DART或其他Token裁剪算法的组合兼容性。

**`phase_tuned`按负载缓存调优结果。** 原生AWQ Triton自动调优缓存键只有N、K；新包装复用同一底层JIT算子、候选配置、INT4打包格式及group128参数，将缓存键改成M、N、K、SPLIT_K，使长prefill与单token decode分别选择配置。没有新增激活量化或常驻FP16权重副本。AWQ和GPTQ各自使用原冻结权重，二者不能混用。

局部筛选还测了`decode_nosplit`、`prefill_dequant`、`hybrid`：分别尝试小M关闭split-K、长prefill逐投影临时反量化后调用FP16 GEMM，以及两者组合。实际第0层四种融合投影、M=1/701/2815的热微测中，关闭split-K令down投影变慢，临时反量化路线未显示足以优先于`phase_tuned`的稳定优势，因此未进入全模型候选。这个淘汰只依据局部微测，不能称为这些路线已经完成全模型或导航验证。微测中的FP16参考是同一量化矩阵反量化后的表示，并非原FP16模型质量对照。

实现：[候选内核](../checks/navigation_w4a16_candidate_kernels.py)、[局部筛选](../checks/probe_navigation_w4a16_fixes.py)、[独立运行类](../checks/navigation_w4a16_fix_runtime.py)。

## 如何证明实际执行，而不是只打开配置

服务首次合成前向生成`execution_audit.json`。AWQ/GPTQ必须证明756个量化源张量加载完整、144个运行投影确实执行，随后移除审计hook。FP16单独记录首次原生前向，不套用量化计数。

`startup_graph_capture.json`记录真实捕获的图键；本次batch1 decode图键为`[1, true, false, 1]`。每条测量的`timing_NNN.json`再记录`eager/capture/replay`实际模式，控制器要求开启graph组的每次测量decode均为replay。不会仅凭`graph=on`宣布成功。

服务对首次两种长度的实际input_ids和pixel_values保存SHA，控制器跨配置核对一致；这些首次请求属于预热。请求日志保存完整返回token序列，检查同配置重复输出以及同一份权重优化前后的逐token一致性。固定两条输入上输出相同是回归检查，不能证明100条闭环导航的SR/SPL不变。

## 两次启动失败及修复

| 尝试 | 观察与原因 | 处理 |
|---|---|---|
| `fullmodel_screen` | 首个FP16服务完成请求，但没有runtime初始化、执行审计或计时文件。自定义映射仅在主进程的`main()`注册，spawn子进程未继承，实际使用了普通原生模型 | 此批不作为完整优化测量；增加启动即检查审计文件的停止条件 |
| `fullmodel_screen_v2` | 将路径传入`PytorchEngineConfig.custom_module_map`仍未生效。安装的0.11.1将该字段复制到`MiscConfig`，而`ModelAgent._build_model`读取`ModelConfig.custom_module_map` | 启动检查在发送测量请求前终止；不改安装包 |
| `fullmodel_screen_v3` | 按此前项目服务的方式，将映射注册移至服务模块顶层，spawn执行`__mp_main__`时也完成注册 | 实际CPU spawn检查选中`navigation_w4a16_fix_runtime.W4A16FixModel`，且`cuda_initialized=false`；之后GPU服务的执行审计、输入SHA和graph replay记录均出现 |

CPU检查通过只说明子进程映射正确，不替代模型运行验证。控制器现在在API就绪后立即检查执行审计；graph组还要求实际捕获记录；缺失即停止，不继续发送整批请求。映射、运行类和服务均为独立新文件。

## 调度尾部计算与确认协议

原生`prefill_interval=16`保持不变。初筛观察到：返回18 token的请求实际执行1次prefill和32次decode；返回64 token的请求执行1次prefill和64次decode。模型工作进程按`loop_count`完成整个decode调度块，收到停止标志后本轮循环也没有立即break，因此返回token数与实际前向次数并非一一对应。

零间隔连续请求时，上一个短输出请求的尾部计算可能进入下一个请求的排队时间，使下一条TTFT不再代表孤立请求。确认控制器在每条请求前等待1秒，等待发生在请求计时开始之前，所有配置采用相同规则，以隔离这类跨请求影响。它是基准测量条件，不是给部署服务人为增加1秒延迟的修复方案，也没有消除尾部多余计算本身。初筛与确认属于不同负载条件，结果分开统计，不合并样本。

确认顺序为GPTQ优化、AWQ优化、FP16 graph、GPTQ stock eager、AWQ stock eager、FP16 eager；每格2次预热、4次测量，另加1条尾请求刷新最后一条计时，每个服务共25次请求。两种输入、两种输出长度及所有请求次序写入`request_plan.json`。最后的刷新请求不进入结果。

计时口径须同时保留：HTTP总耗时、TTFT、首末返回token间隔，以及模型CUDA事件区间。`decode_model_ms_per_token`沿用现有字段名，但分母是**实际decode前向次数**，可包含响应后的计算；stream decode的分母是18输出的17个间隔或64输出的63个间隔。二者不能互换。

没有有效CUPTI CUDA活动轨迹时，CUDA事件区间可能包含主机派发导致的设备空闲，HTTP还含采样、调度和传输；本轮不能给出纯CPU启动开销的精确占比。`nvidia-smi`为整卡采样，含桌面及预热，不等于模型分配显存或精确瞬时峰值。

最终复核另发现FP16长prefill跨服务存在漂移。追加[同服务间隔探针](../checks/probe_navigation_w4a16_idle.py)，交替0/1秒间隔各6次，得到约372.83/375.40ms；这点差异不足以解释首轮314–323ms与确认405–407ms的差距。语言forward主机区间约15ms，具体设备或系统状态原因仍未定位。探针单独保留在[尝试记录](navigation_w4a16_optimization_attempts_20261112.json)，不混入192次主对照，也不把这次较慢的FP16基线当作量化的纯收益。

## WSL复现入口

以下命令在现有WSL环境中执行；同一张卡同时只运行一个控制器或服务。控制器要求新的输出目录，自动核对checkpoint SHA并串行启动、停止其自有服务。UUID用于避免覆盖历史证据。

```bash
cd "/mnt/e/000_博士/000_实习与工作/AI infra/项目2"
python_bin=/home/guzhenqian/.venvs/project2/bin/python

# 仅验证spawn选类，不加载模型，不初始化CUDA。
CUDA_VISIBLE_DEVICES="" "$python_bin" -B \
  checks/serve_navigation_w4a16_fix.py --check-spawn-registration

# 与实际确认协议相同；1秒等待由此控制器在计时外加入。
run_tag="$("$python_bin" -c 'import uuid; print(uuid.uuid4().hex)')"
"$python_bin" -B -u checks/run_navigation_w4a16_fix_confirmation.py \
  --output "logs/navigation_w4a16_fix_20261112/reproduce_${run_tag}" \
  --warmups 2 --repeats 4 --base-port 23610 \
  --sessions gptq:phase_tuned:on awq:phase_tuned:on fp16:stock:on \
             gptq:stock:off awq:stock:off fp16:stock:off
```

也可单独启动一个优化服务，供独立客户端验证；以下示例的API模型名是`w4a16-fix-awq`，地址为`http://127.0.0.1:23630/v1`。同一算法在stock/候选、eager/graph下模型名保持一致。单独服务不自动发送测试请求；首条同长度输入含哈希采集，须预热，最后另发一条图像请求刷新前一条计时。

```bash
run_tag="$("$python_bin" -c 'import uuid; print(uuid.uuid4().hex)')"
run_dir="logs/navigation_w4a16_fix_20261112/manual_${run_tag}"
mkdir -p "$run_dir"
"$python_bin" -B -u checks/serve_navigation_w4a16_fix.py \
  --group awq --variant phase_tuned --graph on \
  --run-dir "$run_dir" --port 23630
```

`--group`支持`fp16/awq/gptq`；FP16只允许`stock`。基线复现使用`--variant stock --graph off`，单独检验图收益使用`stock --graph on`。服务中的phase_tuned是原W4A16实现优化，不是W4A8或重新校准的模型。

发布证据入口：[局部微测与失败记录](navigation_w4a16_optimization_attempts_20261112.json)、[初筛独立核验](navigation_w4a16_optimization_screen_20261112.json)、[确认实验独立核验](navigation_w4a16_optimization_20261112.json)。原始日志仅在本地保留、未上传：`logs/navigation_w4a16_fix_20261112/`下的`projection_probe/`、`fullmodel_screen/`、`fullmodel_screen_v2/`、`fullmodel_screen_v3/`和`confirmation/`分别记录微测、两次失败、有效初筛与确认协议及请求计时。最终性能结论以[优化结果报告](navigation_w4a16_optimization_20261112.md)为准；原精度枚举仍保持停止，本轮没有新的100条导航或Token组合结果。
