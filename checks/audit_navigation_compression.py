"""Audit matched precision/quantization evaluation and report measured tradeoffs."""
import argparse
import csv
import hashlib
import json
import statistics
import struct
from pathlib import Path
from analyze_navigation_sr import audit
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--awq-suite',type=Path);args=p.parse_args()
    fp16_a=ROOT/(ROOT/'tmp/navigation_fp16_latest.txt').read_text().strip()/'A_diagnostic'
    fp16_b=ROOT/(ROOT/'tmp/navigation_fp16_B_latest.txt').read_text().strip()
    baseline={'A':audit(fp16_a),'B':audit(fp16_b)}
    prefix=ROOT/json.loads((fp16_b/'provenance.json').read_text())['prefix_run']
    # Include the interrupted session; continuation-only peak would undercount.
    prefix_peak=max(float(row[1]) for row in csv.reader((prefix/'gpu_samples.csv').read_text().splitlines()) if len(row)>=2)
    baseline['B']['continuation_only_peak_mib']=baseline['B']['whole_card_peak_mib']
    baseline['B']['prefix_peak_mib']=prefix_peak
    baseline['B']['whole_card_peak_mib']=max(prefix_peak,baseline['B']['whole_card_peak_mib'])
    count=0
    for file in prefix.glob('episode_*/steps.jsonl'):
        original=[json.loads(s) for s in file.read_text().splitlines()]
        resumed=[json.loads(s) for s in (fp16_b/file.relative_to(prefix)).read_text().splitlines()]
        assert len(resumed)>=len(original)
        for old,new in zip(original,resumed):
            assert dict(new,prefix_reused=True)==dict(old,prefix_reused=True)
            count+=1
    calibration=json.loads((ROOT/'data/baseline/navigation_awq_calibration_v1/manifest.json').read_text())
    samples=json.loads((ROOT/'data/calibration/navigation_awq_v1/samples.json').read_text())
    assert len(samples)==32
    scenes=set(calibration['scenes'])
    for group in [fp16_a,fp16_b]:
        m=json.loads((group/'manifest.json').read_text())
        assert not scenes.intersection(m['scenes'])
    for sample in samples:
        hashes=[]
        assert [m['role'] for m in sample['messages']]==['system','user']
        for m in sample['messages']:
            for item in m['content']:
                assert set(item) in [{'type','text'},{'type','image'}]
                if item['type']=='image':hashes.append(hashlib.sha256((ROOT/item['image']).read_bytes()).hexdigest())
        assert hashes==sample['image_sha256']
    result=dict(fp16=baseline,prefix_records_preserved=count,calibration_samples=32,calibration_eval_scene_overlap=[])
    lines=['# 导航量化对照实验','',
        '模型：SimpleVLN_Qwen3VL_4B_RL。A/B 各6条固定任务；400步；策略、历史图像、相机、解码与STOP规则不变。',
        '', '现有LMDeploy0.11.1的AWQ后端强制FP16激活，因此新增未量化FP16对照，避免将BF16→FP16的变化归因为INT4。原BF16成绩保留。',
        '', '| 对照 | SR | SPL | 真实请求中位延迟ms | 真实请求P95 ms | 整卡采样峰值MiB |',
        '| --- | --- | --- | --- | --- | --- |']
    for k,v in baseline.items():
        lines.append('| FP16 '+k+' | {:.1%} | {:.4f} | {:.1f} | {:.1f} | {:.0f} |'.format(v['sr'],v['spl'],v['request_median_ms'],v['request_p95_ms'],v['whole_card_peak_mib']))
    if args.awq_suite:
        quant={k:audit(args.awq_suite.resolve()/folder) for k,folder in [('A','A_diagnostic'),('B','B_confirmation')]}
        result['awq']=quant
        comparisons=[]
        for k,v in quant.items():
            before_path=ROOT/baseline[k]['path'];after_path=ROOT/v['path']
            for filename in ['system_prompt.txt','habitat_config.yaml']:
                assert (before_path/filename).read_bytes()==(after_path/filename).read_bytes(),filename
            before_manifest=json.loads((before_path/'manifest.json').read_text())
            after_manifest=json.loads((after_path/'manifest.json').read_text())
            for field in ['episodes','trajectory_ids','scenes','max_episode_steps','success_distance_m','step_size_m','turn_deg','rgb_hw','camera_height_m','hfov_deg','temperature','max_tokens','seed','action_parser','model_inputs']:
                assert before_manifest[field]==after_manifest[field],field
            lines.append('| AWQ '+k+' | {:.1%} | {:.4f} | {:.1f} | {:.1f} | {:.0f} |'.format(v['sr'],v['spl'],v['request_median_ms'],v['request_p95_ms'],v['whole_card_peak_mib']))
            assert [e['episode_id'] for e in baseline[k]['episodes']]==[e['episode_id'] for e in v['episodes']]
            for before,after in zip(baseline[k]['episodes'],v['episodes']):
                assert before['trajectory_id']==after['trajectory_id']
                assert abs(before['start_distance_m']-after['start_distance_m'])<1e-5
                comparisons.append(dict(group=k,episode_id=before['episode_id'],fp16_success=int(before['metrics']['success']),
                    awq_success=int(after['metrics']['success']),fp16_spl=before['metrics']['spl'],awq_spl=after['metrics']['spl'],
                    fp16_steps=before['steps'],awq_steps=after['steps'],awq_failure_category=after['failure_category']))
        result['paired_episodes']=comparisons
        result['runtime_valid']=all(v['errors']==0 and v['truncations']==0 for v in list(baseline.values())+list(quant.values()))
        lost=[r for r in comparisons if r['fp16_success'] and not r['awq_success']]
        gained=[r for r in comparisons if r['awq_success'] and not r['fp16_success']]
        result['decision']=dict(adopted=False,status='retain original navigation baseline; first AWQ candidate has observed quality regression',
            lost_successes=[r['group']+':'+r['episode_id'] for r in lost],gained_successes=[r['group']+':'+r['episode_id'] for r in gained],
            completed_local_date='2026-10-31',scope='engineering decision on small fixed subset, not a claim about all AWQ methods')
        lines[2:2]=['**本轮结论（2026-10-31）：首个AWQ候选已完成，但不替换原导航基线。** 它降低了权重大小与整卡显存、略微缩短了请求时间；本轮12任务丢失{}条原成功、新增{}条成功，质量代价不接受。保留全部失败结果，后续Token压缩应从未量化基线单独开始；不在B组反复调AWQ后宣称独立验证。'.format(len(lost),len(gained)), '']
        conversion=json.loads((ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_AWQ/conversion_audit.json').read_text())
        kernel=json.loads((ROOT/'reports/navigation_awq_kernel.json').read_text())
        assert conversion['quantized_layer_count']==252 and kernel['status']=='passed'
        with (ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy/model.safetensors').open('rb') as stream:
            header=json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]))
        tensors=[v for k,v in header.items() if k!='__metadata__']
        assert all(v['dtype']=='F32' for v in tensors)
        fp16_tensor_bytes=sum((v['data_offsets'][1]-v['data_offsets'][0])//2 for v in tensors)
        result['weights']=dict(awq_file_bytes=conversion['bytes'],source_fp16_tensor_bytes=fp16_tensor_bytes,
            reduction_vs_fp16=1-conversion['bytes']/fp16_tensor_bytes,awq_sha256=conversion['sha256'],
            note='AWQ file includes header; FP16 tensor bytes computed from original F32 header, no duplicate output head',
            max_scale_relative_rounding=max(v['fp16_scale_max_relative_rounding'] for v in conversion['layers']),
            kernel_checks=len(kernel['checks']),kernel_max_relative_l2=max(v['relative_l2'] for v in kernel['checks']))
        replays=[json.loads((ROOT/('reports/navigation_replay_'+variant+'.json')).read_text()) for variant in ['fp16','awq']]
        pairs=[]
        assert len(replays[0]['requests'])==len(replays[1]['requests'])==48
        for before,after in zip(replays[0]['requests'],replays[1]['requests']):
            assert before['payload_sha256']==after['payload_sha256']
            assert before['image_sha256']==after['image_sha256']
            assert before['usage']['prompt_tokens']==after['usage']['prompt_tokens']
            assert before['finish_reason']==after['finish_reason']=='stop'
            pairs.append(before['request_s']/after['request_s'])
        assert len(pairs)==48
        result['fixed_input_replay']=dict(fp16_median_s=replays[0]['median_s'],awq_median_s=replays[1]['median_s'],paired_speed_ratio_median=statistics.median(pairs),requests=48)
        result['fixed_input_replay']['completion_tokens']={name:sum(r['usage']['completion_tokens'] for r in replay['requests']) for name,replay in zip(['fp16','awq'],replays)}
        lines+=['','## 相同输入速度对照','', '24条固定B组原始输入，每条2次，逐项核对文本与JPEG哈希；前2次预热排除。FP16中位{:.3f}s；AWQ中位{:.3f}s；配对速度比中位{:.3f}。请求可能生成不同长度文本，此为真实HTTP决策耗时，不是纯GPU吞吐。'.format(replays[0]['median_s'],replays[1]['median_s'],statistics.median(pairs))]
        lines+=['','## 权重与计算检查','',
            '语言模型252个线性模块W4A16，group128非对称；视觉、嵌入和其他浮点模块保持FP16。AWQ文件{:.3f}GB；源模型以FP16表示的张量约{:.3f}GB，减少{:.1%}。原下载文件是F32，不能把它的17.75GB直接当作FP16部署基线。'.format(conversion['bytes']/1e9,fp16_tensor_bytes/1e9,result['weights']['reduction_vs_fp16']),
            '', '整数与零点转换逐项一致；9个实际投影层分别测试decode/prefill，共18次真实CUDA计算，与独立反量化矩阵乘最大相对L2误差{:.6f}。这验证布局/内核，不代表量化质量无损。'.format(result['weights']['kernel_max_relative_l2']),
            '', '校准使用FP16舍入后的源权重、FP32算术；前一轮FP16尺度搜索在后层溢出，失败日志保留。模型部署仍FP16激活；未用A/B成绩反复搜索校准方案。',
            '', '全部运行无请求错误或截断：{}。'.format(result['runtime_valid']),
            '', '## 逐任务配对结果','', '| 组 | Episode | FP16成功 | AWQ成功 | FP16 SPL | AWQ SPL | FP16步数 | AWQ步数 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in comparisons:
            lines.append('| {group} | {episode_id} | {fp16_success} | {awq_success} | {fp16_spl:.4f} | {awq_spl:.4f} | {fp16_steps} | {awq_steps} |'.format(**row))
        divergence_path=ROOT/'reports/navigation_awq_divergence.json'
        if divergence_path.exists():
            divergence=json.loads(divergence_path.read_text())
            result['lost_success_analysis']=divergence
            lines+=['','## 为什么没有采用这个候选','',
                '已排除本轮请求失败、动作解析错误、输出截断和转换整数布局错误。质量下降出现在正常返回动作后的闭环轨迹中。下面记录的是可观察到的失败过程，不能仅凭单种子12任务确定每一层误差贡献或泛化损失。',
                '', 'A196是一个直接例子：两版在同一第16次观察位置、同一组JPEG输入下，FP16回答“forward 25 cm, turn right 15 degree”，AWQ回答“forward 50 cm, turn right 15 degree”。执行到第18个原子动作（日志step17，从0计）时，FP16转弯，AWQ继续前进，后续画面随之不同。FP16在58步成功；AWQ最终400步失败、累计361次碰撞。不能据此断言这25cm差异是唯一原因，但它明确显示了早期决策差异如何进入不同的闭环轨迹。',
                '', '另外两条丢失成功：B97在75步提前STOP，终距7.61m；B112曾到2.90m附近，后来在4.31m处STOP而失败。后两条在动作首次分歧时，其最近请求的历史/队列长度已不同，不能称为同输入输出对照。',
                '', '减小权重文件并不等于等比例加速：视觉编码、FP16激活、历史图像处理、输出生成和HTTP调度仍在，AWQ还有反量化内核开销。这里未单独测各环节时间，不能精确分摊瓶颈；同输入计时仅支持本机约1.11倍的配对速度比。FP16与AWQ共48次生成分别808/890个token，输出长度也不完全相同。']
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        positions=[0,1];width=0.34
        for ax,metric,title in [(axes[0,0],'sr','Navigation success (6 episodes per group)'),(axes[0,1],'spl','SPL (same fixed episodes)'),(axes[1,1],'whole_card_peak_mib','Whole-card sampled peak (GiB)')]:
            scale=1024 if metric=='whole_card_peak_mib' else 1
            ax.bar([p-width/2 for p in positions],[baseline[k][metric]/scale for k in ['A','B']],width,label='FP16',color='#3569a8')
            ax.bar([p+width/2 for p in positions],[quant[k][metric]/scale for k in ['A','B']],width,label='AWQ W4A16',color='#d88032')
            ax.set_xticks(positions);ax.set_xticklabels(['A diagnostic','B confirmation']);ax.set_title(title,fontsize=10)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if metric in ['sr','spl']:ax.set_ylim(0,1)
        axes[0,0].legend(fontsize=8)
        axes[1,0].bar(['FP16','AWQ W4A16'],[replays[0]['median_s'],replays[1]['median_s']],color=['#3569a8','#d88032'])
        axes[1,0].set_title('Fixed-input HTTP median (s), 24 inputs x2',fontsize=10)
        axes[1,0].grid(axis='y',alpha=.2);axes[1,0].set_axisbelow(True)
        fig.suptitle('SimpleVLN Qwen3-VL 4B RL: first AWQ comparison\nSmall engineering subset, not full benchmark',fontsize=12)
        fig.tight_layout();fig.savefig(str(ROOT/'reports/navigation_compression_20261030.png'),dpi=170);plt.close(fig)
        lines+=['','![量化对照图](navigation_compression_20261030.png)']
    else:
        lines+=['','AWQ尚未完成，当前不报告量化质量、显存或加速收益。']
    lines+=['','## 数据与边界','',
        '校准：官方train前4个符合距离筛选的场景，每场景2条不同轨迹，每轨迹4个观察，共32条。与A/B评测场景交集为0。参考路径只用于生成校准观察，模型输入仅为原作者格式的文字与当前/历史图像，不含动作标签、目标或距离。不训练模型。',
        '', 'FP16 B组因进程中断而续跑；重放并逐像素/位姿检查原始前缀，共{}条记录内容保留。续跑组的耗时包含不同进程会话，旧前缀的elapsed_s不表示完整重跑时间；不把重放速度算成模型速度。'.format(count),
        '', 'A曾用于模型诊断，B已有基线成绩，均属于固定工程对照集；不是新盲测或全量benchmark。仅各6条、单种子采样，不能从差异宣称统计显著。峰值显存为整卡200ms采样，含桌面，且缓存按剩余显存自动分配；不是模型独占参数占用。',
        '', '方法依据：[LLM Compressor AWQ](https://github.com/vllm-project/llm-compressor/tree/0.9.0/examples/awq)。实现固定llmcompressor0.9.0、compressed-tensors0.13.0、torch2.8.0、transformers4.57.3；部署仍为已验证的LMDeploy0.11.1独立环境。']
    (ROOT/'reports/navigation_compression_20261030.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'reports/navigation_compression_20261030.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('COMPRESSION_AUDIT_READY',[(k,v['sr'],v['spl']) for k,v in baseline.items()],count)

if __name__=='__main__':main()
