"""Stdlib-only audit/binding of the completed AWQ expanded-INT8 micro screen.

No checkpoint is opened and no GPU library is imported. The prospective mixed
prefill experiment may use only qkv/o with NK layout. This is recorded micro
evidence, not independent tensor-byte verification or model-quality approval.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = '31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db'
SOURCES = {
    'probe_navigation_latency_expanded_int8_prefill_20261120.py': '255dfde2854a873d3065823e073b8a044611810e9134bdf9d8052fbfddc90bb8',
    'navigation_latency_expanded_int8_prefill_20261120.py': '7419f76d0203bf4dcc156958258692262711df0be8cf25793aad748644d8cda4',
    'navigation_stage2_common.py': '473c7ba7548892eaebd31916d494484238c8c66586e6a22d1d000f26a32474c1',
    'navigation_w4a16_repair_kernels.py': '09a181c19e618307f681e02a8ee6964c6ad368a0d47e772d7328f297971e9a87',
    'navigation_w4a16_candidate_kernels.py': 'b38189a9beebec838ea56f059d0e5c2693a13d1e67daae907b6266716ae37e38',
}
SHAPES = {'qkv': (2560,6144), 'o': (4096,2560), 'gate_up': (2560,19456), 'down': (9728,2560)}
LAYOUTS = ('kn_contiguous', 'nk_transposed')
FILES = ('protocol.json', 'result.json', 'baseline_replayed_table.json', 'conversions.json',
         'attempts.jsonl', 'numerical_baselines.json', 'measurements.json', 'boundaries.json')


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def sample_indices(size,count):
    values=[0,size-1,size//2]
    for i in range(1,count*2):
        value=i*size//count
        if value not in values: values.append(value)
        if len(values)==count: break
    return sorted(values)


def error_gate(value,limit):
    relative,maximum=value['relative_l2'],value['max_absolute_error']
    require(value['threshold']==limit and isinstance(value['finite'],bool), 'Changed numerical threshold/finite flag')
    finite_numbers=all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (relative,maximum))
    passed=value['finite'] and finite_numbers and relative<=limit
    require(value['passed'] is passed, 'Saved error gate differs from error/threshold')
    return passed


def timing_gate(block,count,cold):
    values=block['samples_ms']
    require(len(values)==count and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values), 'Invalid timing count/value')
    require(block['mode']=='cuda_graph_replay' and block['cache_condition']==('eviction_before_each_sample' if cold else 'hot') and
            block['eviction_buffer_bytes']==(96*1048576 if cold else 0) and block['flush_in_timed_interval'] is False,
            'Timing mode, eviction or boundary differs')
    require(math.isclose(block['cuda_p50_ms'],statistics.median(values),rel_tol=1e-12), 'Saved timing median differs')
    return len(values) if cold else 0


def load_contract(probe_dir, algorithm, checkpoint_sha, verify_sources=True):
    """Return a JSON-serializable micro proof. Service must verify_sources=True.

    The caller must independently verify its actual checkpoint and full runtime
    module conversions; this loader checks only saved tensor identities.
    """
    directory=Path(probe_dir)
    if not directory.is_absolute(): directory=ROOT/directory
    directory=directory.resolve()
    require(directory.is_relative_to(ROOT/'logs'), 'Probe must be inside project logs')
    require(algorithm=='awq' and checkpoint_sha==CHECKPOINT, 'Only the audited AWQ checkpoint is supported')
    require(not (directory/'failure.json').exists(), 'Microprobe retained a fatal failure')
    docs={name:read(directory/name) for name in FILES if name!='attempts.jsonl'}
    attempts=[json.loads(line) for line in (directory/'attempts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    protocol,result=docs['protocol.json'],docs['result.json']
    require(result['status']=='completed_isolated_micro_screen' and result['protocol']==protocol, 'Incomplete micro result/protocol')
    require(protocol['schema']=='navigation_expanded_int8_prefill_probe/v1' and protocol['group']==algorithm and
            protocol['model_weights_sha256']==checkpoint_sha and protocol['sources_sha256']==SOURCES, 'Micro model/source schema differs')
    require(protocol['layers']==[0,18,35] and protocol['rows']==[17,701,2815] and protocol['amplitudes']==[.1,1.] and
            protocol['projections']==list(SHAPES) and protocol['layouts']==list(LAYOUTS) and
            protocol['logical_cases']==72 and protocol['layout_execution_cases']==144, 'Micro coverage was changed')
    require(protocol['approximation_relative_l2_limit']==.02 and protocol['native_reference_relative_l2_limit']==.005 and
            protocol['pad_m_multiple']==0 and protocol['stability_repeats']==4 and protocol['timing_repeats']==20 and
            protocol['cold_repeats']==10 and protocol['cold_cache_mib']==96 and protocol['seed']==20261120,
            'Predeclared numerical/timing/padding protocol differs')
    require(protocol['versions']==dict(torch='2.5.1+cu121',triton='3.1.0',lmdeploy='0.11.1',numpy='2.5.2') and
            protocol['gpu']['capability']==[8,9], 'Producer version/architecture differs')
    require(protocol['full_model_adapter_created'] is False and protocol['original_checkpoint_modified'] is False and
            result['service_table'] is None and result['whole_model_quality']=='not_evaluated' and
            result['actual_activation_quality']=='not_evaluated', 'Micro evidence cannot approve a full model')
    for name in ('measurements','conversions','boundaries'):
        require(result[name]==docs[name+'.json'], 'Micro sidecar/result disagreement: '+name)
    require(all(result['final_source_checks'].values()) and
            set(result['final_source_checks'])==set(SOURCES)|{'native_awq_source','baseline_table','design'}, 'Incomplete final source checks')
    baseline_path=Path(protocol['baseline_table'])
    require(protocol['baseline_table_sha256']=='8ccdafddb1bcef0397198ca9eb7c3c9f1f7d615071ca81cfb514d19a4c642783', 'Unapproved native fallback table')
    require(protocol['native_awq_source']['sha256']=='0e53a5ae32780e411d4a4f6802968e1cd6aa29048357a77c609b969bf7b8ab18' and
            protocol['design_sha256']=='6834a94ccc825c950066ed5c3dbacadf06a1950d5f48f1cd869100853b589fa5', 'Underlying JIT/design identity differs')
    if verify_sources:
        for name,sha in SOURCES.items(): require(digest(ROOT/'checks'/name)==sha, 'Micro source changed: '+name)
        require(baseline_path.resolve().is_relative_to(ROOT) and digest(baseline_path)==protocol['baseline_table_sha256'], 'Native baseline source changed')
        require(digest(protocol['native_awq_source']['path'])==protocol['native_awq_source']['sha256'], 'Native JIT source changed')
        require(digest(ROOT/'reports/navigation_latency_expanded_int8_prefill_design_20261120.json')==protocol['design_sha256'], 'Predeclared design changed')
        baseline=read(baseline_path);baseline=baseline.get('frozen_launch_table',baseline)
        actual=docs['baseline_replayed_table.json']
        require(actual['shapes']==baseline['shapes'] and actual['representatives']==baseline['representatives'], 'Native actual table configs differ')
    actual=docs['baseline_replayed_table.json']
    require(actual['algorithm']==algorithm and actual['model_weights_sha256']==checkpoint_sha and actual['frozen'] and
            actual['actual_m_preserved'] and actual['generic_m_specialization_disabled'] and
            not actual['runtime_autotuning_allowed'] and not actual['runtime_cold_compilation_allowed'] and
            actual['selection_mode']=='replayed_frozen_table' and actual['frozen_table_source']['sha256']==protocol['baseline_table_sha256'],
            'Native replay was not frozen/actual-M-preserving')

    tensors={}; conversions={}; expected_conversion={(l,p,t) for l in (0,18,35) for p in SHAPES for t in LAYOUTS}
    for value in docs['conversions.json']:
        key=(value['layer'],value['projection'],value['layout']); require(key in expected_conversion and key not in conversions, 'Conversion coverage duplicate/missing')
        l,p,layout=key;k,n=SHAPES[p]
        require((value['k'],value['n'])==(k,n) and value['stride']==([n,1] if layout=='kn_contiguous' else [1,k]) and
                value['resident_weight_bytes']==k*n and value['resident_scale_bytes']==n*4, 'Expanded conversion layout/bytes differ')
        require(valid_sha(value['expanded_weight_logical_sha256']) and valid_sha(value['channel_scale_sha256']), 'Missing expanded tensor SHA')
        packed=value['tensors'];require(set(packed)=={'qweight','scales','qzeros'}, 'Packed tensor inventory differs')
        for name,shape,dtype in (('qweight',[k,n//8],'torch.int32'),('scales',[k//128,n],'torch.float16'),('qzeros',[k//128,n//8],'torch.int32')):
            require(packed[name]['shape']==shape and packed[name]['dtype']==dtype and valid_sha(packed[name]['sha256']), 'Packed tensor metadata/SHA malformed')
        if (l,p) in tensors:require(tensors[(l,p)]==packed, 'Packed tensor SHA differs across layouts')
        tensors[(l,p)]=packed
        error_gate(value['weight_error'],.02)
        conversions[key]=value
    require(set(conversions)==expected_conversion and len(conversions)==len(docs['conversions.json'])==24, 'Require 24 conversions')
    for l,p in tensors:
        a,b=(conversions[(l,p,t)] for t in LAYOUTS)
        require(a['expanded_weight_logical_sha256']==b['expanded_weight_logical_sha256'] and
                a['channel_scale_sha256']==b['channel_scale_sha256'], 'Physical layouts have different logical weights/scales')
    expected={(l,p,m,a) for l in (0,18,35) for p in SHAPES for m in (17,701,2815) for a in (.1,1.)}
    baselines={}
    for value in docs['numerical_baselines.json']:
        key=(value['layer'],value['projection'],value['m'],value['amplitude']);require(key in expected and key not in baselines, 'Native numerical coverage differs')
        require((value['k'],value['n'])==SHAPES[value['projection']] and value['tensors']==tensors[key[:2]] and valid_sha(value['input_sha256']), 'Native shape/input/tensor identity differs')
        require(error_gate(value['native_error'],.005) and error_gate(value['dequant_fp16_gemm_error'],.005), 'A native reference failed')
        baselines[key]=value
    require(set(baselines)==expected and len(baselines)==result['numerical_baseline_cases']==72, 'Require 72 native reference cases')
    boundaries={};boundary_names={'positive_bound','negative_bound','zero_row','ties_to_even','zero_weight'}
    for value in docs['boundaries.json']:
        key=(value['layout'],value['case']);require(key not in boundaries and key[0] in LAYOUTS and key[1] in boundary_names, 'Boundary coverage differs')
        require(value['passed'] is True and 'error' not in value, 'Semantic boundary failed')
        if value['case']!='zero_weight':
            require(set(value['flags'])=={'quantized_exact','scale_exact','integer_exact','half_exact','finite'} and all(v is True for v in value['flags'].values()) and
                    value['m']==value['padded_m']==17 and value['pad_m_multiple']==0 and value['n']==8 and
                    value['k']==(9728 if value['case'] in ('positive_bound','negative_bound') else 128) and
                    value['max_abs_expected_int']<value['int32_limit']==2**31-1, 'Boundary exact/finite/int32 gate differs')
        boundaries[key]=value
    require(len(boundaries)==10, 'Require five semantic boundaries per layout')

    seen={};failed=[];stability_flushes=0;implementation_flags={'packed_fp16_dequant_exact','activation_scale_exact','activation_quant_exact',
        'weight_scale_exact','weight_quant_exact','int32_accumulator_exact','ordered_fp32_epilogue_half_exact'}
    for value in attempts:
        key=(value['layer'],value['projection'],value['m'],value['amplitude'],value['layout'])
        require(key[:4] in expected and key[4] in LAYOUTS and key not in seen, 'Attempt coverage duplicate/missing')
        l,p,m,a,layout=key;k,n=SHAPES[p];base=baselines[key[:4]];conv=conversions[(l,p,layout)]
        require((value['k'],value['n'])==(k,n) and value['tensors']==base['tensors'] and value['input_sha256']==base['input_sha256'] and
                value['pad_m_multiple']==0 and value['padded_m']==m and value['weight_error']==conv['weight_error'], 'Attempt input/tensor/padding differs')
        impl=value['implementation']
        require(impl['passed'] is True and set(impl['flags'])==implementation_flags and all(v is True for v in impl['flags'].values()) and
                impl['rows']==sample_indices(m,4) and impl['columns']==sample_indices(n,8) and impl['k']==k and
                impl['int32_safe'] is True and impl['theoretical_max_abs_integer']==k*127*127<2**31, 'Selected implementation gate failed')
        for left,right in (('int64_expected','int32_actual'),('cpu_expected_half_bits','actual_half_bits')):
            require(impl[left]==impl[right] and len(impl[left])==4 and all(len(row)==8 for row in impl[left]), 'Saved selected independent implementation arrays differ')
        require(all(type(v) is int and abs(v)<=k*127*127 for row in impl['int32_actual'] for v in row) and
                all(type(v) is int and 0<=v<=65535 for row in impl['actual_half_bits'] for v in row), 'Malformed integer/half-bit receipt')
        approx=value['approximation']; gates={name:error_gate(approx[name],.02) for name in ('weight_only','activation_only','combined','activation')}
        passed=error_gate(value['weight_error'],.02) and gates['activation'] and gates['combined']
        require(value['approximation_screen_passed'] is passed and value['semantic_boundaries_passed'] is True,
                'Approximation or semantic gate flag differs')
        stability=value['stability'];require(stability['passed'] is True and valid_sha(stability['reference_output_sha256']), 'Repeat stability failed')
        modes=set()
        for condition in stability['conditions']:
            ident=(condition['mode'],condition['cache_condition']);require(ident not in modes and ident[0] in ('eager','cuda_graph_replay') and ident[1] in ('hot','cold'), 'Stability mode coverage differs')
            require(condition['repeats']==len(condition['full_tensor_bytewise_equal'])==4 and condition['passed'] is True and
                    condition['eviction_buffer_bytes']==(96*1048576 if ident[1]=='cold' else 0), 'Stability repeat/eviction count differs')
            require(all(set(trial)=={'qx','row_scale','integer','output'} and all(v is True for v in trial.values()) for trial in condition['full_tensor_bytewise_equal']), 'A full tensor bytewise repeat differs')
            stability_flushes+=4 if ident[1]=='cold' else 0;modes.add(ident)
        require(len(modes)==len(stability['conditions'])==4, 'Stability conditions incomplete')
        require(value['status']==('accepted_random_micro_case' if passed else 'rejected_random_micro_case'), 'Attempt status differs from strict gate')
        if not passed:
            failed.append(dict(layer=l,projection=p,m=m,amplitude=a,layout=layout,
                failed_gates=[name for name,ok in dict(weight=error_gate(value['weight_error'],.02),activation=gates['activation'],combined=gates['combined']).items() if not ok],
                weight_relative_l2=value['weight_error']['relative_l2'],combined_relative_l2=approx['combined']['relative_l2']))
        seen[key]=value
    require(len(seen)==len(attempts)==144 and set(seen)=={(*key,t) for key in expected for t in LAYOUTS}, 'Require all 144 layout cases')
    counts=dict(Counter(v['status'] for v in attempts));require(counts==result['attempt_status_counts'], 'Saved status totals differ')
    require(counts=={'accepted_random_micro_case':120,'rejected_random_micro_case':24} and
            {(v['layer'],v['projection']) for v in failed}=={(35,'gate_up'),(35,'down')}, 'Evidence differs from the frozen 120/24 screen; no automatic expansion allowed')

    time_rows=[];times={};timing_flushes=0
    for value in docs['measurements.json']:
        key=(value['projection'],value['layout'],value['m']);require(key not in times and key[0] in SHAPES and key[1] in LAYOUTS and key[2] in (701,2815), 'Timing coverage differs')
        p,layout,m=key;attempt=seen[(0,p,m,1.,layout)]
        require(value['layer']==0 and attempt['status']=='accepted_random_micro_case' and value['input_sha256']==attempt['input_sha256'] and
                value['tensors']==attempt['tensors'] and (value['k'],value['n'])==SHAPES[p] and value['pad_m_multiple']==0 and value['padded_m']==m,
                'Timing case is not the numerically accepted input/config')
        require(attempt['timing_measurement_index']==len(times), 'Timing ledger index differs')
        recomputed={};gains={}
        require(set(value['confirmation'])=={'hot','cold'}, 'Timing cache modes incomplete')
        for condition,groups in value['confirmation'].items():
            require(set(groups)=={'native_w4a16','same_w4_dense_fp16','expanded_int8_full'}, 'Timing comparator differs')
            recomputed[condition]={}
            for name,blocks in groups.items():
                require(len(blocks)==2, 'Two independent timing rounds required')
                samples=[]
                for index,block in enumerate(blocks):
                    require(block['round']==index and len(block['execution_order'])==3 and set(block['execution_order'])==set(groups), 'Timing randomized round order malformed')
                    require(all(groups[other][index]['execution_order']==block['execution_order'] for other in groups), 'Timing modes disagree on round order')
                    timing_flushes+=timing_gate(block,10 if condition=='cold' else 20,condition=='cold');samples.extend(block['samples_ms'])
                median=statistics.median(samples);require(math.isclose(median,value['p50_ms'][condition][name],rel_tol=1e-12), 'Pooled two-round median differs')
                recomputed[condition][name]=median
            gains[condition]={name:1-recomputed[condition]['expanded_int8_full']/recomputed[condition][name] for name in ('native_w4a16','same_w4_dense_fp16')}
            require(all(math.isclose(gain,value['gain_fraction'][condition][name],rel_tol=1e-12,abs_tol=1e-15) for name,gain in gains[condition].items()), 'Timing gain recomputation differs')
        require(value['timing_screen_passed'] is all(g>=.03 for group in gains.values() for g in group.values()), '3 percent full-path speed gate differs')
        require(set(value['diagnostic_hot_components'])=={'activation_quant','int_mm_only','epilogue_only'}, 'Diagnostic components differ')
        for block in value['diagnostic_hot_components'].values():timing_gate(block,20,False)
        times[key]=value;time_rows.append(dict(projection=p,layout=layout,m=m,p50_ms=recomputed,gain_percent={c:{k:100*v for k,v in g.items()} for c,g in gains.items()},
            timing_screen_passed=value['timing_screen_passed'],eligible_projection=p in ('qkv','o') and layout=='nk_transposed'))
    require(len(times)==16, 'Require both measured M and layouts for all projections')
    groups=[];expected_groups={(p,t) for p in SHAPES for t in LAYOUTS};observed_groups=set()
    for value in result['per_projection_layout']:
        p,layout=value['projection'],value['layout'];require((p,layout) in expected_groups and (p,layout) not in observed_groups, 'Summary group duplicate/unknown')
        cases=[v for k,v in seen.items() if k[1]==p and k[4]==layout]
        gates=len(cases)==18 and all(v['status']=='accepted_random_micro_case' for v in cases)
        promising=gates and all(times[(p,layout,m)]['timing_screen_passed'] for m in (701,2815))
        require(value['cases']==18 and value['all_random_micro_gates_passed'] is gates and value['promising_for_next_investigation'] is promising, 'Projection advancement decision differs')
        groups.append(dict(projection=p,layout=layout,all_random_micro_gates_passed=gates,promising_for_next_investigation=promising,
            max_weight_relative_l2=max(v['weight_error']['relative_l2'] for v in cases),
            max_activation_relative_l2=max(v['approximation']['activation']['relative_l2'] for v in cases),
            max_combined_relative_l2=max(v['approximation']['combined']['relative_l2'] for v in cases)))
        observed_groups.add((p,layout))
    require(observed_groups==expected_groups and {(v['projection'],v['layout']) for v in groups if v['promising_for_next_investigation']}=={('qkv','nk_transposed'),('o','nk_transposed')},
            'Only qkv/o NK may advance under the original gates')
    cold=result['cold_cache'];counts_flush=dict(stability=stability_flushes,timing=timing_flushes);total=sum(counts_flush.values())
    require(cold['passed'] is True and cold['bytes']==96*1048576 and cold['flush_counts_by_purpose']==counts_flush and cold['flush_count']==total and
            cold['expected_final_uint8_value']==total%256 and cold['checksum']==cold['expected_checksum']==(total%256)*cold['bytes'], 'Actual cache flush ledger/checksum differs')
    expansion=result['all144_weight_expansion_estimate'];int8_bytes=36*sum(k*n for k,n in SHAPES.values());scale_bytes=36*4*sum(n for k,n in SHAPES.values())
    require(expansion['int8_bytes']==int8_bytes and expansion['fp32_scale_bytes']==scale_bytes and expansion['measured_fullmodel_peak'] is False and
            math.isclose(expansion['total_gib'],(int8_bytes+scale_bytes)/2**30), 'Weight-only memory estimate differs')
    return dict(status='verified_micro_contract' if verify_sources else 'document_checks_only_sources_unverified',
        eligible_for_runtime_binding=bool(verify_sources),algorithm=algorithm,model_weights_sha256=checkpoint_sha,
        allowed_projections=['qkv','o'],layout='nk_transposed',allowed_measured_m=[701,2815],
        predeclared_approximation_relative_l2_limit=.02,full_model_approval=False,actual_activation_quality='not_evaluated',
        actual_checkpoint_independently_read=False,prior_w4_same_weight_output_gate_inherited=False,
        recorded_packed_tensor_receipts={f'{l}/{p}':v for (l,p),v in tensors.items()},
        recorded_expanded_weight_receipts={f'{l}/{p}/{t}':{name:v[name] for name in ('expanded_weight_logical_sha256','channel_scale_sha256','weight_error','stride')} for (l,p,t),v in conversions.items()},
        probe_directory=directory.relative_to(ROOT).as_posix(),evidence_sha256={name:digest(directory/name) for name in FILES},
        producer_sources_sha256=SOURCES,native_awq_source=protocol['native_awq_source'],design_sha256=protocol['design_sha256'],
        baseline_table_sha256=protocol['baseline_table_sha256'],native_reference_cases=72,layout_cases=144,
        implementation_cases_passed=144,stability_cases_passed=144,semantic_boundary_cases_passed=10,
        attempt_status_counts=counts,rejected_cases=failed,projection_layout_summary=groups,timing_rows=time_rows,
        cold_cache_ledger_verified=cold,
        memory=dict(all144_weight_and_scale_estimate=expansion,allowed72_weight_bytes=36*sum(math.prod(SHAPES[p]) for p in ('qkv','o')),
            allowed72_scale_bytes=36*4*sum(SHAPES[p][1] for p in ('qkv','o')),full_model_fit_verified=False),
        auditor_sha256=digest(__file__),limitations=[
            'Recorded producer receipts were recomputed and source-bound; actual checkpoint bytes were not independently opened by this audit.',
            'Only qkv/o NK passes every original gate. Layer35 gate_up/down failures remain excluded even when layer0 timing is faster.',
            'This is new W8/A8 approximation in prefill; keeping W4 decode makes a new mixed-precision model. No exact-token or navigation-quality approval is inherited.',
            'The 2 percent screen covers reconstructed weights, activations and combined random output. Weight-only/activation-only output errors are recorded diagnostics.',
            'The dense FP16 comparator uses dequantized W4 weights, not the original FP16 model checkpoint.',
            'Graph-replay projection timings exclude startup conversion/capture and cannot establish eager-prefill, HTTP or navigation gains.',
            'Steady-state timing includes activation quantization, whole-K GEMM and epilogue. Cold eviction is outside events; components are not summed.',
            'Micro memory retains both layouts and dense references. Weight-only estimates omit original W4 decode, KV cache, activations and workspaces.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    require(output.is_relative_to(ROOT) and output.suffix=='.json' and not output.exists(), 'Use fresh project JSON output')
    proof=load_contract(args.probe,'awq',CHECKPOINT)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(proof,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=proof['status'],allowed=proof['allowed_projections'],layout=proof['layout'],counts=proof['attempt_status_counts'],output=str(output))))


if __name__=='__main__':main()
