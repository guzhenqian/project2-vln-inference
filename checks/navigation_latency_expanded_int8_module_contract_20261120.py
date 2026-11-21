"""Stdlib contract for a NEW per-module W4/W8A8 prefill exploration.

Original projection-class failures are retained. A new fixed mask may select an
individual module only after all six independently seeded cases pass. This is
not full-model approval and never permits per-input numerical fallback.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'navigation_expanded_int8_module_screen/v1'
CHECKPOINT = '31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db'
SOURCES = {
    'probe_navigation_latency_expanded_int8_prefill_20261120.py':'255dfde2854a873d3065823e073b8a044611810e9134bdf9d8052fbfddc90bb8',
    'navigation_latency_expanded_int8_prefill_20261120.py':'7419f76d0203bf4dcc156958258692262711df0be8cf25793aad748644d8cda4',
    'navigation_latency_int8_attention_contract_20261120.py':'2df06ce9fe8a15c93d790b6bdcc71cf42f941b25e2081d17b73a4a55831767a6',
    'navigation_stage2_common.py':'473c7ba7548892eaebd31916d494484238c8c66586e6a22d1d000f26a32474c1',
    'navigation_w4a16_repair_kernels.py':'09a181c19e618307f681e02a8ee6964c6ad368a0d47e772d7328f297971e9a87',
    'navigation_w4a16_candidate_kernels.py':'b38189a9beebec838ea56f059d0e5c2693a13d1e67daae907b6266716ae37e38',
}
SHAPES = {'qkv':(2560,6144),'o':(4096,2560),'gate_up':(2560,19456),'down':(9728,2560)}
ROWS = (17,701,2815)
AMPLITUDES = (.1,1.)
LIMIT = .02
LAYOUT = 'nk_transposed'
BOUNDARIES = ('positive_bound','negative_bound','zero_row','ties_to_even','zero_weight')
IMPL_FLAGS = {'packed_fp16_dequant_exact','activation_scale_exact','activation_quant_exact',
              'weight_scale_exact','weight_quant_exact','int32_accumulator_exact','ordered_fp32_epilogue_half_exact'}


def require(value,message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def verify_sources():
    for name,expected in SOURCES.items():
        require(digest(ROOT/'checks'/name)==expected,'Frozen dependency source changed: '+name)


def case_key(value):
    return (value['layer'],value['projection'],value['m'],value['amplitude'])


def case_plan():
    result=[]
    for layer in range(36):
        for projection,(k,n) in SHAPES.items():
            for m in ROWS:
                for amplitude in AMPLITUDES:
                    identity=dict(schema=SCHEMA,seed_base=20261120,layer=layer,projection=projection,m=m,amplitude=amplitude)
                    seed=int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:8],'big')%(2**63-1)
                    result.append(dict(index=len(result),layer=layer,projection=projection,m=m,k=k,n=n,
                        amplitude=amplitude,case_seed=seed,layout=LAYOUT,pad_m_multiple=0))
    require(len(result)==864 and len({r['case_seed'] for r in result})==864,'Unexpected seed-plan coverage/collision')
    return result


def bind_prior(directory,baseline_table):
    """Small saved-document audit; no model/checkpoint import or mmap here."""
    verify_sources()
    from navigation_latency_int8_attention_contract_20261120 import load_contract
    directory=Path(directory).resolve()
    proof=load_contract(directory,'awq',CHECKPOINT,verify_sources=True)
    require(proof['status']=='verified_micro_contract' and proof['allowed_projections']==['qkv','o'],
            'Original class-level conclusions changed')
    require(digest(baseline_table)==proof['baseline_table_sha256'],'New screen must retain exact native W4 table')
    rows=[r for r in proof['timing_rows'] if r['layout']==LAYOUT]
    require(len(rows)==8 and {(r['projection'],r['m']) for r in rows}=={(p,m) for p in SHAPES for m in (701,2815)},
            'Require eight NK shape-M records, each with hot/cold evidence')
    eligible={p:all(r['timing_screen_passed'] for r in rows if r['projection']==p) for p in SHAPES}
    negatives=[r for r in proof['rejected_cases'] if r['layout']==LAYOUT]
    negative_modules=sorted({f"{r['layer']}/{r['projection']}" for r in negatives})
    require(negative_modules==['35/down','35/gate_up'] and all('weight' in r['failed_gates'] and r['weight_relative_l2']>.02 for r in negatives),
            'Original layer35 negative evidence missing or changed')
    return dict(schema='navigation_expanded_int8_prior_shape_binding/v1',
        source_status='independently audited frozen recorded micro evidence; no actual checkpoint bytes opened here',
        model_weights_sha256=CHECKPOINT,producer_sources_sha256=proof['producer_sources_sha256'],
        auditor_sha256=proof['auditor_sha256'],evidence_sha256=proof['evidence_sha256'],
        prior_directory=directory.relative_to(ROOT).as_posix(),baseline_table_sha256=proof['baseline_table_sha256'],
        original_allowed_projections=['qkv','o'],original_class_failures_preserved=True,
        negative_modules=negative_modules,negative_cases=negatives,
        shape_timing_rows=rows,shape_timing_eligible=eligible,nk_shape_m_records=8,nk_hotcold_summaries=16,
        timing_scope='Prior layer0 shape evidence only, two randomized graph timing rounds; not144 module timings or eager/HTTP speed',
        input_scope='New case-local seed inputs are independent; original72 inputs are not claimed to be rerun')


def error_pass(value,limit):
    if not isinstance(value,dict): return False
    relative,maximum=value.get('relative_l2'),value.get('max_absolute_error')
    return (value.get('threshold')==limit and value.get('finite') is True and value.get('passed') is True and
        type(relative) in (int,float) and type(maximum) in (int,float) and
        math.isfinite(relative) and math.isfinite(maximum) and 0<=relative<=limit and maximum>=0)


def selected_indices(size,count):
    values=[0,size-1,size//2]
    for i in range(1,count*2):
        value=i*size//count
        if value not in values: values.append(value)
        if len(values)==count: break
    return sorted(values)


def implementation_pass(value,m,k,n):
    if not isinstance(value,dict): return False
    if not (value.get('passed') is True and set(value.get('flags',{}))==IMPL_FLAGS and all(v is True for v in value['flags'].values()) and
            value.get('rows')==selected_indices(m,4) and value.get('columns')==selected_indices(n,8) and value.get('k')==k and
            value.get('int32_safe') is True and value.get('theoretical_max_abs_integer')==k*127*127<2**31): return False
    for left,right in (('int64_expected','int32_actual'),('cpu_expected_half_bits','actual_half_bits')):
        values=value.get(left)
        if not (isinstance(values,list) and len(values)==4 and all(isinstance(r,list) and len(r)==8 for r in values) and values==value.get(right)):
            return False
    return (all(type(x) is int and abs(x)<=k*127*127 for r in value['int32_actual'] for x in r) and
            all(type(x) is int and 0<=x<=65535 for r in value['actual_half_bits'] for x in r))


def stability_pass(value,repeats):
    if not isinstance(value,dict) or value.get('passed') is not True or not valid_sha(value.get('reference_output_sha256')): return False
    conditions=value.get('conditions',[])
    if len(conditions)!=4: return False
    seen=set()
    for condition in conditions:
        ident=(condition.get('mode'),condition.get('cache_condition'))
        if ident in seen or ident[0] not in ('eager','cuda_graph_replay') or ident[1] not in ('hot','cold'): return False
        trials=condition.get('full_tensor_bytewise_equal',[])
        if not (condition.get('repeats')==repeats==len(trials) and condition.get('passed') is True and
                condition.get('eviction_buffer_bytes')==(96*1048576 if ident[1]=='cold' else 0) and
                all(isinstance(t,dict) and set(t)=={'qx','row_scale','integer','output'} and all(v is True for v in t.values()) for t in trials)):
            return False
        seen.add(ident)
    return len(seen)==4


def case_reasons(value,expected,repeats=4,require_status=False):
    if not isinstance(value,dict): return ['missing_or_null_case']
    reasons=[]
    if require_status and value.get('status')!='accepted_new_module_case': reasons.append('case_not_accepted')
    if any(value.get(k)!=v for k,v in expected.items()): reasons.append('case_identity_seed_or_layout_changed')
    if not valid_sha(value.get('input_sha256')): reasons.append('input_sha_missing')
    if value.get('status')=='candidate_error': reasons.append('candidate_execution_error')
    if value.get('semantic_boundaries_passed') is not True: reasons.append('semantic_boundaries_failed')
    if not implementation_pass(value.get('implementation'),expected['m'],expected['k'],expected['n']): reasons.append('independent_implementation_failed')
    if not error_pass(value.get('weight_error'),LIMIT): reasons.append('weight_relative_l2_gate_failed')
    approx=value.get('approximation') or {}
    if not error_pass(approx.get('activation'),LIMIT): reasons.append('activation_relative_l2_gate_failed')
    if not error_pass(approx.get('combined'),LIMIT): reasons.append('combined_relative_l2_gate_failed')
    if not error_pass(value.get('native_error'),.005) or not error_pass(value.get('dequant_fp16_gemm_error'),.005): reasons.append('same_weight_reference_failed')
    if not stability_pass(value.get('stability'),repeats): reasons.append('full_tensor_repeat_stability_failed')
    return reasons


def decisions(attempts,prior,repeats=4):
    """Return144 decisions even for a partial failed run; publication is separate."""
    require(prior['model_weights_sha256']==CHECKPOINT and prior['original_allowed_projections']==['qkv','o'] and
            prior['negative_modules']==['35/down','35/gate_up'],'Prior model/negative evidence identity differs')
    require(prior['producer_sources_sha256'].get('navigation_latency_expanded_int8_prefill_20261120.py')==SOURCES['navigation_latency_expanded_int8_prefill_20261120.py'],
            'Timing source differs from frozen arithmetic')
    require(prior['auditor_sha256']==SOURCES['navigation_latency_int8_attention_contract_20261120.py'],'Prior audit source differs')
    index={}
    for value in attempts:
        if value is None: continue
        key=case_key(value);require(key not in index,'Duplicate module screen case');index[key]=value
    plan=case_plan(); expected_keys={case_key(v) for v in plan}
    require(set(index)<=expected_keys,'Unregistered module screen case')
    result=[]
    for layer in range(36):
        for projection,(k,n) in SHAPES.items():
            module_id=f'{layer}/{projection}';wanted=[v for v in plan if v['layer']==layer and v['projection']==projection]
            failures=[dict(case_index=v['index'],m=v['m'],amplitude=v['amplitude'],reasons=why)
                      for v in wanted if (why:=case_reasons(index.get(case_key(v)),v,repeats,require_status=True))]
            reasons=[]
            if failures: reasons.append('not_all_six_new_cases_pass')
            if module_id in prior['negative_modules']: reasons.append('original_v1_weight_above_two_percent')
            if prior['shape_timing_eligible'].get(projection) is not True: reasons.append('prior_shape_full_path_timing_not_qualified')
            result.append(dict(module_id=module_id,layer=layer,projection=projection,k=k,n=n,
                expected_cases=6,observed_cases=sum(case_key(v) in index for v in wanted),case_failures=failures,
                route='original_w4' if reasons else 'expanded_int8_nk',eligible=not reasons,reasons=reasons,
                original_negative_evidence_bound=module_id in prior['negative_modules'],
                int8_weight_bytes=k*n,fp32_scale_bytes=4*n))
    return result


def publish_mask(attempts,prior,repeats=4):
    require(len(attempts)==864 and all(isinstance(v,dict) for v in attempts),'Only complete864-case ledgers may publish a fixed mask')
    records=decisions(attempts,prior,repeats)
    require(len(records)==144 and all(r['observed_cases']==6 for r in records),'Every module requires six actual cases')
    selected=[r['module_id'] for r in records if r['eligible']]
    payload=dict(schema='navigation_expanded_int8_fixed_module_mask/v1',model_weights_sha256=CHECKPOINT,
        arithmetic_source_sha256=SOURCES['navigation_latency_expanded_int8_prefill_20261120.py'],
        selected_modules=selected,original_w4_modules=[r['module_id'] for r in records if not r['eligible']],
        prefill_only=True,decode='original_w4',layout=LAYOUT,pad_m_multiple=0,per_input_fallback=False,
        validated_rows=list(ROWS),validated_random_amplitudes=list(AMPLITUDES),runtime_shape_and_activation_domain_approval=False,
        approximation_relative_l2_limit=LIMIT,case_plan_sha256=object_sha(case_plan()),
        new_case_ledger_sha256=object_sha(attempts),module_decisions_sha256=object_sha(records),
        module_contract_sha256=digest(__file__),
        prior_evidence_sha256=prior['evidence_sha256'],prior_auditor_sha256=prior['auditor_sha256'],
        whole_model_approval=False,actual_activation_quality='not_evaluated',navigation_quality='not_evaluated')
    require(not {'35/gate_up','35/down'}&set(selected),'Original negative modules cannot enter the mask')
    return dict(mask=payload,mask_sha256=object_sha(payload),module_decisions=records,
        accepted_modules=len(selected),retained_w4_modules=144-len(selected),
        selected_weight_and_scale_bytes=sum(r['int8_weight_bytes']+r['fp32_scale_bytes'] for r in records if r['eligible']),
        memory_scope='Only additional selected W8 weights/scales; original W4 decode weights, KV, activations/workspaces/peak memory excluded')
