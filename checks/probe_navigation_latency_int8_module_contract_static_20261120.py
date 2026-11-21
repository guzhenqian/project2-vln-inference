"""Small stdlib/AST negative fixtures for the independent per-module mask.

No torch/numpy/LMDeploy imports, weights, images, services or GPU operations.
Only a few synthetic modules are materialized; no full864 numerical ledger.
"""
import ast
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import navigation_latency_expanded_int8_module_contract_20261120 as contract
import probe_navigation_latency_expanded_int8_modules_20261120 as probe


def require(value,message):
    if not value:raise AssertionError(message)


def error(limit):
    return dict(relative_l2=.001,max_absolute_error=.001,finite=True,threshold=limit,passed=True)


def passing_case(expected):
    m,k,n=expected['m'],expected['k'],expected['n']
    arrays=[[0]*8 for _ in range(4)]
    implementation=dict(rows=contract.selected_indices(m,4),columns=contract.selected_indices(n,8),k=k,
        flags={key:True for key in contract.IMPL_FLAGS},passed=True,int32_safe=True,theoretical_max_abs_integer=k*127*127,
        int64_expected=arrays,int32_actual=arrays,cpu_expected_half_bits=arrays,actual_half_bits=arrays)
    conditions=[]
    for mode in ('eager','cuda_graph_replay'):
        for temperature in ('hot','cold'):
            conditions.append(dict(mode=mode,cache_condition=temperature,repeats=4,passed=True,
                eviction_buffer_bytes=96*1048576 if temperature=='cold' else 0,
                full_tensor_bytewise_equal=[dict(qx=True,row_scale=True,integer=True,output=True) for _ in range(4)]))
    return dict(expected,status='accepted_new_module_case',input_sha256='1'*64,semantic_boundaries_passed=True,
        implementation=implementation,weight_error=error(.02),approximation=dict(activation=error(.02),combined=error(.02)),
        native_error=error(.005),dequant_fp16_gemm_error=error(.005),
        stability=dict(passed=True,reference_output_sha256='2'*64,conditions=conditions))


def rejected(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected fail-closed rejection')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=contract.ROOT/'reports/navigation_latency_expanded_int8_modules_static_20261120.json')
    args=parser.parse_args()
    require(not any(name in sys.modules for name in ('torch','numpy','lmdeploy')),'Heavy package imported')
    checks=[];plan=contract.case_plan();require(len(plan)==864 and len({r['case_seed'] for r in plan})==864,'Case coverage changed')
    checks.append('864 distinct deterministic cases/seeds and144modules x6cases')
    helpers=probe.verify_copied_helpers();require(len(helpers)==13,'Frozen helper copy count differs')
    checks.append('13 copied numerical/stability/boundary helper ASTs equal frozen producer')
    design=json.loads((contract.ROOT/probe.DESIGN).read_text(encoding='utf-8'))
    require(contract.digest(contract.ROOT/probe.DESIGN)==probe.DESIGN_SHA and design['case_plan']==plan,'Prewritten seed protocol changed')
    checks.append('Prewritten protocol SHA and case plan match source constants')
    prior=dict(model_weights_sha256=contract.CHECKPOINT,original_allowed_projections=['qkv','o'],
        negative_modules=['35/down','35/gate_up'],producer_sources_sha256=dict(contract.SOURCES),
        auditor_sha256=contract.SOURCES['navigation_latency_int8_attention_contract_20261120.py'],
        shape_timing_eligible={p:True for p in contract.SHAPES},evidence_sha256={'result.json':'3'*64})
    cases=[passing_case(v) for v in plan if v['layer']==0 and v['projection']=='qkv']
    def decision(values,proof=prior,module='0/qkv'):
        return next(v for v in contract.decisions(values,proof) if v['module_id']==module)
    require(decision(cases)['eligible'],'All six valid cases unexpectedly rejected')
    checks.append('One whole module may enter mask only with six individually valid cases')
    for changed in (cases[:-1],cases[:-1]+[None]):require(not decision(changed)['eligible'],'Missing/null case entered mask')
    checks.append('Missing/null cases retain entire module W4')
    bad=copy.deepcopy(cases);bad[0]['implementation']['flags']['int32_accumulator_exact']=False
    require(not decision(bad)['eligible'],'CPU implementation failure entered mask')
    bad=copy.deepcopy(cases);bad[0]['weight_error']=dict(error(.02),relative_l2=.02001,passed=False)
    require(not decision(bad)['eligible'],'Weight error above2 percent entered mask')
    bad=copy.deepcopy(cases);bad[0]['stability']['conditions'][3]['full_tensor_bytewise_equal'][2]['output']=False
    require(not decision(bad)['eligible'],'One full-output repeat mismatch entered mask')
    checks.append('CPUexact,2percent approximation and full tensor repeat failures retain W4')
    old_negative=[passing_case(v) for v in plan if v['layer']==35 and v['projection']=='down']
    require(not decision(old_negative,module='35/down')['eligible'] and 'original_v1_weight_above_two_percent' in decision(old_negative,module='35/down')['reasons'],
            'Old failed module entered mask after synthetic new passing inputs')
    checks.append('Original layer35 failed module remains excluded even if all new fixture cases pass')
    untimed=copy.deepcopy(prior);untimed['shape_timing_eligible']['qkv']=False
    require(not decision(cases,untimed)['eligible'],'Unqualified timing shape entered mask')
    checks.append('Unqualified prior shape full-path timing retains W4')
    wrong=copy.deepcopy(prior);wrong['producer_sources_sha256']['navigation_latency_expanded_int8_prefill_20261120.py']='4'*64
    rejected(lambda:contract.decisions(cases,wrong))
    checks.append('Changed arithmetic source identity rejects mask evidence')
    rejected(lambda:contract.decisions(cases+[cases[0]],prior));rejected(lambda:contract.publish_mask(cases,prior))
    checks.append('Duplicate cases and incomplete864 publication rejected')
    for path in (Path(contract.__file__),Path(probe.__file__)):
        ast.parse(path.read_text(encoding='utf-8'))
    require(not any(name in sys.modules for name in ('torch','numpy','lmdeploy')),'Heavy package unexpectedly imported')
    report=dict(status='passed_stdlib_static_fixtures',checks=checks,copied_helper_ast_sha256=helpers,
        heavy_imports=False,gpu_executed=False,checkpoint_or_images_read=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(contract.__file__),Path(probe.__file__))},
        scope='Static control/gate fixtures only. No actual module arithmetic, mask eligibility or performance has been tested.')
    output=args.output
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
