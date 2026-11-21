"""Small stdlib-only synthetic evidence tests; no model/probe/service execution."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import navigation_latency_w4_stable_audit_20261120 as audit
import summarize_navigation_latency_stable_20261120 as summary
from navigation_latency_w4_stable_selector_20261120 import native_config, shortlist


def fixture():
    h = 'a'*64
    numerical = dict(passed=True, finite=True, relative_l2=.001, max_absolute_error=.002)
    baseline = dict(shapes=[dict(k=k, n=n, split_k=max(1, k//4096),
        configs={'1': dict(BLOCK_SIZE_N=64, num_warps=4, num_stages=2, GROUP_SIZE_M=4)}) for k, n in audit.SHAPES.values()])
    configs = {p:native_config(baseline, p) for p in audit.SHAPES}
    table = dict(scope='exact_m_only', algorithm='awq', model_weights_sha256=audit.legacy.FROZEN['awq'],
        shapes=[dict(m=m, k=k, n=n, config=deepcopy(configs[p])) for p, (k, n) in audit.SHAPES.items() for m in (1, 701, 2815)],
        numerical_gate_passed=True, validation_all_passed=True, validation_cases=72, validation_layers=[0, 18, 35],
        validation_input_magnitudes=[.1, 1.], stability_gate_passed=True, stability_cases=24, decode_policy='native_split',
        native_bytewise_required=True, native_bytewise_all_m1_validation=True, prefill_source_wide_table_sha256=h,
        whole_model_output_gate='pending_new_four_cell_test')
    protocol = dict(algorithm='awq', model_weights_sha256=table['model_weights_sha256'], search_layer=0,
        relative_l2_limit=.005, seed=20261120, rows=[1, 701, 2815], projections=list(audit.SHAPES),
        validation_layers=[0, 18, 35], validation_input_magnitudes=[.1, 1.], stability_repeats=16,
        cold_cache_mib=96, target_l2_mib=48, require_native_bytewise=True, decode_policy='native_split',
        source_wide_table_sha256=h, candidate_shortlists={p:shortlist([], baseline, p, limit=1) for p in audit.SHAPES},
        repeats=2, cold_search_repeats=2, cold_repeats=2, confirm_repeats=2)
    source = deepcopy(table)
    tensors = {f'{l}/{p}': {name:dict(sha256=h, dtype='torch.float16' if name == 'scales' else 'torch.int32',
        shape=[k//128 if name != 'qweight' else k, n if name == 'scales' else n//8])
        for name in ('qweight', 'scales', 'qzeros')} for l in (0, 18, 35) for p, (k, n) in audit.SHAPES.items()}
    proof = dict(table_sha256=h, real_checkpoint_tensor_hashes=tensors)

    def suite(n):
        return dict(conditions=[dict(mode=mode, cache_condition=cache, repeats=16, output_shape=[1, n],
            output_dtype='torch.float16', output_sha256=[h]*16, unique_sha256_counts={h:16}, all_identical=True,
            eviction_buffer_bytes=96*1048576 if cache != 'hot' else 0)
            for mode in ('eager', 'cuda_graph_replay') for cache in ('hot', 'eviction_before_each_sample')],
            global_unique_sha256_counts={h:64}, all_identical=True, first_sha256=h)

    def timed(cold):
        return dict(samples_ms=[1., 1.], cuda_p50_ms=1., mode='cuda_graph_replay',
            cache_condition='eviction_before_each_sample' if cold else 'hot',
            eviction_buffer_bytes=96*1048576 if cold else 0, flush_gpu_ms_in_kernel_timing=False)

    numeric, stability, controls, attempts, measurements = [], [], [], [], []
    for l in (0, 18, 35):
        for p, (k, n) in audit.SHAPES.items():
            for m in (1, 701, 2815):
                for a in (.1, 1.):
                    numeric.append(dict(layer=l, projection=p, m=m, k=k, n=n, input_magnitude=a,
                        tensors=deepcopy(tensors[f'{l}/{p}']), config=deepcopy(configs[p]), **numerical))
                    if m == 1:
                        stability.append(dict(layer=l, projection=p, m=1, k=k, n=n, input_magnitude=a, input_sha256=h,
                            tensors=deepcopy(tensors[f'{l}/{p}']), config=deepcopy(configs[p]), native_numerical=deepcopy(numerical),
                            native_stability=suite(n), candidate_stability=suite(n), candidate_repeat_stable=True,
                            native_repeat_stable=True, exact_native_bytes=True))
    for p, (k, n) in audit.SHAPES.items():
        native, old, checks = [], [], []
        for a in (.1, 1.):
            diagnostic = dict(input_magnitude=a, input_sha256=h, numerical=deepcopy(numerical), stability=suite(n),
                scope='Unapproved original wide M1 diagnostic reference; never an acceptance baseline')
            controls.append(dict(projection=p, input_magnitude=a, input_sha256=h, native_numerical=deepcopy(numerical),
                native_stability=suite(n), old_wide_diagnostic=deepcopy(diagnostic)))
            native.append(dict(input_magnitude=a, input_sha256=h, numerical=deepcopy(numerical), stability=suite(n)))
            old.append(diagnostic)
            checks.append(dict(input_magnitude=a, input_sha256=h, numerical=deepcopy(numerical), stability=suite(n),
                candidate_repeat_stable=True, native_repeat_stable=True, exact_native_bytes=True))
        attempts.append(dict(projection=p, m=1, k=k, n=n, **protocol['candidate_shortlists'][p][0],
            checks=deepcopy(checks), rejection_reasons=[], status='accepted_for_selection',
            fresh_hot_timing=timed(False), fresh_cold_timing=timed(True)))
        measurements.append(dict(projection=p, m=1, k=k, n=n, status='confirmed', tensors=deepcopy(tensors[f'0/{p}']),
            selected_config=deepcopy(configs[p]), winner_checks=checks, native_controls=native, old_wide_diagnostic=old,
            confirmation={c:{mode:[timed(c == 'cold'), timed(c == 'cold')] for mode in ('native', 'candidate', 'old_wide_unapproved')}
                          for c in ('hot', 'cold')},
            confirmation_p50_ms={c:{mode:1. for mode in ('native', 'candidate', 'old_wide_unapproved')} for c in ('hot', 'cold')},
            confirmed_candidate_to_native_ratio=dict(hot=1., cold=1.)))
    sidecars = dict(numerical_validation=numeric, stability_validation=stability, measurements=measurements,
        search_controls=controls, attempts=attempts, source_attempts=[])
    flush = dict(timing=56, stability=2304); total=sum(flush.values())
    probe = dict(status='passed_projection_and_stability_gates', protocol=protocol, launch_table=table,
        numerical_validation=numeric, stability_validation=stability, measurements=measurements,
        attempts=4, attempt_status_counts={'accepted_for_selection':4},
        cache_flush_evidence=dict(verified=True, buffer_bytes=96*1048576, flush_counts_by_purpose=flush,
            flush_count=total, full_buffer_checksum=(total%256)*96*1048576, expected_checksum=(total%256)*96*1048576))
    return table, protocol, probe, sidecars, source, proof, baseline


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args=parser.parse_args(); passed=[]
    def run(name, mutate=None, rejection=False):
        f=fixture()
        if mutate: mutate(f)
        try: audit.stable_document_audit(*f)
        except (ValueError, KeyError):
            if not rejection: raise
        else:
            assert not rejection, name+' unexpectedly passed'
        passed.append(name)
    run('complete_72_numeric_24_suites_strict_native_bytewise')
    run('missing_native_bytewise_flag', lambda f:f[1].update(require_native_bytewise=False), True)
    run('incomplete_stability_repeats', lambda f:f[1].update(stability_repeats=8), True)
    run('duplicate_stability_case', lambda f:f[2]['stability_validation'].__setitem__(1, deepcopy(f[2]['stability_validation'][0])), True)
    run('numeric_nonfinite', lambda f:f[2]['numerical_validation'][0].update(finite=False), True)
    run('packed_tensor_sha_differs', lambda f:f[2]['numerical_validation'][0]['tensors']['qweight'].update(sha256='b'*64), True)
    run('prefill_config_changed', lambda f:f[0]['shapes'][1]['config'].update(split_k=2), True)
    run('m1_split4_rejected', lambda f:f[0]['shapes'][0]['config'].update(split_k=4), True)
    run('forged_global_hash_counter', lambda f:f[2]['stability_validation'][0]['candidate_stability']['global_unique_sha256_counts'].update({'b'*64:1}), True)
    run('forged_cold_p50', lambda f:f[3]['attempts'][0]['fresh_cold_timing'].update(cuda_p50_ms=.5), True)
    run('missing_flush', lambda f:f[2]['cache_flush_evidence']['flush_counts_by_purpose'].update(stability=2303), True)
    run('extra_flush_without_error', lambda f:f[2]['cache_flush_evidence']['flush_counts_by_purpose'].update(stability=2305), True)
    run('whole_model_claim_from_probe_rejected', lambda f:f[0].update(whole_model_output_gate='passed'), True)

    def change_suite(suite, values):
        for c in suite['conditions']:
            c['output_sha256']=list(values); c['unique_sha256_counts']=dict(Counter(values)); c['all_identical']=len(set(values))==1
        suite['global_unique_sha256_counts']=dict(Counter(values*4)); suite['all_identical']=len(set(values))==1; suite['first_sha256']=values[0]
    run('honest_unstable_candidate_rejected', lambda f:change_suite(f[2]['stability_validation'][0]['candidate_stability'], ['a'*64]*15+['b'*64]), True)
    run('repeat_stable_but_native_bytes_differ_rejected', lambda f:change_suite(f[2]['stability_validation'][0]['candidate_stability'], ['b'*64]*16), True)

    def session(sid, stable=False):
        return dict(id=sid, group='awq_w4a16', http_mode='nonstream', warmups=2,
            model_weights_sha256='a'*64, model_config_sha256='b'*64,
            cells={c:dict(n=10, output_sequences_identical=True) for c in audit.CELLS},
            wide_evidence=dict(numerical_probe=dict(validation_kind='strict_stable_w4')) if stable else None)
    native, candidate=session('native'),session('stable', True)
    rows=[dict(kind='same_model_kernel_implementation_change', baseline='native', candidate='stable', cell=c,
        comparable=True, exact_output_tokens_equal=True, both_repeats_identical=True, reasons=[],
        http_p50_ratio=.8, http_p50_reduction_percent=20., http_p95_ratio=.8) for c in audit.CELLS]
    candidate['wide_native_output_gate']=audit.stable_output_gate(candidate,[native,candidate],rows)
    assert candidate['wide_native_output_gate']['status']=='verified'
    summary.apply_wide_comparison_gates([native,candidate], rows)
    assert all(r['benefit_claim_allowed'] for r in rows);passed.append('four_cell_native_gate_success')
    changed_config=deepcopy(native);changed_config['model_config_sha256']='c'*64
    assert audit.stable_output_gate(candidate,[changed_config,candidate],rows)['status']!='verified'
    passed.append('same_weight_model_config_difference_blocks_gate')
    bad=deepcopy(candidate);bad['cells']['0_64']['output_sequences_identical']=False
    bad['wide_native_output_gate']=audit.stable_output_gate(bad,[native,bad],rows)
    bad_rows=deepcopy(rows)+[dict(kind='matched_quantization',baseline='fp16',candidate='stable',cell='0_18',
        comparable=True, reasons=[], http_p50_ratio=.7,http_p50_reduction_percent=30.,http_p95_ratio=.7)]
    summary.apply_wide_comparison_gates([native,bad],bad_rows)
    assert all(not r['comparable'] and not r['benefit_claim_allowed'] and r['http_p50_ratio'] is None for r in bad_rows)
    passed.append('failed64_gate_blocks_all_cells_and_fp16_claims')
    short=deepcopy(candidate);del short['cells']['1_64']
    assert audit.stable_output_gate(short,[native,short],rows)['status']!='verified';passed.append('partial_candidate_cannot_pass')
    one=deepcopy(native);two=deepcopy(native);one['id']='n1';two['id']='n2'
    halves=deepcopy(rows)
    for i,r in enumerate(halves):r['baseline']='n1' if i<2 else 'n2'
    assert audit.stable_output_gate(candidate,[one,two,candidate],halves)['status']!='verified'
    passed.append('cannot_merge_cells_from_different_native_sessions')
    failed_wide=deepcopy(native);failed_wide['id']='failed_old_wide';failed_wide['wide_evidence']={'numerical_probe':{}}
    for r in halves:r['baseline']='failed_old_wide'
    assert audit.stable_output_gate(candidate,[failed_wide,candidate],halves)['status']!='verified'
    passed.append('old_wide_cannot_substitute_native_reference')
    legacy_rows=deepcopy(rows)
    for r in legacy_rows:
        r['candidate']='failed_old_wide'
        if r['cell']=='0_64':
            r['comparable']=False;r['exact_output_tokens_equal']=False;r['both_repeats_identical']=False
    failed_wide['wide_native_output_gate']=summary.wide_output_gate(failed_wide,[native,failed_wide,candidate],legacy_rows)
    assert failed_wide['wide_native_output_gate']['status']!='verified'
    summary.apply_wide_comparison_gates([native,failed_wide,candidate],legacy_rows)
    assert all(not r['benefit_claim_allowed'] and r['http_p50_ratio'] is None for r in legacy_rows)
    passed.append('old_wide_failed64_gate_remains_failed_in_new_copy')
    for name,sha in audit.FROZEN_SOURCES.items():assert audit.digest(audit.ROOT/'checks'/name)==sha
    assert not any(name.split('.')[0] in ('torch','triton','lmdeploy') for name in sys.modules)
    passed.append('frozen_dependencies_unchanged_and_no_gpu_imports')
    result=dict(status='passed', tests=passed, count=len(passed), scope='Synthetic in-memory stdlib fixtures only; no actual probe/checkpoint/GPU proof')
    if args.output:
        assert not args.output.exists();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
