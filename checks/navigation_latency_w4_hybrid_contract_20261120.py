"""Stdlib-only provenance gate for the experimental M1-down GEMV recipe.

This checks saved projection evidence before service construction. It neither
opens model weights nor proves complete-model outputs or navigation quality.
The runtime adapter and later fixed-input audit must establish those separately.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {'awq':'31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
          'gptq':'2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f'}
SHAPES = {'qkv':(2560,6144),'o':(4096,2560),'gate_up':(2560,19456),'down':(9728,2560)}
IMPLEMENTATION_SOURCES = {
    'navigation_latency_w4_hybrid_contract_20261120.py',
    'navigation_latency_w4_hybrid_adapter_20261120.py',
    'navigation_latency_w4_hybrid_module_map_20261120.py',
    'serve_navigation_latency_hybrid_20261120.py',
    'navigation_latency_w4_kernels_20261120.py',
    'navigation_latency_w4_gemv_20261120.py',
    'navigation_w4a16_repair_kernels.py',
    'navigation_w4a16_candidate_kernels.py',
}
WIDE_SOURCES = {'navigation_latency_w4_kernels_20261120.py','probe_navigation_latency_w4_20261120.py',
                'navigation_w4a16_repair_kernels.py','navigation_w4a16_candidate_kernels.py'}
GEMV_SOURCES = {'navigation_latency_w4_gemv_20261120.py','probe_navigation_latency_w4_gemv_20261120.py',
                'navigation_w4a16_repair_kernels.py','navigation_latency_w4_kernels_20261120.py',
                'navigation_w4a16_candidate_kernels.py'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    # All inputs here are small JSON/source files, never a checkpoint.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _valid_sha(value):
    return isinstance(value,str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def _project_path(value, relative_only=False):
    path = Path(value)
    _require(not relative_only or not path.is_absolute(), 'Recipe paths must be project-relative')
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    _require(path.is_relative_to(ROOT) and path.is_file(), 'Evidence path missing or outside project: '+str(path))
    return path


def _bound(path, expected):
    path = _project_path(path,relative_only=True)
    _require(_valid_sha(expected) and _sha(path) == expected, 'Bound evidence SHA mismatch: '+str(path))
    return path, _read(path)


def _numeric(value, with_finite=False):
    error = value.get('relative_l2')
    _require(value.get('passed') is True and isinstance(error,(int,float)) and
             math.isfinite(error) and 0 <= error < .005, 'Projection numerical check failed')
    if with_finite:
        maximum = value.get('max_absolute_error')
        _require(value.get('finite') is True and isinstance(maximum,(int,float)) and
                 math.isfinite(maximum) and maximum >= 0, 'Projection finite/max-error evidence failed')


def _tensor_record(value, k, n):
    expected = {'qweight':([k,n//8],'torch.int32'),
                'scales':([k//128,n],'torch.float16'),'qzeros':([k//128,n//8],'torch.int32')}
    _require(set(value) == set(expected), 'Packed tensor inventory differs')
    for name,(shape,dtype) in expected.items():
        item = value[name]
        _require(item['shape'] == shape and item['dtype'] == dtype and _valid_sha(item['sha256']),
                 'Packed tensor shape/dtype/SHA invalid: '+name)


def _sources(values, expected_names, verify, aggregate):
    _require(set(values) == expected_names, 'Source inventory differs')
    for name,sha in values.items():
        _require(Path(name).name == name and name.endswith('.py') and _valid_sha(sha), 'Invalid source identity')
        _require(name not in aggregate or aggregate[name] == sha, 'Sources disagree between recipe/probes: '+name)
        if verify:
            _require(_sha(ROOT/'checks'/name) == sha, 'Current implementation/probe source differs: '+name)
        aggregate[name] = sha


def _baseline(directory, protocol, source, algorithm, model_sha, expected_sha):
    table = _read(directory/'baseline_replayed_table.json')
    _require(protocol['baseline_table_sha256'] == expected_sha and
             _project_path(protocol['baseline_table']) == source, 'Probe baseline source differs')
    reference = _read(source); reference = reference.get('frozen_launch_table',reference)
    _require(table['algorithm'] == algorithm and table['model_weights_sha256'] == model_sha and
             table['frozen'] is True and table['actual_m_preserved'] is True and
             table['generic_m_specialization_disabled'] is True and
             table['runtime_autotuning_allowed'] is False and table['runtime_cold_compilation_allowed'] is False and
             table['selection_mode'] == 'replayed_frozen_table' and
             table['frozen_table_source']['sha256'] == expected_sha and
             all(table[key] == reference[key] for key in ('shapes','representatives','max_rows')),
             'Probe native frozen-table replay evidence differs')


def _confirm(blocks, repeats, cold, mode, cache_bytes):
    _require(len(blocks) == 2, 'Need two independent confirmation rounds')
    samples = []
    for block in blocks:
        values = block['samples_ms']
        _require(len(values) == repeats and all(isinstance(v,(int,float)) and math.isfinite(v) and v > 0 for v in values),
                 'Invalid confirmation samples/repeats')
        _require(block['mode'] == mode and block['cache_condition'] == ('eviction_before_each_sample' if cold else 'hot'),
                 'Confirmation execution/cache condition differs')
        _require(math.isclose(statistics.median(values),block['cuda_p50_ms'],rel_tol=1e-12), 'Block median differs')
        if cold:
            _require(block['eviction_buffer_bytes'] == cache_bytes and block['flush_gpu_ms_in_kernel_timing'] is False,
                     'Eviction entered kernel timing or buffer differs')
        samples.extend(values)
    return statistics.median(samples), len(samples)


def _eviction(result, protocol, count):
    value = result['cache_flush_evidence']
    _require(protocol['target_l2_mib'] == 48 and protocol['cold_cache_mib'] >= 64 and
             value['verified'] is True and value['flush_count'] == count and
             value['buffer_bytes'] == protocol['cold_cache_mib']*1048576 and
             value['full_buffer_checksum'] == value['expected_checksum'] == (count%256)*value['buffer_bytes'],
             'Cold-cache flush count/full-buffer checksum differs')


def load_hybrid_recipe(path, algorithm, model_sha, verify_sources=True):
    """Return ``dict(recipe=..., proof=...)``; never import GPU libraries.

    ``verify_sources=False`` is for static fixtures only and is explicit in the
    proof. A runtime installation must use the default strict source checking.
    ``pending`` full-model output approval is permitted for candidate testing,
    but this loader never promotes it to a successful full-model quality gate.
    """
    _require(algorithm in FROZEN and model_sha == FROZEN[algorithm], 'Hybrid checkpoint identity differs')
    path = _project_path(path); recipe = _read(path)
    _require(recipe['scope'] == 'm1_down_gemv_over_wide' and recipe['algorithm'] == algorithm and
             recipe['model_weights_sha256'] == model_sha, 'Hybrid recipe scope/model identity differs')
    _require(recipe['required_model_output_gate'] == 'pending', 'Projection recipe cannot issue model-output approval')
    sources = {}
    _sources(recipe['implementation_sources_sha256'],IMPLEMENTATION_SOURCES,verify_sources,sources)
    base_path,_ = _bound(recipe['baseline_table']['path'],recipe['baseline_table']['sha256'])
    wp,wide = _bound(recipe['wide']['result_path'],recipe['wide']['result_sha256'])
    wpp,wprotocol = _bound(recipe['wide']['protocol_path'],recipe['wide']['protocol_sha256'])
    wtp,table = _bound(recipe['wide']['table_path'],recipe['wide']['table_sha256'])
    gp,gemv = _bound(recipe['gemv']['result_path'],recipe['gemv']['result_sha256'])
    gpp,gprotocol = _bound(recipe['gemv']['protocol_path'],recipe['gemv']['protocol_sha256'])
    _require(wp.parent == wpp.parent == wtp.parent and gp.parent == gpp.parent, 'Probe evidence directories differ')
    for result,protocol,directory in ((wide,wprotocol,wp.parent),(gemv,gprotocol,gp.parent)):
        _require(not (directory/'failure.json').exists() and result['protocol'] == protocol,
                 'Probe failed or final protocol differs')
        _require(protocol['algorithm'] == algorithm and protocol['model_weights_sha256'] == model_sha and
                 protocol['validation_layers'] == [0,18,35] and protocol['relative_l2_limit'] == .005 and
                 protocol['seed'] == 20261120 and set(protocol['projections']) == set(SHAPES),
                 'Probe identity/numerical coverage protocol differs')
        _require(_read(directory/'numerical_validation.json') == result['numerical_validation'] and
                 _read(directory/'measurements.json') == result['measurements'], 'Probe final/sidecar disagreement')
        _baseline(directory,protocol,base_path,algorithm,model_sha,recipe['baseline_table']['sha256'])
    _sources(wprotocol['sources_sha256'],WIDE_SOURCES,verify_sources,sources)
    _sources(gprotocol['sources_sha256'],GEMV_SOURCES,verify_sources,sources)
    helper = gprotocol['native_awq_helper_source']
    _require(_valid_sha(helper['sha256']) and Path(helper['path']).name == 'awq_kernels.py', 'Native unpack helper identity invalid')
    if verify_sources:
        _require(_sha(helper['path']) == helper['sha256'], 'Native unpack helper source changed')
    _require(_project_path(gprotocol['wide_table']) == wtp and gprotocol['wide_table_sha256'] == recipe['wide']['table_sha256'],
             'GEMV probe did not use the bound wide table')
    _require(wide['status'] == 'passed' and wide['numerical_gate_passed'] is True and wide['launch_table'] == table and
             table['scope'] == 'exact_m_only' and table['algorithm'] == algorithm and table['model_weights_sha256'] == model_sha and
             table['numerical_gate_passed'] is True and table['validation_all_passed'] is True and
             table['validation_cases'] == 72 and table['validation_layers'] == [0,18,35] and
             table['validation_input_magnitudes'] == wprotocol['validation_input_magnitudes'] == [.1,1.] and
             wprotocol['layers'] == [0] and set(wprotocol['rows']) == {1,701,2815}, 'Wide numerical table gate differs')
    wide_configs = {(r['m'],r['k'],r['n']):r['config'] for r in table['shapes']}
    expected_shapes = {(m,k,n) for k,n in SHAPES.values() for m in (1,701,2815)}
    _require(len(table['shapes']) == len(wide_configs) == 12 and set(wide_configs) == expected_shapes,
             'Wide exact-M/layout configurations incomplete')
    tensors, seen = {}, set()
    expected = {(l,p,m,a) for l in (0,18,35) for p in SHAPES for m in (1,701,2815) for a in (.1,1.)}
    for value in wide['numerical_validation']:
        identity = (value['layer'],value['projection'],value['m'],value['input_magnitude'])
        _require(identity in expected and identity not in seen, 'Wide numerical cases missing/duplicated'); seen.add(identity)
        k,n = SHAPES[value['projection']]; _numeric(value); _tensor_record(value['tensors'],k,n)
        _require((value['k'],value['n']) == (k,n) and value['config'] == wide_configs[(value['m'],k,n)], 'Wide numerical config differs')
        key = (value['layer'],value['projection'])
        _require(key not in tensors or tensors[key] == value['tensors'], 'Wide tensor identity varies between numerical cases')
        tensors[key] = value['tensors']
    _require(seen == expected and len(wide['numerical_validation']) == 72, 'Wide needs all 72 numerical cases')
    wide_cold = 0; seen_shapes = set()
    for value in wide['measurements']:
        shape = (value['m'],value['k'],value['n'])
        _require(shape in wide_configs and shape not in seen_shapes and value['selected_config'] == wide_configs[shape],
                 'Wide measurement selected config differs'); seen_shapes.add(shape)
        _numeric(value['numerical']); _numeric(value['baseline_numerical'])
        for cold,prefix in ((False,'confirmation'),(True,'cold_confirmation')):
            for impl in ('baseline','candidate'):
                median,count = _confirm(value[prefix][impl],wprotocol['cold_repeats' if cold else 'confirm_repeats'],
                                        cold,'cuda_graph_replay' if shape[0] == 1 else 'eager',wprotocol['cold_cache_mib']*1048576)
                _require(math.isclose(median,value[prefix+'_p50_ms'][impl],rel_tol=1e-12), 'Wide confirmation median differs')
                if cold: wide_cold += count
    _require(seen_shapes == expected_shapes, 'Wide measurement coverage incomplete'); _eviction(wide,wprotocol,wide_cold)
    _require(gemv['status'] == 'candidate_for_full_model' and gemv['selected_all_projections'] is True and
             gprotocol['search_layer'] == 0 and gprotocol['m'] == 1 and
             gprotocol['input_magnitudes'] == [.1,1.] and gprotocol['cold_min_gain'] == .03,
             'GEMV candidate/protocol gate differs')
    selection = _read(gp.parent/'selection.json')
    _require(selection['scope'] == 'diagnostic_only_not_a_runtime_table' and
             selection['service_integration_authorized'] is False and selection['numerical_all_selected_passed'] is True and
             set(selection['selected']) == set(SHAPES), 'GEMV diagnostic selection differs')
    replacement = recipe['replacement']
    _require(set(replacement) == {'m','projection','k','n','config'} and replacement['m'] == 1 and
             replacement['projection'] == 'down' and (replacement['k'],replacement['n']) == SHAPES['down'] and
             replacement['config'] == selection['selected']['down'], 'Hybrid replacement must be exactly selected M1-down GEMV')
    seen = set(); expected = {(l,p,a,i) for l in (0,18,35) for p in SHAPES for a in (.1,1.) for i in ('native','wide','gemv')}
    for value in gemv['numerical_validation']:
        identity = (value['layer'],value['projection'],value['input_magnitude'],value['implementation'])
        _require(identity in expected and identity not in seen, 'GEMV numerical cases missing/duplicated'); seen.add(identity)
        k,n = SHAPES[value['projection']]; _numeric(value,with_finite=True); _tensor_record(value['tensors'],k,n)
        config = (None if value['implementation'] == 'native' else wide_configs[(1,k,n)] if value['implementation'] == 'wide'
                  else selection['selected'][value['projection']])
        _require(value['m'] == 1 and (value['k'],value['n']) == (k,n) and value['config'] == config and
                 value['tensors'] == tensors[(value['layer'],value['projection'])], 'GEMV config or cross-probe tensor identity differs')
    _require(seen == expected and len(gemv['numerical_validation']) == 72, 'GEMV needs all 72 numerical cases')
    promising, seen, gemv_cold = [], set(), 0
    for value in gemv['measurements']:
        label = value['projection']; _require(label in SHAPES and label not in seen, 'GEMV timing projection repeated'); seen.add(label)
        _require(value['status'] == 'confirmed' and value['m'] == 1 and (value['k'],value['n']) == SHAPES[label] and
                 value['selected_config'] == selection['selected'][label] and value['tensors'] == tensors[(0,label)],
                 'GEMV measured selection/tensors differ')
        _require(set(value['baseline_numerical']) == {'native','wide'}, 'GEMV baseline numerical inventory differs')
        for checks in value['baseline_numerical'].values():
            _require(len(checks) == 2, 'GEMV baseline must cover two amplitudes')
            for check in checks: _numeric(check,with_finite=True)
        reductions = {}
        for condition in ('hot','cold'):
            medians = {}; cold = condition == 'cold'
            _require(set(value['confirmation'][condition]) == {'native','wide','gemv'}, 'GEMV confirmation implementations differ')
            for impl in ('native','wide','gemv'):
                median,count = _confirm(value['confirmation'][condition][impl],
                    gprotocol['cold_repeats' if cold else 'confirm_repeats'],cold,'cuda_graph_replay',gprotocol['cold_cache_mib']*1048576)
                _require(math.isclose(median,value['confirmation_p50_ms'][condition][impl],rel_tol=1e-12), 'GEMV median differs')
                medians[impl] = median
                if cold: gemv_cold += count
            reductions[condition] = {impl:1-medians['gemv']/medians[impl] for impl in ('native','wide')}
            _require(all(math.isclose(v,value['reduction_fraction'][condition][impl],rel_tol=1e-12,abs_tol=1e-12)
                         for impl,v in reductions[condition].items()), 'GEMV reduction recomputation differs')
        passed = all(v > 0 for v in reductions['hot'].values()) and all(v >= .03 for v in reductions['cold'].values())
        _require(value['promising_for_whole_model'] is passed, 'GEMV predeclared advance rule differs')
        if passed: promising.append(label)
    _require(seen == set(SHAPES) and promising == gemv['promising_projections'] == selection['promising_projections'] == ['down'],
             'Hybrid recipe requires only down to pass the predeclared hot/cold rule')
    _eviction(gemv,gprotocol,gemv_cold)
    tails = _read(gp.parent/'boundary_checks.json')
    _require(tails == gemv['boundary_checks'], 'GEMV tail sidecar differs')
    selected_tails = set()
    for value in tails:
        _numeric(value,with_finite=True)
        if value['config'] == replacement['config']:
            identity = (value['k'],value['n'])
            _require(value['m'] == 1 and identity not in selected_tails, 'Duplicate selected GEMV tail'); selected_tails.add(identity)
    _require(selected_tails == {(128,8),(384,40),(9728,40)}, 'Selected down GEMV lacks all three synthetic-tail checks')
    attempts_path = gp.parent/'attempts.jsonl'
    attempts = [json.loads(line) for line in attempts_path.read_text(encoding='utf-8').splitlines()]
    _require(len(attempts) == gemv['attempts'] and dict(Counter(v.get('status','numerical_and_timing_passed') for v in attempts)) ==
             gemv['attempt_status_counts'], 'GEMV failed/accepted attempt ledger differs')
    for label,config in selection['selected'].items():
        matching = [v for v in attempts if v['projection'] == label and v['config'] == config and 'timing' in v]
        _require(len(matching) == 1 and all(v['passed'] for v in matching[0]['numerical']), 'Selected GEMV configuration lacks an accepted attempt')
    proof = dict(status='projection_evidence_verified' if verify_sources else 'projection_evidence_sources_not_verified',
        recipe_sha256=_sha(path),model_weights_sha256=model_sha,source_verification_enabled=verify_sources,
        sources_sha256=sources,native_unpack_helper=helper,
        wide_result_sha256=recipe['wide']['result_sha256'],wide_table_sha256=recipe['wide']['table_sha256'],
        gemv_result_sha256=recipe['gemv']['result_sha256'],baseline_table_sha256=recipe['baseline_table']['sha256'],
        wide_numerical_cases=72,gemv_numerical_cases=72,gemv_attempts=len(attempts),gemv_attempts_sha256=_sha(attempts_path),
        recorded_fused_tensor_sha256={f'{l}/{p}':v for (l,p),v in tensors.items()},
        replacement=replacement,promising_projections=promising,selected_tail_shapes=sorted(map(list,selected_tails)),
        cold_flush_counts=dict(wide=wide_cold,gemv=gemv_cold),
        checkpoint_bytes_read=False,independent_checkpoint_tensor_hash_verification=False,
        full_model_output_gate='pending',scope='Recorded projection/source evidence only. Runtime capture/routes and same-weight full-model tokens must pass a separate audit; this is not navigation-quality or HTTP-speed approval.')
    return dict(recipe=recipe,proof=proof)
