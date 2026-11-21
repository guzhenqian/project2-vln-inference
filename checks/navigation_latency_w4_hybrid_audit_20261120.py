"""Independent stdlib audit of the frozen hybrid runtime receipts.

Installation/projection evidence and full-model output approval are distinct.
No torch, LMDeploy, Triton, GPU access or checkpoint mmap is performed here.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

from navigation_latency_w4_hybrid_contract_20261120 import ROOT, SHAPES, load_hybrid_recipe


PROJECTIONS = {'qkv':'self_attn.qkv_proj','o':'self_attn.o_proj',
               'gate_up':'mlp.gate_up_proj','down':'mlp.down_proj'}
MODULES = {f'language_model.layers.{i}.{p}':label for i in range(36) for label,p in PROJECTIONS.items()}
DOWN = {name for name,label in MODULES.items() if label == 'down'}
OTHER = set(MODULES)-DOWN
FOUR_CELLS = {'0_18','0_64','1_18','1_64'}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    _require(path.is_relative_to(ROOT), 'Hybrid evidence outside project')
    return path


def _inventory(native):
    _require(native['verified'] is True and native['module_count'] == 144 and native['loaded_quantized_tensor_count'] == 756,
             'Native loaded-weight execution is not verified')
    rows = native['module_calls']
    _require(len(rows) == 144 and {v['name'] for v in rows} == set(MODULES),
             'Hybrid native inventory is not exactly 36 layers by four projections')


def audit_hybrid_install(service, launch, verified_sources):
    """Verify the real install plus its numerical/source/launch bindings."""
    service = _path(service); env = launch.get('candidate_environment',{})
    requested = env.get('PROJECT2_LATENCY_HYBRID_TABLE')
    paths = list(service.glob('hybrid_*.json'))
    if not requested:
        _require(not paths, 'Unexpected hybrid runtime receipts without opt-in')
        return None
    _require(not env.get('PROJECT2_LATENCY_W4_TABLE') and not env.get('PROJECT2_LATENCY_W4_PHASE'),
             'Hybrid recipe cannot be mixed with wide opt-in variables')
    _require(launch['recipe']['weight_bits'] == 4 and launch['recipe']['activation_bits'] == 16,
             'Hybrid must keep W4A16 precision')
    _require(Path(launch['engine']['custom_module_map']).name == 'navigation_latency_w4_hybrid_module_map_20261120.py',
             'Hybrid model map differs')
    recipe_path = _path(requested)
    _require(_path(launch['hybrid_recipe_path']) == recipe_path and launch['hybrid_recipe_sha256'] == _sha(recipe_path),
             'Launch hybrid recipe binding differs')
    binding = load_hybrid_recipe(recipe_path,launch['recipe']['method'],launch['model_weights_sha256'])
    recipe,proof = binding['recipe'],binding['proof']
    for name,sha in recipe['implementation_sources_sha256'].items():
        _require(name in verified_sources and verified_sources[name]['sha256'] == sha,
                 'Hybrid recipe source is absent/different in service launch: '+name)
    install_path = service/'hybrid_launch_install.json'; install = _read(install_path)
    _require(install['algorithm'] == recipe['algorithm'] and install['checkpoint_sha256'] == recipe['model_weights_sha256'] and
             install['scope'] == recipe['scope'] and _path(install['recipe_path']) == recipe_path and
             install['recipe_sha256'] == proof['recipe_sha256'] and install['recipe'] == recipe and install['proof'] == proof,
             'Hybrid installed recipe/projection proof differs')
    _require(install['actual_projection_count'] == 144 and install['gemv_decode_projection_count'] == 36 and
             install['wide_decode_projection_count'] == 108 and install['gemv_decode_modules'] == sorted(DOWN) and
             install['wide_decode_modules'] == sorted(OTHER), 'Hybrid decode module partition is not exact 36/108')
    _require(install['mandatory_m'] == [1,701,2815] and install['wide_prefill_m'] == [701,2815] and
             install['unknown_m_route'] == 'baseline_fallback' and
             install['cold_compile_guards'] == dict(native=True,wide=True,gemv=True,gemv_partial_reduce=True) and
             install['model_output_equivalence'] == 'pending_actual_full_model_gate',
             'Hybrid route/cold-compilation/output-gate contract differs')
    table = _read(_path(recipe['wide']['table_path']))
    frozen = install['wide_table']
    selected = lambda rows:{(v['m'],v['k'],v['n']):v['config'] for v in rows}
    _require(frozen['frozen'] is True and frozen['scope'] == 'exact_m_only' and
             frozen['algorithm'] == recipe['algorithm'] and frozen['model_weights_sha256'] == recipe['model_weights_sha256'] and
             len(frozen['shapes']) == 12 and selected(frozen['shapes']) == selected(table['shapes']),
             'Actually installed wide launch configurations differ from numerical table')
    execution = _read(service/'execution_audit.json')
    _require(execution['first_native_forward_verified'] is True, 'Hybrid native first forward missing')
    _inventory(execution['native_awq_receipt'])
    repair = _read(service/'repair_execution_audit.json'); fallback = repair['frozen_launch_table']
    _require(repair['native_forward'] is True and repair['activation_bits'] == 16 and repair['extension'] is False and
             fallback['frozen'] is True and fallback['actual_m_preserved'] is True and
             fallback['generic_m_specialization_disabled'] is True and fallback['runtime_autotuning_allowed'] is False and
             fallback['runtime_cold_compilation_allowed'] is False and fallback['selection_mode'] == 'replayed_frozen_table' and
             fallback['frozen_table_source']['sha256'] == recipe['baseline_table']['sha256'] and
             fallback['algorithm'] == recipe['algorithm'] and fallback['model_weights_sha256'] == recipe['model_weights_sha256'],
             'Hybrid unknown-M native fallback is not the exact frozen reference')
    baseline = _read(_path(recipe['baseline_table']['path'])); baseline = baseline.get('frozen_launch_table',baseline)
    _require(all(fallback[k] == baseline[k] for k in ('shapes','representatives','max_rows')), 'Hybrid fallback configurations differ')
    layouts = {(v['k'],v['n']):v for v in fallback['shapes']}
    warmup = install['actual_module_warmup']
    _require(len(warmup) == 144 and {v['module'] for v in warmup} == set(MODULES), 'Actual-module warmup inventory incomplete')
    for value in warmup:
        name = value['module']; k,n = SHAPES[MODULES[name]]
        _require((value['k'],value['n']) == (k,n) and value['wide_rows'] == [1,701,2815] and
                 value['decode_route'] == ('gemv_down' if name in DOWN else 'wide'), 'Actual-module preparation shape/route differs')
        native_layout = layouts[(k,n)]
        for observed,expected in (('qweight_stride','weight_stride'),('scales_stride','scales_stride'),('qzeros_stride','zeros_stride')):
            _require(value[observed] == native_layout[expected], 'Prepared packed tensor strides differ from native inventory')
        if name in DOWN:
            config = recipe['replacement']['config']; compiled = value['gemv_compiled']
            _require(value['gemv_config'] == config, 'Actual down GEMV compile configuration differs')
            for key,warps in (('gemv',config['num_warps']),('reduction',4)):
                observed = compiled[key]
                _require(isinstance(observed['n_regs'],int) and observed['n_regs'] > 0 and
                         isinstance(observed['n_spills'],int) and observed['n_spills'] >= 0 and observed['num_warps'] == warps,
                         'Actual GEMV/partial-reduction compiled metadata missing or launch differs')
            schedule = compiled['schedule']; split = config['split_k']
            _require(schedule['launches'] == 2 and schedule['fp32_partial_bytes'] == 4*split*n and
                     schedule['ctas'] == ((n+config['block_n']-1)//config['block_n'])*split,
                     'Hybrid split-partial allocation/launch evidence differs')
        else:
            _require('gemv_config' not in value and 'gemv_compiled' not in value, 'A non-down module was prepared as GEMV')
    capture_path = service/'startup_graph_capture.json'
    _require(_read(capture_path)['actual_capture_observed'] is True, 'Hybrid startup graph capture missing')
    return dict(status='verified_install_projection_evidence_pending_model_gate',recipe=recipe,projection_proof=proof,
        implementation_identity=dict(backend='hybrid_down_gemv_over_wide',recipe_sha256=proof['recipe_sha256'],
            wide_table_sha256=recipe['wide']['table_sha256'],gemv_result_sha256=recipe['gemv']['result_sha256']),
        install_receipt_sha256=_sha(install_path),startup_capture_sha256=_sha(capture_path),
        baseline_configuration_sha256=_object_sha({k:fallback[k] for k in ('shapes','representatives','max_rows')}),
        mandatory_projection_partition=dict(gemv_down=36,wide=108),full_model_output_gate='pending',
        scope='Only M1 down projections use GEMV. Full-model fixed outputs and request capture/replay remain separate required gates.')


def audit_hybrid_request(service, index, timing, contract, native_receipt):
    """Check actual Python prefill routes and separately observed graph replay."""
    hybrid = contract.get('hybrid_evidence')
    if not hybrid:
        return None
    service = _path(service); _inventory(native_receipt)
    path = service/f'hybrid_execution_{index:03d}.json'; value = _read(path)
    _require(value['index'] == timing['index'] == index and value['prefill_route_coverage_passed'] is True,
             'Hybrid request index/coverage differs')
    steps = timing['steps']; prefills = [v for v in steps if v['phase'] == 'prefill']
    _require(len(prefills) == 1 and steps[0] == prefills[0] and value['prefill_m'] == [prefills[0]['rows']],
             'Hybrid prefill/timing mapping differs')
    m = prefills[0]['rows']; decode = steps[1:]
    _require(decode and all(v['phase'] == 'decode' and v['rows'] == 1 and v['execution_mode'] == 'replay' for v in decode),
             'Hybrid decode is not actual post-capture M1 replay')
    _require(value['outside_graph_execution_modes'] == dict(Counter(v['execution_mode'] for v in steps)) and
             value['graph_replay_python_counters_increment'] is False,
             'Hybrid outside-graph execution evidence differs')
    expected = {name:('gemv_down' if m == 1 and name in DOWN else 'wide' if m in (1,701,2815) else 'baseline_fallback')
                for name in MODULES}
    calls = value['python_projection_calls']; observed = {}
    _require(len(calls) == 144, 'Hybrid must execute all 144 prefill projections once')
    for call in calls:
        name = call['module']
        _require(name in expected and name not in observed and call['calls'] == 1 and call['m'] == m and
                 call['route'] == expected[name], 'Hybrid prefill projection route/count/M differs')
        observed[name] = call['route']
    _require(observed == expected and value['prefill_route_projection_counts'] == dict(Counter(expected.values())),
             'Hybrid prefill route partition differs')
    for key,names in (('gemv_capture_projection_calls',DOWN),('wide_capture_projection_calls',OTHER)):
        capture = value[key]
        _require(set(capture) == names and all(isinstance(n,int) and not isinstance(n,bool) and n > 0 for n in capture.values()),
                 'Hybrid capture does not cover exactly 36 down GEMV/108 wide projections')
    _require(not value['baseline_capture_projection_calls'] and not value['unexpected_capture_projection_calls'] and
             value['expected_decode_route_projection_counts'] == dict(gemv_down=36,wide=108),
             'Hybrid captured an unexpected decode route')
    return dict(index=index,prefill_m=m,prefill_route_projection_counts=dict(Counter(expected.values())),
        actual_decode_forwards=len(decode),decode_projection_partition=dict(gemv_down=36,wide=108),
        capture_signature_sha256=_object_sha({k:value[k] for k in ('gemv_capture_projection_calls','wide_capture_projection_calls')}),
        receipt_sha256=_sha(path),scope='Python counts describe prefill only; graph capture routes and outside-graph decode replay are separate evidence.')


def audit_hybrid_sequence(service, contract, end_exclusive):
    """Reusable by later navigation/replay runners after their final flush."""
    if not contract.get('hybrid_evidence'):
        return None
    service = _path(service)
    _require(isinstance(end_exclusive,int) and end_exclusive >= 2, 'Completed request range required; exclude final tail')
    native = _read(service/'execution_audit.json')['native_awq_receipt']
    rows = [audit_hybrid_request(service,i,_read(service/f'timing_{i:03d}.json'),contract,native) for i in range(end_exclusive)]
    _require(len({v['capture_signature_sha256'] for v in rows}) == 1, 'Hybrid capture counters changed during measured sequence')
    return dict(requests=rows,covered_prefill_m=sorted({r['prefill_m'] for r in rows}),
        unknown_m_fallback_observed=any(r['prefill_route_projection_counts'] == {'baseline_fallback':144} for r in rows),
        actual_decode_forwards=sum(r['actual_decode_forwards'] for r in rows),full_model_output_gate='requires_fixed_binding')


def hybrid_output_gate(session, sessions, comparisons):
    """Require all four nonstream cells against the identical wide-all model."""
    if not session.get('hybrid_evidence'):
        return dict(status='not_applicable')
    if set(session['cells']) != FOUR_CELLS or session['http_mode'] != 'nonstream' or session['warmups'] < 2:
        return dict(status='unverified_incomplete_four_cell_protocol',missing_cells=sorted(FOUR_CELLS-set(session['cells'])),matching_comparisons=[])
    wide_sha = session['hybrid_evidence']['recipe']['wide']['table_sha256']
    references = {s['id']:s for s in sessions if s['group'] == session['group'] and s.get('wide_evidence') and
                  not s.get('hybrid_evidence') and s['wide_evidence']['phase'] == 'all' and
                  set(s['cells']) == FOUR_CELLS and
                  s['wide_evidence']['implementation_identity']['table_sha256'] == wide_sha and
                  s.get('wide_native_output_gate',{}).get('status') == 'verified'}
    matches = []
    for row in comparisons:
        ids = {row['baseline'],row['candidate']}
        reference = next((r for rid,r in references.items() if rid in ids),None)
        if session['id'] not in ids or reference is None:
            continue
        cell = row['cell']
        if (row['kind'] == 'same_model_kernel_implementation_change' and row['comparable'] and
            row['exact_output_tokens_equal'] is True and row['both_repeats_identical'] and cell in FOUR_CELLS and
            session['cells'][cell]['n'] >= 10 and reference['cells'][cell]['n'] >= 10 and
            reference['http_mode'] == 'nonstream' and reference['warmups'] >= 2):
            matches.append(row)
    covered = {row['cell'] for row in matches}; missing = sorted(FOUR_CELLS-covered)
    return dict(status='verified' if not missing else 'unverified_missing_matched_wide_all_cells',missing_cells=missing,
        matching_comparisons=matches,scope='Same model weights, actual inputs, common execution settings and wide table; only the 36 M1 down projections change to GEMV. Wide reference also needs its native output gate.')


def apply_hybrid_output_gates(sessions, comparisons):
    """Fail closed for every comparison involving an unapproved hybrid run."""
    blocked = {s['id']:s.get('hybrid_fixed_output_gate',{'status':'missing_gate','missing_cells':sorted(FOUR_CELLS)})
               for s in sessions if s.get('hybrid_evidence') and s.get('hybrid_fixed_output_gate',{}).get('status') != 'verified'}
    for row in comparisons:
        failures = {sid:blocked[sid] for sid in (row['baseline'],row['candidate']) if sid in blocked}
        if not failures:
            row.setdefault('benefit_claim_allowed',row['comparable'])
            continue
        row['comparable'] = row['benefit_claim_allowed'] = False
        row['hybrid_output_gate_failures'] = {sid:{k:g[k] for k in ('status','missing_cells')} for sid,g in failures.items()}
        for sid,gate in failures.items():
            reason = 'Hybrid four-cell output gate not verified: '+sid+'; '+gate['status']+'; missing='+','.join(gate['missing_cells'])
            if reason not in row['reasons']: row['reasons'].append(reason)
        row.setdefault('unvalidated_latency_observation',{k:row[k] for k in ('http_p50_ratio','http_p50_reduction_percent','http_p95_ratio')})
        for key in ('http_p50_ratio','http_p50_reduction_percent','http_p95_ratio'): row[key] = None
        row['ratio_scope'] = 'Hybrid full-model output gate is not verified; retained observations cannot support any speedup claim.'
    return comparisons
