"""Strict independent evidence audit for the new split-restricted W4 table.

Only stdlib and frozen CPU helpers are imported. audit_stable_probe reads packed
checkpoint bytes through the original verifier: run it outside GPU timings.
The document functions accept in-memory fixtures and never access checkpoints.
Projection approval never replaces the separate four-cell model-output gate.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

import summarize_navigation_latency_20261120 as legacy
from navigation_latency_w4_stable_selector_20261120 import admissible, shortlist

ROOT = Path(__file__).resolve().parents[1]
require, read, digest = legacy.require, legacy.read, legacy.digest
SHAPES = legacy.PROJECTION_SHAPES
CELLS = {'0_18', '0_64', '1_18', '1_64'}
FROZEN_SOURCES = {
    'summarize_navigation_latency_20261120.py': '0bd36efc0d71831b2caaac20c934fb41fa16c2e443debd2704b484567da29509',
    'probe_navigation_latency_w4_stable_20261120.py': '2a9b187be2fc7800ede53e32f50021cff5aff00ef2663ab8d0302751ae3d2f1f',
    'navigation_latency_w4_stable_selector_20261120.py': '06835b749a41176a21b56b70af6b167cb03abea9d0d715c2495db9ba2d71aa9a',
}
STABLE_SOURCES = set(FROZEN_SOURCES) - {'summarize_navigation_latency_20261120.py'} | {
    'navigation_latency_w4_kernels_20261120.py', 'navigation_w4a16_repair_kernels.py',
    'navigation_w4a16_candidate_kernels.py'}


def sha_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def numerical_audit(value):
    require(value['passed'] is True and value['finite'] is True, 'Stable numerical finite/pass gate failed')
    for name in ('relative_l2', 'max_absolute_error'):
        require(isinstance(value[name], (float, int)) and math.isfinite(value[name]) and value[name] >= 0,
                'Invalid stable numerical error: '+name)
    require(value['relative_l2'] < .005, 'Stable numerical tolerance exceeded')


def suite_audit(suite, n, repeats=16, buffer_bytes=96*1048576, require_stable=True):
    """Recompute all hashes/counts, including across the four execution modes."""
    conditions = suite['conditions']; seen = set(); all_hashes = []; cold = 0
    expected = {(mode, cache) for mode in ('eager', 'cuda_graph_replay')
                for cache in ('hot', 'eviction_before_each_sample')}
    require(len(conditions) == 4 and repeats == 16, 'Require four conditions and sixteen complete byte hashes')
    for value in conditions:
        key = (value['mode'], value['cache_condition'])
        require(key in expected and key not in seen, 'Duplicate/missing stability execution condition')
        seen.add(key)
        hashes = value['output_sha256']
        require(value['repeats'] == len(hashes) == repeats and all(sha_valid(v) for v in hashes),
                'Incomplete or malformed repeated output byte SHA')
        require(value['output_shape'] == [1, n] and value['output_dtype'] == 'torch.float16',
                'Stability output shape/dtype differs')
        require(value['unique_sha256_counts'] == dict(Counter(hashes)) and
                value['all_identical'] is (len(set(hashes)) == 1), 'Per-condition stability counter differs')
        is_cold = key[1] != 'hot'
        require(value['eviction_buffer_bytes'] == (buffer_bytes if is_cold else 0), 'Stability eviction buffer differs')
        cold += repeats if is_cold else 0
        all_hashes.extend(hashes)
    stable = len(set(all_hashes)) == 1
    require(seen == expected and suite['global_unique_sha256_counts'] == dict(Counter(all_hashes)) and
            suite['all_identical'] is stable and suite['first_sha256'] == all_hashes[0],
            'Global stability hash/counter differs')
    require(not require_stable or stable, 'Output bytes differ across repeat/eager/replay/hot/cold')
    return dict(stable=stable, first_sha256=all_hashes[0], cold_flushes=cold,
                output_hash_count=len(all_hashes))


def relation_audit(record, candidate, native, strict=True):
    exact = candidate['stable'] and native['stable'] and candidate['first_sha256'] == native['first_sha256']
    require(record['candidate_repeat_stable'] is candidate['stable'] and
            record['native_repeat_stable'] is native['stable'] and record['exact_native_bytes'] is exact,
            'Native/candidate bytewise relation flag differs')
    require(candidate['stable'] and native['stable'] and (not strict or exact),
            'Native/candidate stability or native-bytewise gate failed')
    return exact


def timing_audit(block, count, cold, buffer_bytes):
    samples = block['samples_ms']
    require(len(samples) == count and all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in samples),
            'Invalid stable timing samples/repeat count')
    require(block['mode'] == 'cuda_graph_replay' and block['cache_condition'] ==
            ('eviction_before_each_sample' if cold else 'hot'), 'Stable timing mode/cache condition differs')
    require(block['eviction_buffer_bytes'] == (buffer_bytes if cold else 0) and
            block['flush_gpu_ms_in_kernel_timing'] is False, 'Eviction entered timing or buffer differs')
    require(math.isclose(block['cuda_p50_ms'], statistics.median(samples), rel_tol=1e-12), 'Timing median differs')
    return len(samples) if cold else 0


def stable_document_audit(table, protocol, probe, sidecars, source_table, source_proof, baseline):
    """Validate saved small JSONs, given independently verified source tensors.

    This function alone returns document evidence, never checkpoint approval.
    The public file auditor supplies source_proof from the frozen mmap verifier.
    """
    require(probe['status'] == 'passed_projection_and_stability_gates' and probe['protocol'] == protocol,
            'Stable probe incomplete or protocol differs')
    require(probe['launch_table'] == table and table['scope'] == 'exact_m_only', 'Stable table/result differs')
    method = table['algorithm']
    require(method in ('awq', 'gptq') and protocol['algorithm'] == source_table['algorithm'] == method and
            table['model_weights_sha256'] == protocol['model_weights_sha256'] == source_table['model_weights_sha256'] == legacy.FROZEN[method],
            'Stable/source/native model identity differs')
    require(protocol['search_layer'] == 0 and protocol['relative_l2_limit'] == .005 and protocol['seed'] == 20261120 and
            protocol['rows'] == [1, 701, 2815] and set(protocol['projections']) == set(SHAPES), 'Stable probe scope differs')
    require(protocol['validation_layers'] == table['validation_layers'] == [0, 18, 35] and
            protocol['validation_input_magnitudes'] == table['validation_input_magnitudes'] == [.1, 1.], 'Stable numerical scope differs')
    require(protocol['stability_repeats'] == 16 and protocol['cold_cache_mib'] == 96 and protocol['target_l2_mib'] == 48,
            'Stable strict protocol requires 16 repeats and 96 MiB eviction')
    require(protocol['require_native_bytewise'] is True and table['native_bytewise_required'] is True and
            table['native_bytewise_all_m1_validation'] is True, 'Strict native-bytewise gate was not requested/passed')
    require(table['decode_policy'] == protocol['decode_policy'] and table['decode_policy'] in ('native_split', 'split_le_2'),
            'Decode selection policy differs')
    require(table['whole_model_output_gate'] == 'pending_new_four_cell_test', 'Projection table must not claim model approval')
    require(table['prefill_source_wide_table_sha256'] == protocol['source_wide_table_sha256'] == source_proof['table_sha256'],
            'Prefill source table SHA differs')
    for key in ('measurements', 'numerical_validation', 'stability_validation'):
        require(sidecars[key] == probe[key], 'Stable sidecar/result differs: '+key)
    shapes = {(v['m'], v['k'], v['n']): v['config'] for v in table['shapes']}
    old = {(v['m'], v['k'], v['n']): v['config'] for v in source_table['shapes']}
    expected_shapes = {(m, k, n) for k, n in SHAPES.values() for m in (1, 701, 2815)}
    require(len(table['shapes']) == len(shapes) == 12 and set(shapes) == expected_shapes, 'Stable shape coverage incomplete')
    for projection, (k, n) in SHAPES.items():
        require(admissible(shapes[(1, k, n)], projection, table['decode_policy']), 'M1 violates restricted split/config policy')
        for m in (701, 2815):
            require(shapes[(m, k, n)] == old[(m, k, n)], 'Unapproved prefill configuration change')
    require(table['numerical_gate_passed'] is True and table['validation_all_passed'] is True and
            table['validation_cases'] == len(probe['numerical_validation']) == 72, 'Require all 72 numerical cases')
    numeric = {}
    expected = {(l, p, m, a) for l in (0, 18, 35) for p in SHAPES for m in (1, 701, 2815) for a in (.1, 1.)}
    for value in probe['numerical_validation']:
        key = (value['layer'], value['projection'], value['m'], value['input_magnitude'])
        require(key in expected and key not in numeric, 'Missing/duplicate stable numerical case')
        k, n = SHAPES[value['projection']]
        require((value['k'], value['n']) == (k, n) and value['config'] == shapes[(value['m'], k, n)], 'Numerical shape/config differs')
        numerical_audit(value)
        require(value['tensors'] == source_proof['real_checkpoint_tensor_hashes'][f"{value['layer']}/{value['projection']}"],
                'Stable numerical tensor fusion differs from independently verified checkpoint')
        numeric[key] = value
    require(set(numeric) == expected, 'Stable numerical coverage differs')
    buffer_bytes = 96*1048576
    require(table['stability_gate_passed'] is True and table['stability_cases'] == len(probe['stability_validation']) == 24,
            'Require all 24 stability cases')
    stable = {}; flushes = Counter(timing=0, stability=0)
    expected_stable = {(l, p, a) for l in (0, 18, 35) for p in SHAPES for a in (.1, 1.)}
    for value in probe['stability_validation']:
        key = (value['layer'], value['projection'], value['input_magnitude'])
        require(key in expected_stable and key not in stable and value['m'] == 1 and sha_valid(value['input_sha256']),
                'Missing/duplicate stability case or input SHA')
        reference = numeric[(key[0], key[1], 1, key[2])]
        require(all(value[name] == reference[name] for name in ('k', 'n', 'config', 'tensors')), 'Stability/numerical identities differ')
        numerical_audit(value['native_numerical'])
        native = suite_audit(value['native_stability'], value['n'])
        candidate = suite_audit(value['candidate_stability'], value['n'])
        relation_audit(value, candidate, native)
        flushes['stability'] += native['cold_flushes'] + candidate['cold_flushes']
        stable[key] = value
    require(set(stable) == expected_stable, 'Incomplete stability layer/projection/amplitude coverage')

    controls = {}; native_controls = {}; diagnostic = {}
    for value in sidecars['search_controls']:
        key = (value['projection'], value['input_magnitude']); n = SHAPES[key[0]][1]
        require(key in {(p, a) for p in SHAPES for a in (.1, 1.)} and key not in controls and sha_valid(value['input_sha256']),
                'Search control coverage/input differs')
        numerical_audit(value['native_numerical'])
        native = suite_audit(value['native_stability'], n)
        old_value = value['old_wide_diagnostic']
        require(old_value['input_magnitude'] == key[1] and old_value['input_sha256'] == value['input_sha256'] and
                'Unapproved' in old_value['scope'], 'Old wide diagnostic was confused with acceptance baseline')
        # Old failed wide may be unstable or numerically rejected; never approve it here.
        old_suite = suite_audit(old_value['stability'], n, require_stable=False)
        flushes['stability'] += native['cold_flushes'] + old_suite['cold_flushes']
        controls[key] = value; native_controls[key] = native; diagnostic[key] = old_value
    require(len(controls) == 8, 'Require eight native/old diagnostic search controls')

    attempts = sidecars['attempts']; require(len(attempts) == probe['attempts'], 'Attempt ledger count differs')
    require(dict(Counter(v['status'] for v in attempts)) == probe['attempt_status_counts'], 'Attempt status count differs')
    pools = protocol['candidate_shortlists']; require(set(pools) == set(SHAPES), 'Shortlist projection inventory differs')
    for p in SHAPES:
        require(pools[p] == shortlist(sidecars['source_attempts'], baseline, p, protocol['decode_policy'], len(pools[p])),
                'Shortlist not reproducible from original launch plus historical search hints')
    accepted = {p: [] for p in SHAPES}; per_projection = Counter(); errors = []
    for attempt in attempts:
        p = attempt['projection']; require(p in SHAPES, 'Unknown attempted projection')
        k, n = SHAPES[p]; index = per_projection[p]; per_projection[p] += 1
        require(index < len(pools[p]) and all(attempt[key] == pools[p][index][key] for key in pools[p][index]) and
                (attempt['m'], attempt['k'], attempt['n']) == (1, k, n), 'Attempt differs from ordered shortlist')
        require(attempt['status'] in ('candidate_error', 'rejected', 'accepted_for_selection'), 'Unknown attempt status')
        if attempt['status'] == 'candidate_error':
            require(isinstance(attempt.get('error'), str), 'Candidate error detail absent')
            errors.append(dict(projection=p, index=index, error=attempt['error']))
        tests = attempt.get('checks', []); seen_amplitudes = set(); reasons = []
        if len(tests) != 2 or any(not v['numerical']['passed'] for v in tests): reasons.append('numerical_failed')
        for value in tests:
            a = value['input_magnitude']; key = (p, a)
            require(a in (.1, 1.) and a not in seen_amplitudes and value['input_sha256'] == controls[key]['input_sha256'],
                    'Attempt input/control differs')
            seen_amplitudes.add(a)
            if value['numerical']['passed']: numerical_audit(value['numerical'])
            if 'stability' in value:
                suite = suite_audit(value['stability'], n, require_stable=False)
                base = native_controls[key]
                exact = suite['stable'] and base['stable'] and suite['first_sha256'] == base['first_sha256']
                require(value['candidate_repeat_stable'] is suite['stable'] and value['native_repeat_stable'] is base['stable'] and
                        value['exact_native_bytes'] is exact, 'Attempt byte relation differs')
                if not suite['stable'] and 'candidate_repeat_bytes_differ' not in reasons: reasons.append('candidate_repeat_bytes_differ')
                if not exact and 'candidate_bytes_differ_from_native' not in reasons: reasons.append('candidate_bytes_differ_from_native')
                flushes['stability'] += suite['cold_flushes']
            elif value['numerical']['passed']:
                require(attempt['status'] == 'candidate_error', 'Numerically accepted attempt lacks stability evidence')
        if attempt['status'] != 'candidate_error':
            require(set(attempt['rejection_reasons']) == set(reasons), 'Attempt rejection reasoning differs')
            require((attempt['status'] == 'accepted_for_selection') == (not reasons), 'Attempt acceptance gate differs')
        if 'fresh_hot_timing' in attempt:
            timing_audit(attempt['fresh_hot_timing'], protocol['repeats'], False, buffer_bytes)
        if 'fresh_cold_timing' in attempt:
            flushes['timing'] += timing_audit(attempt['fresh_cold_timing'], protocol['cold_search_repeats'], True, buffer_bytes)
        if attempt['status'] == 'accepted_for_selection':
            require('fresh_hot_timing' in attempt and 'fresh_cold_timing' in attempt and len(tests) == 2 and
                    all('stability' in v for v in tests), 'Accepted candidate timing/gates incomplete')
            accepted[p].append(attempt)
    require(all(accepted.values()), 'A projection lacks an eligible measured candidate')
    measurements = {}; confirmations = {}
    for value in probe['measurements']:
        p = value['projection']; require(p in SHAPES and p not in measurements, 'Duplicate/missing confirmation projection')
        k, n = SHAPES[p]
        winner = min(accepted[p], key=lambda v: v['fresh_cold_timing']['cuda_p50_ms'])
        require(value['status'] == 'confirmed' and (value['m'], value['k'], value['n']) == (1, k, n) and
                value['selected_config'] == winner['config'] == shapes[(1, k, n)] and value['winner_checks'] == winner['checks'],
                'Winner not lowest freshly measured eligible cold median')
        require(value['tensors'] == source_proof['real_checkpoint_tensor_hashes'][f'0/{p}'], 'Winner tensors differ')
        expected_controls = [dict(input_magnitude=a, input_sha256=controls[(p, a)]['input_sha256'],
            numerical=controls[(p, a)]['native_numerical'], stability=controls[(p, a)]['native_stability']) for a in (.1, 1.)]
        require(value['native_controls'] == expected_controls and value['old_wide_diagnostic'] == [diagnostic[(p, a)] for a in (.1, 1.)],
                'Winner controls/diagnostic differ from search ledger')
        require(set(value['confirmation']) == {'hot', 'cold'}, 'Confirmation cache modes incomplete')
        medians = {}
        for condition, modes in value['confirmation'].items():
            require(set(modes) == {'native', 'candidate', 'old_wide_unapproved'}, 'Confirmation baselines differ')
            medians[condition] = {}
            for mode, blocks in modes.items():
                require(len(blocks) == 2, 'Require two independent confirmation rounds')
                samples = []
                for block in blocks:
                    flushes['timing'] += timing_audit(block, protocol['cold_repeats'] if condition == 'cold' else protocol['confirm_repeats'],
                                                    condition == 'cold', buffer_bytes)
                    samples.extend(block['samples_ms'])
                medians[condition][mode] = statistics.median(samples)
                require(math.isclose(medians[condition][mode], value['confirmation_p50_ms'][condition][mode], rel_tol=1e-12),
                        'Confirmation median recomputation differs')
            ratio = medians[condition]['candidate']/medians[condition]['native']
            require(math.isclose(ratio, value['confirmed_candidate_to_native_ratio'][condition], rel_tol=1e-12), 'Confirmation ratio differs')
        measurements[p] = value; confirmations[p] = medians
    require(set(measurements) == set(SHAPES), 'Require four confirmed projections')
    eviction = probe['cache_flush_evidence']; recorded = eviction['flush_counts_by_purpose']
    require(set(recorded) == {'timing', 'stability'} and all(type(v) is int and v >= flushes[k] for k, v in recorded.items()),
            'Flush ledger has missing purposes or fewer evictions than complete saved evidence')
    unassigned = {k: recorded[k]-flushes[k] for k in recorded}
    require(errors or not any(unassigned.values()), 'Unexplained cache flushes without a retained candidate error')
    total = sum(recorded.values())
    require(eviction['verified'] is True and eviction['buffer_bytes'] == buffer_bytes and eviction['flush_count'] == total and
            eviction['full_buffer_checksum'] == eviction['expected_checksum'] == (total % 256)*buffer_bytes,
            'Full eviction-buffer checksum/flush counter differs')
    return dict(status='document_gates_verified_checkpoint_binding_required', numerical_cases=72, stability_cases=24,
        stability_conditions_per_case_per_implementation=4, repeats_per_condition=16,
        repeated_hashes_in_validation=24*2*4*16, native_bytewise_required=True, native_bytewise_all_validation=True,
        max_relative_l2=max(v['relative_l2'] for v in numeric.values()), decode_policy=table['decode_policy'],
        confirmation_p50_ms=confirmations, attempt_status_counts=probe['attempt_status_counts'], candidate_errors=errors,
        cache_flush_ledger=dict(recorded=recorded, attributable_complete_evidence=dict(flushes), unassigned_after_errors=unassigned,
            exact_attribution=not any(unassigned.values()), full_buffer_checksum_verified=True),
        prior_wide_full_model_approval_inherited=False, whole_model_output_gate='pending_new_four_cell_test')


def audit_stable_probe(table_path, launch, verified_sources, service):
    """Verify actual packed tensors plus all strict stable evidence (idle only)."""
    for name, expected in FROZEN_SOURCES.items():
        require(digest(ROOT/'checks'/name) == expected, 'Frozen stable audit dependency changed: '+name)
    table_path = legacy.recorded_path(table_path); directory = table_path.parent
    require(not (directory/'failure.json').exists(), 'Stable probe retained failure evidence')
    table, protocol, probe = read(table_path), read(directory/'protocol.json'), read(directory/'result.json')
    require(table['algorithm'] == launch['recipe']['method'] and table['model_weights_sha256'] == launch['model_weights_sha256'],
            'Stable service/probe model identity differs')
    require(set(protocol['sources_sha256']) == STABLE_SOURCES, 'Stable probe source inventory differs')
    for name, expected in protocol['sources_sha256'].items():
        snapshot = Path(service).parent.parent/'source_snapshot'/name
        path = snapshot if snapshot.is_file() and digest(snapshot) == expected else ROOT/'checks'/name
        require(path.is_file() and digest(path) == expected, 'Stable probe source SHA cannot be verified: '+name)
        if name in FROZEN_SOURCES: require(expected == FROZEN_SOURCES[name], 'Unreviewed stable probe/selector source')
        if name in verified_sources: require(verified_sources[name]['sha256'] == expected, 'Stable runtime/probe kernel differs')
    from importlib.metadata import distribution
    helper = Path(distribution('lmdeploy').locate_file('lmdeploy/pytorch/kernels/cuda/awq_kernels.py')).resolve()
    require(Path(protocol['native_awq_source']['path']).resolve() == helper and digest(helper) == protocol['native_awq_source']['sha256'],
            'Underlying native W4 JIT source differs')
    source_table_path = legacy.recorded_path(protocol['source_wide_table']); source_dir = source_table_path.parent
    for name, key in (('launch_table.json', 'source_wide_table_sha256'), ('result.json', 'source_wide_result_sha256'),
                      ('protocol.json', 'source_wide_protocol_sha256'), ('attempts.jsonl', 'source_wide_attempts_sha256')):
        path = source_table_path if name == 'launch_table.json' else source_dir/name
        require(digest(path) == protocol[key], 'Inherited source-wide evidence SHA differs: '+name)
    source_protocol = read(source_dir/'protocol.json')
    require(source_protocol['sources_sha256'] == protocol['source_wide_sources_sha256'], 'Inherited original probe sources differ')
    baseline_path = legacy.recorded_path(protocol['baseline_table'])
    require(digest(baseline_path) == protocol['baseline_table_sha256'] == source_protocol['baseline_table_sha256'],
            'Stable/source baseline table SHA differs')
    baseline = read(baseline_path); baseline = baseline.get('frozen_launch_table', baseline)
    replayed = read(directory/'baseline_replayed_table.json')
    require(replayed['frozen'] and replayed['actual_m_preserved'] and replayed['generic_m_specialization_disabled'] and
            not replayed['runtime_autotuning_allowed'] and not replayed['runtime_cold_compilation_allowed'] and
            replayed['selection_mode'] == 'replayed_frozen_table' and replayed['algorithm'] == table['algorithm'] and
            replayed['model_weights_sha256'] == table['model_weights_sha256'] and
            replayed['frozen_table_source']['sha256'] == protocol['baseline_table_sha256'] and
            replayed['shapes'] == baseline['shapes'] and replayed['representatives'] == baseline['representatives'],
            'Stable actual frozen-native replay differs')
    # This verifies real tensor bytes once per cached original probe. Its result
    # approves numerical provenance only, never the old wide model-token gate.
    source_proof = legacy.wide_probe_audit(source_table_path, launch, verified_sources, Path(service))
    sidecars = {name: read(directory/(name+'.json')) for name in
                ('measurements', 'numerical_validation', 'stability_validation', 'search_controls')}
    sidecars['attempts'] = [json.loads(line) for line in (directory/'attempts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    sidecars['source_attempts'] = [json.loads(line) for line in (source_dir/'attempts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    result = stable_document_audit(table, protocol, probe, sidecars, read(source_table_path), source_proof, baseline)
    evidence_files = [table_path, directory/'result.json', directory/'protocol.json', directory/'baseline_replayed_table.json']
    evidence_files += [directory/(name+'.json') for name in ('measurements', 'numerical_validation', 'stability_validation', 'search_controls')]
    evidence_files.append(directory/'attempts.jsonl')
    result.update(status='verified', validation_kind='strict_stable_w4', table=table, table_sha256=digest(table_path),
        probe_result_sha256=digest(directory/'result.json'), probe_protocol_sha256=digest(directory/'protocol.json'),
        baseline_table_sha256=protocol['baseline_table_sha256'], real_checkpoint_tensor_hashes=source_proof['real_checkpoint_tensor_hashes'],
        tensor_verification=source_proof['tensor_verification'], source_projection_proof=source_proof,
        evidence_sha256={legacy.relative(p): digest(p) for p in evidence_files},
        implementation_sources_sha256=protocol['sources_sha256'], auditor_sha256=digest(__file__),
        limitations='Finite random-input byte stability and native equality on layers 0/18/35. Full checkpoint identity uses frozen receipts; selected packed tensors are independently hashed. New whole-model four-cell output gate is still mandatory; old failed wide-all gate remains failed.')
    return result


def is_stable_session(session):
    return session.get('wide_evidence', {}).get('numerical_probe', {}).get('validation_kind') == 'strict_stable_w4' if session.get('wide_evidence') else False


def stable_output_gate(session, sessions, comparisons):
    """Require one complete matching native reference, never mixed cell proofs."""
    if not is_stable_session(session): return dict(status='not_applicable')
    failures = []; references = []; matching = []
    def shape_ready(value):
        return (set(value['cells']) == CELLS and value['http_mode'] == 'nonstream' and value['warmups'] >= 2 and
                all(c['n'] >= 10 and c['output_sequences_identical'] for c in value['cells'].values()))
    if not shape_ready(session): failures.append('Candidate requires four nonstream cells, >=2 warmups, >=10 identical repeats each')
    for ref in sessions:
        if ref['group'] != session['group'] or ref.get('wide_evidence') or not shape_ready(ref): continue
        if (ref['model_weights_sha256'] != session['model_weights_sha256'] or
                ref['model_config_sha256'] != session['model_config_sha256']): continue
        rows = [row for row in comparisons if row['baseline'] == ref['id'] and row['candidate'] == session['id'] and
                row['kind'] == 'same_model_kernel_implementation_change' and row['comparable'] and
                row['exact_output_tokens_equal'] is True and row['both_repeats_identical']]
        if {row['cell'] for row in rows} == CELLS:
            references.append(ref['id']); matching.extend(rows)
    if not references: failures.append('No single complete same-weight/same-engine native four-cell reference with identical output tokens')
    return dict(status='verified' if not failures else 'unverified_stable_four_cell_output_gate',
        missing_cells=sorted(CELLS-{row['cell'] for row in matching}), reasons=failures,
        matching_native_sessions=references, matching_native_comparisons=matching,
        inherited_old_wide_approval=False,
        scope='New full four-cell native comparison; actual inputs, effective engine, fallback table and common sources match. Native-bytewise microtests do not replace model tokens, including forced output after EOS.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and output.suffix == '.json' and not output.exists(), 'Use fresh project JSON output')
    table = read(args.table); protocol = read(args.table.parent/'protocol.json')
    launch = dict(recipe=dict(method=table['algorithm']), model_weights_sha256=table['model_weights_sha256'])
    sources = {name: dict(sha256=sha) for name, sha in protocol['sources_sha256'].items()}
    result = audit_stable_probe(args.table, launch, sources, args.table.parent/'standalone_probe_audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(legacy.canonical(dict(status=result['status'], validation_kind=result['validation_kind'],
        whole_model_output_gate=result['whole_model_output_gate'], output=str(output))))


if __name__ == '__main__': main()
