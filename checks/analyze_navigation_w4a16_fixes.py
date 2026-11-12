"""CPU-only independent audit/comparison of completed fixed-input W4A16 runs.

Usage: python -B checks/analyze_navigation_w4a16_fixes.py --runs RUN [RUN ...]
           --output reports/NEW_REPORT.json

Only standard-library imports are used. No model loading, CUDA initialization,
inference, or subprocesses occur. The output must not already exist. Failed or
unfinished controller attempts are rejected, never treated as complete data.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    'fp16': 'b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b',
    'awq': '31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
    'gptq': '2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f',
}
METRICS = ('http_total_s', 'ttft_s', 'stream_decode_ms_per_token',
           'prefill_language_ms', 'prefill_vision_ms', 'prefill_model_ms',
           'decode_model_ms_per_token', 'decode_through_logits_ms_per_token')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def label(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def positive(value, context, allow_zero=False):
    assert isinstance(value, (int, float)) and not isinstance(value, bool), context
    assert math.isfinite(value) and (value >= 0 if allow_zero else value > 0), context
    return value


def same(actual, expected, context='value'):
    """Deep comparison with tight tolerance for recomputed floating timings."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), context
        for key in expected:
            same(actual[key], expected[key], context + '/' + str(key))
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), context
        for i, value in enumerate(expected):
            same(actual[i], value, context + '/' + str(i))
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1.e-9, abs_tol=1.e-7), context
    else:
        assert actual == expected, (context, actual, expected)


def sequence_sha(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def summarize(rows):
    result = []
    for sample in (0, 1):
        for tokens in (18, 64):
            chosen = [r for r in rows if (r['sample'], r['tokens']) == (sample, tokens)]
            assert chosen, (sample, tokens)
            result.append(dict(sample=sample, tokens=tokens, n=len(chosen),
                medians={k: statistics.median(r[k] for r in chosen) for k in METRICS},
                ranges={k: [min(r[k] for r in chosen), max(r[k] for r in chosen)] for k in METRICS},
                output_sequences_identical=all(r['output_tokens'] == chosen[0]['output_tokens'] for r in chosen),
                output_tokens=chosen[0]['output_tokens']))
    return result


class Audit:
    def __init__(self):
        self.artifacts = {}
        self.tensor_inputs = None
        self.protocol_inputs = None
        self.engine = None
        self.model_identities = {}
        self.source_versions = defaultdict(set)
        self.request_hashes = {}
        self.request_idle_seconds = None
        self.run_conditions = []

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        key = label(path)
        sha = hashlib.sha256(raw).hexdigest()
        if key in self.artifacts:
            assert self.artifacts[key] == sha, 'Artifact changed during analysis: ' + key
        self.artifacts[key] = sha
        return json.loads(raw)

    def sources(self, sources):
        for name, sha in sources.items():
            assert Path(name).name == name and len(sha) == 64 and int(sha, 16) >= 0
            self.source_versions[name].add(sha)

    def inspect_session(self, run, index, recorded, protocol):
        session = recorded['session']
        group, variant, graph = (session[k] for k in ('group', 'variant', 'graph'))
        assert group in FROZEN and graph in ('off', 'on')
        expected_session = dict(protocol['sessions'][index], directory=session['directory'])
        same(session, expected_session, 'session identity')
        directory = (run/session['directory']).resolve()
        assert directory.is_relative_to(run) and directory != run
        service = directory/'service'
        launch = self.read(service/'launch.json')
        for key in ('group', 'variant', 'graph'):
            assert launch[key] == session[key], key
        self.sources(launch['sources_sha256'])
        for name, sha in protocol['sources_sha256'].items():
            if name in launch['sources_sha256']:
                assert launch['sources_sha256'][name] == sha, name
        identity = (launch['model'], launch['model_config_sha256'],
                    launch['model_safetensors_size'], protocol['actual_weight_sha256'][group])
        if group in self.model_identities:
            same(list(identity), list(self.model_identities[group]), 'frozen model identity/' + group)
        self.model_identities[group] = identity
        engine = dict(launch['engine'])
        assert engine.pop('eager_mode') == (graph == 'off')
        expected_engine = dict(dtype='float16', tp=1, dp=1, session_len=4096,
            max_batch_size=1, max_prefill_token_num=3072, num_gpu_blocks=64,
            block_size=64, enable_prefix_caching=False, quant_policy=0)
        for key, value in expected_engine.items():
            assert engine[key] == value, ('engine', key)
        if self.engine is not None:
            same(engine, self.engine, 'common engine configuration')
        self.engine = engine
        execution = self.read(service/'execution_audit.json')
        assert execution['first_native_forward_verified'] and execution['native_forward_preserved']
        assert (execution['group'], execution['variant'], execution['graph_requested']) == (group, variant, graph)
        if group != 'fp16':
            receipt = execution['native_awq_receipt']
            assert receipt['verified'] and receipt['module_count'] == 144
            assert receipt['loaded_quantized_tensor_count'] == 756
            assert len(receipt['module_calls']) == 144 and all(r['calls'] == 1 for r in receipt['module_calls'])
        else:
            assert execution['native_awq_receipt'] is None

        inputs = {}
        for length in (701, 2815):
            evidence = self.read(service/f'input_{length}.json')
            inputs[str(length)] = {k: evidence[k] for k in ('input_ids', 'pixel_values')}
            assert evidence['input_ids']['shape'] == [1, length]
            assert evidence['input_ids']['dtype'] == 'torch.int64'
            assert evidence['pixel_values']['dtype'] == 'torch.float32'
        if self.tensor_inputs is not None:
            same(inputs, self.tensor_inputs, 'actual processed input tensors across sessions/runs')
        self.tensor_inputs = inputs

        plan = self.read(service/'request_plan.json')
        expected_plan = [dict(phase='warmup', sample=s, tokens=t, repeat=r)
            for s in (0, 1) for t in (18, 64) for r in range(protocol['warmups'])]
        expected_plan += [dict(phase='measured', sample=s, tokens=t, repeat=r)
            for r in range(protocol['repeats']) for s in (0, 1) for t in (18, 64)]
        expected_plan += [dict(phase='flush', sample=0, tokens=18, repeat=0)]
        same(plan, expected_plan, 'request plan')
        requests = self.read(directory/'requests.json')
        assert len(requests) == len(plan)
        measurements = []
        for i, request in enumerate(requests):
            spec = plan[i]
            same(request['spec'], spec, 'request spec')
            count, prompt = spec['tokens'], 701 if spec['sample'] == 0 else 2815
            tokens, arrivals, usage = request['output_tokens'], request['token_arrival_s'], request['usage']
            assert len(tokens) == len(arrivals) == usage['completion_tokens'] == count
            assert usage['prompt_tokens'] == prompt and usage['total_tokens'] == prompt + count
            assert all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in tokens)
            assert all(positive(t, 'arrival') for t in arrivals)
            assert all(a <= b for a, b in zip(arrivals, arrivals[1:]))
            positive(request['total_s'], 'HTTP time')
            assert request['total_s'] >= arrivals[-1]
            same(request['ttft_s'], arrivals[0], 'TTFT')
            stream_decode = 1000 * (arrivals[-1] - arrivals[0]) / (count - 1)
            positive(stream_decode, 'stream decode')
            same(request['decode_ms_per_token'], stream_decode, 'stream token intervals')
            key = (group, spec['sample'], count)
            sha = request['request_sha256']
            if key in self.request_hashes:
                assert sha == self.request_hashes[key], 'Request payload differs across implementations'
            self.request_hashes[key] = sha
            if spec['phase'] == 'flush':
                continue  # Its purpose is to flush the preceding measured timing.
            timing = self.read(service/f'timing_{i:03d}.json')
            assert timing['index'] == i and timing['plan'] == spec
            assert (timing['group'], timing['variant'], timing['graph_requested']) == (group, variant, graph)
            assert timing['capture_replay_observed_outside_graph']
            steps = timing['steps']
            assert len(steps) > 1 and steps[0]['phase'] == 'prefill' and steps[0]['rows'] == prompt
            assert steps[0]['execution_mode'] == 'eager'
            assert all(s['phase'] == 'decode' and s['rows'] == 1 for s in steps[1:])
            same(timing['logical_forward_counts'], dict(Counter(s['phase'] for s in steps)), 'forward counts')
            same(timing['actual_execution_modes'], dict(Counter(s['execution_mode'] for s in steps)), 'mode counts')
            for step in steps:
                assert step['execution_mode'] in ('eager', 'capture', 'replay')
                for name, timer in step['timers'].items():
                    positive(timer['cuda_interval_ms'], name + ' CUDA time')
                    positive(timer['host_wall_ms'], name + ' host time', allow_zero=True)
                positive(step['model_through_logits_ms'], 'through logits')
                assert step['model_through_logits_ms'] + 1.e-6 >= step['timers']['model']['cuda_interval_ms']
            if spec['phase'] != 'measured':
                continue
            pre, dec = steps[0], steps[1:]
            modes = dict(Counter(s['execution_mode'] for s in dec))
            assert modes == {('replay' if graph == 'on' else 'eager'): len(dec)}
            if graph == 'on':
                assert all(s['graph_keys'] for s in dec), 'No graph replay key recorded'
            measurements.append(dict(sample=spec['sample'], tokens=count, repeat=spec['repeat'],
                http_total_s=request['total_s'], ttft_s=request['ttft_s'],
                stream_decode_ms_per_token=stream_decode,
                prefill_language_ms=pre['timers']['language']['cuda_interval_ms'],
                prefill_vision_ms=pre['timers']['vision']['cuda_interval_ms'],
                prefill_model_ms=pre['timers']['model']['cuda_interval_ms'],
                decode_model_ms_per_token=statistics.mean(s['timers']['model']['cuda_interval_ms'] for s in dec),
                decode_through_logits_ms_per_token=statistics.mean(s['model_through_logits_ms'] for s in dec),
                actual_decode_forwards=len(dec), execution_modes=modes, output_tokens=tokens))
        calculated = dict(session=session, summary=summarize(measurements), measurements=measurements)
        same(self.read(directory/'result.json'), calculated, 'per-session result recomputed')
        same(recorded, calculated, 'controller session result recomputed')
        samples = self.read(directory/'gpu_samples.json')
        memory = []
        invalid = 0
        for sample in samples:
            if 'error' in sample:
                invalid += 1
                continue
            try:
                value = float(sample['value'].split(',')[0].strip())
                positive(value, 'whole-card memory')
                memory.append(value)
            except (ValueError, KeyError, TypeError):
                invalid += 1
        assert memory, 'No valid whole-card memory samples'
        return dict(run=label(run), directory=label(directory), group=group, variant=variant, graph=graph,
            request_idle_seconds=float(protocol.get('request_idle_seconds', 0)),
            idle_scope=protocol.get('idle_scope', 'Back-to-back requests; no intentional untimed idle.'),
            summary=calculated['summary'], measurements=measurements,
            memory=dict(sampled_whole_card_peak_mib=max(memory), valid_samples=len(memory),
                failed_samples=invalid, sampling_window='Service readiness through warmups, measured requests and tail flush; model loading excluded.',
                scope='One-second whole-card nvidia-smi sampling, including desktop and graph pools; NOT model-only or torch allocated memory.'))

    def inspect_run(self, run):
        result = self.read(run/'result.json')
        status = self.read(run/'status.json')
        assert result['status'] == status['status'] == 'complete', 'Incomplete/failed run rejected'
        protocol = self.read(run/'protocol.json')
        idle = protocol.get('request_idle_seconds', 0)
        positive(idle, 'request_idle_seconds', allow_zero=True)
        idle = float(idle)
        if self.request_idle_seconds is not None:
            assert idle == self.request_idle_seconds, (
                'Cannot pool runs with different untimed request-idle conditions. '
                'Analyze zero-idle screening and idle-separated confirmation independently.',
                label(run), idle, self.request_idle_seconds)
        self.request_idle_seconds = idle
        self.run_conditions.append(dict(run=label(run), request_idle_seconds=idle,
            started_utc=protocol.get('started_utc'),
            idle_scope=protocol.get('idle_scope', 'Back-to-back requests; no intentional untimed idle.')))
        assert protocol['warmups'] >= 1 and protocol['repeats'] >= 1
        assert len(result['sessions']) == len(protocol['sessions']) == status['completed_sessions']
        for group in {s['group'] for s in protocol['sessions']}:
            assert protocol['actual_weight_sha256'][group] == FROZEN[group], 'Unrecognized checkpoint SHA'
        self.sources(protocol['sources_sha256'])
        if self.protocol_inputs is not None:
            same(protocol['inputs'], self.protocol_inputs, 'frozen request input inventory')
        self.protocol_inputs = protocol['inputs']
        sessions = [self.inspect_session(run, i, row, protocol) for i, row in enumerate(result['sessions'])]
        same(result['actual_input_tensors'], self.tensor_inputs, 'controller processed tensors')
        return sessions


def aggregate(sessions):
    assert len({s['request_idle_seconds'] for s in sessions}) == 1, (
        'Cannot pool sessions with different untimed request-idle conditions')
    groups = defaultdict(list)
    for session in sessions:
        groups[(session['group'], session['variant'], session['graph'])].append(session)
    output = []
    for (group, variant, graph), chosen in sorted(groups.items()):
        rows = [r for s in chosen for r in s['measurements']]
        summary = summarize(rows)
        for item in summary:
            matching = [r for r in rows if (r['sample'], r['tokens']) == (item['sample'], item['tokens'])]
            item['actual_decode_forward_counts'] = dict(sorted(Counter(r['actual_decode_forwards'] for r in matching).items()))
            item['output_sequence_counts'] = dict(sorted(Counter(sequence_sha(r['output_tokens']) for r in matching).items()))
        output.append(dict(group=group, variant=variant, graph=graph, session_count=len(chosen),
            run_count=len({s['run'] for s in chosen}), session_directories=[s['directory'] for s in chosen],
            request_idle_seconds=chosen[0]['request_idle_seconds'],
            summary=summary, sampled_whole_card_peak_mib=max(s['memory']['sampled_whole_card_peak_mib'] for s in chosen)))
    return output


def comparisons(configurations):
    index = {(c['group'], c['variant'], c['graph']): c for c in configurations}
    output = []
    for candidate in configurations:
        group, variant, graph = (candidate[k] for k in ('group', 'variant', 'graph'))
        controls = []
        if graph == 'on':
            controls.append(((group, variant, 'off'), 'graph_change_same_weight_and_kernel'))
        if variant != 'stock':
            controls.append(((group, 'stock', graph), 'kernel_change_same_weight_and_graph'))
            if graph == 'on':
                controls.append(((group, 'stock', 'off'), 'combined_kernel_and_graph_change'))
        if group != 'fp16':
            controls.append((('fp16', 'stock', graph), 'quantized_deployment_vs_fp16_same_graph'))
        for control_key, kind in controls:
            if control_key not in index:
                continue
            control = index[control_key]
            cells = []
            for a, b in zip(control['summary'], candidate['summary']):
                assert (a['sample'], a['tokens']) == (b['sample'], b['tokens'])
                stats = {}
                for key in METRICS:
                    baseline, observed = a['medians'][key], b['medians'][key]
                    stats[key] = dict(control_median=baseline, candidate_median=observed,
                        delta=observed-baseline, latency_reduction_pct=100*(baseline-observed)/baseline,
                        speedup_ratio=baseline/observed, control_range=a['ranges'][key], candidate_range=b['ranges'][key])
                exact = (a['output_sequences_identical'] and b['output_sequences_identical']
                         and a['output_tokens'] == b['output_tokens'])
                counts_a, counts_b = set(a['actual_decode_forward_counts']), set(b['actual_decode_forward_counts'])
                cells.append(dict(sample=a['sample'], tokens=a['tokens'], control_n=a['n'], candidate_n=b['n'],
                    exact_output_tokens_equal_for_all_requests=exact,
                    control_output_sequence_counts=a['output_sequence_counts'], candidate_output_sequence_counts=b['output_sequence_counts'],
                    actual_decode_workload_identical_and_constant=len(counts_a)==len(counts_b)==1 and counts_a==counts_b,
                    control_decode_forward_counts=a['actual_decode_forward_counts'], candidate_decode_forward_counts=b['actual_decode_forward_counts'],
                    timing=stats))
            output.append(dict(kind=kind, control=dict(zip(('group', 'variant', 'graph'), control_key)),
                candidate={k:candidate[k] for k in ('group', 'variant', 'graph')}, cells=cells))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists(), 'Refuse to overwrite an existing report'
    runs = [p.resolve() for p in args.runs]
    assert len(set(runs)) == len(runs), 'Duplicate input run'
    audit = Audit()
    sessions = [s for run in runs for s in audit.inspect_run(run)]
    configurations = aggregate(sessions)
    current_sources = {}
    for name, versions in sorted(audit.source_versions.items()):
        path = ROOT/'checks'/name
        current = digest(path) if path.is_file() else None
        current_sources[name] = dict(recorded_sha256=sorted(versions), current_sha256=current,
                                     current_matches_all_recorded=versions == {current})
    report = dict(status='complete_independently_audited', created_utc=datetime.now(timezone.utc).isoformat(),
        runs=[label(r) for r in runs], analyzer_sha256=digest(__file__),
        validation=dict(raw_requests_and_timing_summaries_recomputed=True,
            actual_input_tensor_hashes_identical=True, frozen_checkpoint_recorded_sha_verified=True,
            checkpoint_scope='Compared controller-recorded full checkpoint SHA with frozen identities; analyzer did not reread multi-GB weight files.',
            actual_graph_replay_verified_for_every_measured_graph_decode=True,
            returned_token_counts_verified=[18, 64], session_count=len(sessions),
            measured_requests=sum(len(s['measurements']) for s in sessions),
            identical_declared_request_idle_condition=True,
            source_versions_consistent_across_complete_runs=all(len(v)==1 for v in audit.source_versions.values())),
        request_idle_seconds=audit.request_idle_seconds, run_conditions=audit.run_conditions,
        metric_definitions=dict(
            stream_decode_ms_per_token='Elapsed first-to-last returned token divided by returned_tokens-1;17 gaps for18 tokens and63 gaps for64.',
            decode_model_ms_per_token='Mean CUDA-event model interval per ACTUAL backend decode forward, including any post-response lookahead forwards; legacy field name is retained.',
            decode_through_logits_ms_per_token='Mean model-through-logits interval per ACTUAL backend decode forward, including any post-response lookahead forwards.',
            actual_decode_forwards='Backend decode invocations excluding prefill. Record observed counts without assuming one-to-one alignment with returned tokens or an exact number of discarded tokens.'),
        actual_input_tensors=audit.tensor_inputs, input_inventory=audit.protocol_inputs,
        frozen_checkpoint_sha256={k:FROZEN[k] for k in audit.model_identities},
        common_engine_excluding_graph_switch=audit.engine,
        configurations=configurations, comparisons=comparisons(configurations), sessions=sessions,
        source_versions=current_sources, artifact_sha256=audit.artifacts,
        caveats=[
            'Fixed two/nine-image prompts, greedy forced18/64 outputs; not closed-loop navigation or a success-rate evaluation.',
            'Exact token equality on these prompts is a regression check, not proof of unchanged navigation quality.',
            'CUDA-event intervals may include host-dispatch/idle gaps; streaming decode also includes scheduling/sampling.',
            'Actual backend forward counts are reported separately from returned tokens; asynchronous lookahead may differ.',
            'decode_model_ms_per_token and decode_through_logits_ms_per_token average actual backend forwards, potentially including post-response work; stream_decode_ms_per_token averages returned-token gaps. These denominators are different.',
            'Zero-idle screening sends back-to-back requests. A response returning18 tokens can execute32 backend decode forwards; remaining lookahead can queue into the next request TTFT. Do not treat that TTFT as an isolated-request measurement.',
            'Idle-separated confirmation declares a one-second pause before every request, outside its timer, to reduce previous-request carryover. It is a separate load/queueing condition, not pooled with zero-idle screening. The idle is verified from protocol/source records, not independently reconstructed from absolute request timestamps.',
            'Graph improvements are execution/scheduling improvements, not quantization-only gains; compare FP16 and W4 under the same graph mode.',
            'Memory is whole-card sampling across warmup/capture and measured requests; not model-allocated memory or an exact transient peak.',
            'Repeated timings on two inputs and ranges do not establish statistical significance or generalize to all trajectories.',
            'Source versions are retained per run. If versions differ, comparisons can confound code changes with variant/graph choices.',
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=report['status'], output=str(output),
        measured_requests=report['validation']['measured_requests'], sessions=len(sessions),
        configurations=len(configurations), comparisons=len(report['comparisons']))))


if __name__ == '__main__':
    main()
