"""Review retained precision logs without loading models or starting GPU work."""
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from export_navigation_precision_results import verify_result, percentile
from navigation_precision_common import ROOT, RUN, read, digest


def input_key(step):
    payload = [step['request_text'], step['request_image_sha256'],
               step['usage']['prompt_tokens'], step['image_count']]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def main():
    assert read(RUN/'status.json')['status'] == 'cancelled'
    ids = read(ROOT/'datasets/navigation_20261110/selection.json')['same100']
    annotations = read(ROOT/'datasets/navigation_20261110/val_unseen150.json')
    trajectories = {str(e['episode_id']): str(e['trajectory_id']) for e in annotations['episodes']}
    old = ROOT/'logs/navigation_stage2_sequence_20261109/results'
    paths = {'fp16': old/'fp16_100.json', 'awq_w4a16': old/'awq_100.json',
             'gptq_w4a16': old/'gptq_100.json'}
    paths.update({name: RUN/'results'/f'{name}_100.json' for name in (
        'awq_w4a4', 'awq_w4a8', 'awq_w8a16', 'gptq_w4a4', 'gptq_w4a8')})
    groups, first = {}, {}
    for name, path in paths.items():
        result = verify_result(path, ids, trajectories)
        if name not in ('fp16', 'awq_w4a16', 'gptq_w4a16'):
            for source, sha in result['source_hashes'].items():
                assert digest(ROOT/'checks'/source) == sha
            for batch, sha in result['runtime_audit_sha256'].items():
                audit = ROOT/batch/'runtime_audit.json'
                assert digest(audit) == sha
                assert read(audit)['passed'] and read(audit)['infrastructure_errors'] == 0
        requests = []
        first[name] = {}
        for ep in result['episodes']:
            eid = ep['episode_id']
            source = next(p for p in result['step_files_sha256']
                          if p.endswith(f'/episode_{eid}/steps.jsonl'))
            steps = [json.loads(line) for line in (ROOT/source).read_text(encoding='utf-8').splitlines()]
            calls = [s for s in steps if s['model_called']]
            assert calls and calls[0]['step'] == 0
            requests.extend(calls)
            s = calls[0]
            first[name][eid] = dict(input_sha256=input_key(s), step=0,
                input_tokens=s['usage']['prompt_tokens'], image_count=s['image_count'],
                output_tokens=s['usage']['completion_tokens'], request_s=s['model_request_s'],
                finish_reason=s['finish_reason'], error=s.get('error'), source=source)
        tokens = [s['usage']['completion_tokens'] for s in requests]
        groups[name] = dict(result_path=path.relative_to(ROOT).as_posix(), result_sha256=digest(path),
            metrics={k: result[k] for k in ('case_count', 'success_count', 'spl', 'actual_model_calls',
                'request_median_s', 'request_p95_s', 'sampled_whole_card_peak_mib', 'error_count',
                'quality_error_counts')},
            completion_tokens=dict(median=statistics.median(tokens), p95=percentile(tokens, .95),
                maximum=max(tokens), requests_at_512=sum(t == 512 for t in tokens)),
            finish_reasons=dict(Counter(s['finish_reason'] for s in requests)),
            image_counts=dict(Counter(s['image_count'] for s in requests)))
    pairs = []
    for control, candidate in [('fp16', name) for name in paths if name != 'fp16'] + [
            ('awq_w4a16', 'awq_w4a8'), ('awq_w4a16', 'awq_w8a16'),
            ('gptq_w4a16', 'gptq_w4a8')]:
        matching, same_length = [], []
        for eid in ids:
            a, b = first[control][eid], first[candidate][eid]
            if a['input_sha256'] != b['input_sha256']:
                continue
            matching.append(eid)
            if (a['output_tokens'] == b['output_tokens'] and not a['error'] and not b['error']
                    and a['finish_reason'] == b['finish_reason'] == 'stop'):
                same_length.append(eid)
        row = dict(control=control, candidate=candidate, matching_initial_inputs=len(matching),
                   matching_input_and_output_length=len(same_length), case_ids=same_length)
        if same_length:
            row.update(control_request_median_s=statistics.median(first[control][e]['request_s'] for e in same_length),
                candidate_request_median_s=statistics.median(first[candidate][e]['request_s'] for e in same_length),
                paired_slowdown_median=statistics.median(first[candidate][e]['request_s']/first[control][e]['request_s'] for e in same_length))
        pairs.append(row)
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='offline_review_complete',
        no_new_gpu_tests=True, experiment_state='cancelled_by_user',
        completed_quantized_groups=7, baseline_groups=1,
        incomplete_group=dict(name='gptq_w8a16', retained_completed_cases=50, complete100_metrics=None),
        groups=groups, initial_request_pairs=pairs, first_requests=first,
        caveats=['Matched initial requests are observational, from separate run dates; not repeated interleaved timing trials.',
                 'Output length equality does not imply identical generated content. First requests contain two images.',
                 'No new prefill/decode timing or kernel attribution was measured for the experimental configurations.',
                 'W4A4 long outputs are quality failures, not equal-workload kernel speed comparisons.'],
        source_sha256={p:digest(ROOT/p) for p in (
            'checks/analyze_navigation_precision_latency_offline.py',
            'checks/navigation_precision_kernels.py', 'checks/navigation_precision_runtime.py',
            'checks/check_navigation_precision_kernels.py', 'reports/navigation_quant_timing_20261110.json')})
    output = ROOT/'reports/navigation_precision_latency_review_20261112.json'
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status':data['status'], 'tokens':{k:v['completion_tokens'] for k,v in groups.items()},
                      'pairs':pairs}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
