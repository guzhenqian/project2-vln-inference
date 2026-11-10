"""Reuse frozen navigation invariants and additionally require real low-bit receipts."""
import audit_navigation_stage2 as original
import json
from collections import Counter
from navigation_precision_quality import verify_step
from navigation_precision_common import ROOT, GROUPS, SOURCES, read, digest, model_name

original.GROUPS, original.SOURCES, original.model_name = GROUPS, SOURCES, model_name
base_trace = original.audit_trace
base_service = original.audit_service


def audit_service(service, group):
    launch = base_service(service, group)
    recipe = GROUPS[group]
    model = ROOT/recipe['model']
    conversion = read(model/'conversion_audit.json')
    assert launch['model_weights_sha256'] == conversion['sha256']
    assert launch['model_bytes'] == conversion['bytes'] == (model/'model.safetensors').stat().st_size
    assert launch['conversion_audit_sha256'] == digest(model/'conversion_audit.json')
    assert launch['precision_config_sha256'] == digest(ROOT/'setup/navigation-precision-20261110.json')
    init = read(next(service.glob('runtime_init_*.json')))
    for key in ('weight_bits', 'activation_bits'):
        assert init[key] == recipe[key]
    assert init['stock_backend'] is False
    assert all(row['backend_class'] == 'NavigationPrecisionImpl' for row in init['awq_modules']['modules'])
    return launch


def audit_trace(path, group):
    result = base_trace(path, group)
    row = read(path)
    for record in (row, read(path.with_name('first_decode_'+row['trace_id']+'.json'))):
        receipt = record['awq_execution']
        assert receipt['precision_extension_executed'] and receipt['quantized_projection_calls'] == 144
        for key in ('weight_bits', 'activation_bits'):
            assert receipt[key] == GROUPS[group][key]
        assert receipt['compute'] == GROUPS[group]['compute']
        if GROUPS[group]['activation_bits'] < 16:
            assert receipt['no_fp16_activation_fallback']
        if GROUPS[group]['activation_bits'] == 4:
            assert receipt['activation_packed']
    return result


original.audit_trace, original.audit_service = audit_trace, audit_service

def audit_batch(folder, service, group, before):
    audit_service(service, group)
    summary, manifest = read(folder/'summary.json'), read(folder/'manifest.json')
    assert summary['status'] == 'complete'
    assert summary['model'] == manifest['model'] == model_name(group)
    assert summary['warmup_requests_excluded'] == 2
    expected = dict(zip(map(str, manifest['episodes']), map(str, manifest['trajectory_ids'])))
    assert {str(r['episode_id']): str(r['trajectory_id']) for r in summary['episodes']} == expected
    assert len(summary['episodes']) == len(expected)
    calls, evidence, errors = [], {}, Counter()
    for result in summary['episodes']:
        path = folder/('episode_'+str(result['episode_id']))/'steps.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == result['steps'] and all(r['step'] == i for i, r in enumerate(rows))
        for row in rows:
            error = verify_step(row)
            if error:
                errors[error] += 1
                assert row is rows[-1] and result['terminal_stop_source'] == 'error_fallback'
        assert result['error_count'] == sum(bool(r.get('error')) for r in rows)
        calls.extend(r for r in rows if r['model_called'])
        evidence[str(path.relative_to(folder))] = digest(path)
    assert summary['error_count'] == sum(errors.values())
    warmups = read(folder/'warmups.json')
    assert len(warmups) == 2
    for row in warmups:
        verify_step(row)
        assert row['model_called']
    fresh = [p for p in service.glob('prefill_*.json') if p.name not in set(before)]
    fresh.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    assert len(fresh) == len(calls)+2, 'Trace count differs from real calls + warmups'
    traces = [audit_trace(p, group) for p in fresh]
    signatures = [(r['image_count'], r['usage']['prompt_tokens']) for r in warmups+calls]
    assert [(r['image_count'], r['original_length']) for r in traces] == signatures
    return dict(passed=True, group=group, episodes=len(expected), actual_calls=len(calls),
        warmups=2, traces=traces, step_files_sha256=evidence,
        quality_errors=dict(errors), quality_error_count=sum(errors.values()), infrastructure_errors=0,
        quality_failures_retained_in_denominator=True, warmups_sha256=digest(folder/'warmups.json'),
        summary_sha256=digest(folder/'summary.json'), launch_sha256=digest(service/'launch.json'))
