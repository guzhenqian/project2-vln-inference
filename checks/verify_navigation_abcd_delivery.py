"""Independently recount the final ABCD report from all600 saved episode steps."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def verify(comparison_path):
    comparison = read(comparison_path)
    assert comparison['status'] == 'complete'
    source_paths = list(comparison['sources'])
    assert len(source_paths) == 4
    for name, expected in comparison['sources'].items():
        assert sha(ROOT / name) == expected, name
    expected_ids = {}
    for index in (1, 2, 3):
        manifest = read(ROOT / f'data/baseline/overnight200_fp16_20261031/batch_{index:02d}.json')
        expected_ids.update({str(e): str(t) for e, t in zip(manifest['episodes'], manifest['trajectory_ids'])})
    assert len(expected_ids) == 150
    verified = {}
    outcomes = {}
    for key, source in zip('ABCD', source_paths):
        report_path = ROOT / source
        report = read(report_path)
        assert report['completed_episode_count'] == 150 and report['completed_batches'] == [1, 2, 3]
        assert not report.get('errors') and not report.get('error_count', 0)
        if key == 'A':
            run = ROOT / 'logs/overnight200_fp16_20261031'
        elif key == 'B':
            run = ROOT / 'logs/overnight150_awq_20261104'
        else:
            run = report_path.parent
            assert report['status'] == 'complete' and report['all_runtime_audits_passed'] is True
        episodes = report['episodes']
        assert len(episodes) == len({str(e['episode_id']) for e in episodes}) == 150
        assert {str(e['episode_id']): str(e['trajectory_id']) for e in episodes} == expected_ids
        calls, spls, outcomes[key] = [], [], {}
        for episode in episodes:
            batch = 'batch_01_resumed' if key == 'A' and episode['batch'] == 1 else f"batch_{episode['batch']:02d}"
            folder = run / batch / ('episode_' + str(episode['episode_id']))
            path = folder / 'steps.jsonl'
            assert sha(path) == episode['steps_sha256'], str(path)
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            assert len(rows) == episode['steps']
            assert all(row['step'] == i and str(row['episode_id']) == str(episode['episode_id'])
                       and not row.get('error') for i, row in enumerate(rows))
            success = (rows[-1]['action'] == 'STOP' and episode['metrics']['distance_to_goal'] < 3
                       and episode['terminal_stop_source'] == 'model')
            assert success == bool(episode['metrics']['success'])
            outcomes[key][str(episode['episode_id'])] = success
            shortest, length = episode['start_distance_m'], episode['path_length_m']
            spl = int(success) * shortest / max(shortest, length)
            assert math.isclose(spl, episode['metrics']['spl'], abs_tol=1e-5)
            spls.append(episode['metrics']['spl'] if success else 0)
            calls.extend(float(row['model_request_s']) for row in rows if row.get('model_called') is True)
        measured = comparison['metrics'][key]
        successes = sum(outcomes[key].values())
        assert successes == measured['success_count']
        assert math.isclose(successes / 150, measured['sr'], abs_tol=1e-12)
        assert math.isclose(sum(spls) / 150, measured['spl'], abs_tol=1e-12)
        assert len(calls) == measured['actual_model_calls']
        assert math.isclose(statistics.median(calls), measured['request_median_s'], abs_tol=1e-12)
        assert math.isclose(sum(calls), measured['request_total_s'], abs_tol=1e-6)
        ordered = sorted(calls)
        index = (len(ordered) - 1) * .95
        lower = int(index)
        p95 = ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)
        assert math.isclose(p95, measured['request_p95_s'], abs_tol=1e-12)
        verified[key] = {'episodes_recounted': 150, 'successes': successes,
                         'real_calls_recounted': len(calls), 'saved_step_hashes_match': True,
                         'quality_and_request_statistics_match': True}
        if key == 'D':
            frozen = read(run / 'frozen_plan.json')
            for name, expected in frozen['source_checks'].items():
                assert sha(ROOT / name) == expected, name
            trace_count = 0
            service = ROOT / frozen['service_directory']
            for batch in (1, 2, 3):
                audit = read(run / f'batch_{batch:02d}/runtime_audit.json')
                assert audit['passed'] is True and audit['warmup_requests'] == 2
                for trace in audit['traces']:
                    for kind in ('prefill', 'first_decode'):
                        assert sha(service / (kind + '_' + trace['trace_id'] + '.json')) == trace[kind + '_sha256']
                    trace_count += 1
            assert trace_count == len(calls) + 6
            verified[key].update(frozen_inputs_verified=len(frozen['source_checks']), runtime_trace_pairs_verified=trace_count)
    for key in 'ABC':
        pair = comparison['paired_comparisons']['D_vs_' + key]
        assert set(pair['gained_success']) == {eid for eid in expected_ids if outcomes['D'][eid] and not outcomes[key][eid]}
        assert set(pair['lost_success']) == {eid for eid in expected_ids if outcomes[key][eid] and not outcomes['D'][eid]}
    return {'status': 'passed', 'verified_utc': datetime.now(timezone.utc).isoformat(),
            'comparison_sha256': sha(comparison_path), 'recounts': verified,
            'scope': 'Source-report hashes, all600 saved episodes, identities, step hashes, SR/SPL, real request count/median/P95/total, paired changes; D frozen inputs and all prefill/decode trace hashes. No new inference.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.comparison.resolve())
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
