"""Recompute selected-case metrics from saved steps, including reused DART runs."""
import argparse
import json
import math
import statistics
from pathlib import Path
from navigation_stage2_common import ROOT, RUN, PLAN, read, write, digest

DART = {
    'dart75': 'logs/navigation_dart150_keep75_resumed_20261108T135623640378Z_99f2e6a6',
    'dart50': 'logs/navigation_dart150_keep50_20261108T182759405011Z_7a8081e1',
}


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered)-1)*fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(index-lower)


def collect(group, directories, limit):
    roster = read(PLAN/'manifest.json')['pool150'][:limit]
    expected_ids = {}
    for index in range(1,limit//50+1):
        manifest = read(PLAN/f'source_batch_{index:02d}.json')
        expected_ids.update(dict(zip(map(str,manifest['episodes']),map(str,manifest['trajectory_ids']))))
    results, latencies, memory_files = {}, [], []
    step_hashes = {}
    timing_notes = []
    reference = None
    if group in DART:
        run = ROOT/DART[group]
        reference = read(run/'report.json')
        assert reference['status']=='complete' and reference['all_runtime_audits_passed']
        rows = {str(row['episode_id']):row for row in reference['episodes']}
        directories = [run/f'batch_{i:02d}' for i in range(1,limit//50+1)]
        memory_files = [ROOT/path for path in reference.get('memory_source_files', [])
            if any(f'/batch_{i:02d}/' in path for i in range(1,limit//50+1))]
    explicit_memory_sources = bool(memory_files)
    for folder in directories:
        summary = read(folder/'summary.json')
        assert summary['status']=='complete'
        if not explicit_memory_sources:
            memory_files.append(folder/'gpu_samples.csv')
        for episode in summary['episodes']:
            eid = str(episode['episode_id'])
            if eid not in roster:
                continue
            assert eid not in results and str(episode['trajectory_id']) == expected_ids[eid]
            if reference is not None:
                episode = rows[eid]
            path = folder/('episode_'+eid)/'steps.jsonl'
            hashed = digest(path)
            if reference is not None:
                assert hashed == episode['steps_sha256']
            steps = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(steps)==episode['steps'] and all(row['step']==i for i,row in enumerate(steps))
            real = [float(row['model_request_s']) for row in steps if row.get('model_called') is True]
            assert real and all(math.isfinite(value) and value>0 for value in real)
            latencies.extend(real)
            last = steps[-1]
            success = int(last['action']=='STOP' and episode['terminal_stop_source']=='model'
                and not last.get('error') and float(episode['metrics']['distance_to_goal'])<3.)
            spl = float(episode['metrics']['spl']) if success else 0.
            assert math.isclose(spl, success*episode['start_distance_m']/max(episode['start_distance_m'],episode['path_length_m']),abs_tol=1e-5)
            results[eid] = dict(episode_id=eid,trajectory_id=str(episode['trajectory_id']),
                success=success,spl=spl,steps=len(steps),actual_model_calls=len(real),
                request_total_s=sum(real),error_count=sum(bool(row.get('error')) for row in steps),
                collisions=sum(bool(row['metrics']['collisions']['is_collision']) for row in steps),
                terminal_stop_source=episode['terminal_stop_source'])
            step_hashes[path.relative_to(ROOT).as_posix()] = hashed
            if episode.get('timing_note'):
                timing_notes.append(dict(episode_id=eid,timing_note=episode['timing_note']))
    assert set(results)==set(roster) and len(results)==limit
    # C75 batch1 has both original and resumed samples; preserve the same historic scope.
    memory_files = list(dict.fromkeys(memory_files))
    memory = []
    for path in memory_files:
        for line in path.read_text().splitlines():
            parts = line.split(',')
            if len(parts)>=3:
                try:
                    memory.append(float(parts[1]))
                except ValueError:
                    pass
    assert memory
    ordered = [results[eid] for eid in roster]
    return dict(group=group,status='complete',case_count=limit,reused=group in DART,
        success_count=sum(row['success'] for row in ordered),sr=sum(row['success'] for row in ordered)/limit,
        spl=sum(row['spl'] for row in ordered)/limit,actual_model_calls=len(latencies),
        request_median_s=statistics.median(latencies),request_p95_s=percentile(latencies,.95),
        request_total_s=sum(latencies),sampled_whole_card_peak_mib=max(memory),
        error_count=sum(row['error_count'] for row in ordered),
        collisions=sum(row['collisions'] for row in ordered),episodes=ordered,
        step_files_sha256=step_hashes,memory_source_files={path.relative_to(ROOT).as_posix():digest(path) for path in memory_files},
        timing_notes=timing_notes,
        limitations=['Fixed development subset; one run per setting; no independent generalization claim.',
            'Only actual HTTP requests; warmups and queued zero-second actions excluded.',
            '200ms whole-card sampling; retained historical recovery sampling scope for DART75.',
            'Runtime instrumentation and method scoring overhead remain included; no pure-kernel speedup claim.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-dart',action='store_true')
    args=parser.parse_args()
    if args.reuse_dart:
        output=RUN/'reused_dart_v2'
        output.mkdir(exist_ok=True)
        for group in DART:
            complete=collect(group,[],150)
            reference=read(ROOT/DART[group]/'report.json')
            assert math.isclose(complete['sr'],reference['sr'],abs_tol=1e-12)
            assert math.isclose(complete['spl'],reference['spl'],abs_tol=1e-12)
            assert complete['actual_model_calls']==reference['actual_request_latency_s']['count']
            assert math.isclose(complete['request_total_s'],reference['actual_request_latency_s']['total'],abs_tol=1e-6)
            assert complete['sampled_whole_card_peak_mib']==reference['sampled_whole_card_peak_mib']
            for limit in (50,100):
                path=output/f'{group}_{limit}.json'
                assert not path.exists(), 'Preserve previous export'
                result=collect(group,[],limit)
                assert len(result['memory_source_files'])==(limit//50 + (1 if group=='dart75' else 0))
                write(path,result)
                print(group,limit,'success',result['success_count'],'calls',result['actual_model_calls'])
        write(output/'verification.json',dict(status='passed',full150_recomputed_against_both_historical_reports=True,
            batch_memory_sources_complete=True,old_draft='reused_dart directory retained as superseded; dart50_100 memory missed batch02',
            collector_sha256=digest(Path(__file__))))


if __name__=='__main__':
    main()
