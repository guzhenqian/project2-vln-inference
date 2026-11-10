"""Navigation-trained SimpleVLN checkpoint diagnostic; original 2B baseline is preserved."""
import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
from simplevln_env import ROOT, ACTIONS, make_env, pose

from simplevln_policy import SYSTEM, request_action
import simplevln_policy
import run_closed_loop_baseline


def rgb_request(rgb, instruction, history, endpoint, model, feedback):
    return run_closed_loop_baseline.request_action(rgb, instruction, history, endpoint, model)


def execution_feedback(previous_row):
    """Expose only local motion magnitude and bump feedback, never navigation truth."""
    if previous_row is None:
        return {'previous_action':None,'observed_displacement_m':None,
                'collision_detected':None,'forward_execution':None}
    displacement = float(np.linalg.norm(np.asarray(previous_row['after']['position'])
                                        - np.asarray(previous_row['before']['position'])))
    action = previous_row['action']
    status = None
    if action == 'MOVE_FORWARD':
        status = 'blocked' if displacement < 0.02 else 'partial' if displacement < 0.125 else 'moved'
    return {'previous_action':action,'observed_displacement_m':round(displacement,4),
            'collision_detected':bool(previous_row['metrics']['collisions']['is_collision']),
            'forward_execution':status}


def main():
    global SYSTEM, request_action
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--policy',choices=['memory','rgb'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint',default='http://127.0.0.1:23333/v1')
    args = parser.parse_args()
    if args.policy=='rgb':
        SYSTEM, request_action = run_closed_loop_baseline.SYSTEM, rgb_request
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    summary_path = args.output/'summary.json'
    if summary_path.exists():
        raise RuntimeError('Output already has a summary; choose a fresh run directory')
    # Check the server before constructing a simulator or changing run output.
    with urllib.request.urlopen(args.endpoint+'/models',timeout=10) as response:
        models = json.load(response)
    assert manifest['model'] in [m['id'] for m in models['data']], models
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'system_prompt.txt').write_text(SYSTEM)
    provenance = {'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  'system_prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
                  'started_utc':datetime.utcnow().isoformat()+'Z',
                  'scripts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),ROOT/'checks/simplevln_env.py',ROOT/'checks/simplevln_policy.py',ROOT/'checks/run_closed_loop_baseline.py']}}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    all_records = []
    episode_results = []
    memory_stream = (args.output/'gpu_samples.csv').open('w')
    sampler = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits','--loop-ms=200'],stdout=memory_stream,stderr=subprocess.DEVNULL)
    env = None
    try:
        env, config = make_env(manifest['episodes'],max_steps=manifest['max_episode_steps'], dataset_path=manifest.get('dataset_path'), split=manifest.get('split','val_seen'))
        (args.output/'habitat_config.yaml').write_text(config.dump())
        # Two fixed warmups; excluded from latency samples and trajectories.
        observations = env.reset()
        instruction = env.current_episode.instruction.instruction_text
        from navigation_precision_quality import verify_step
        warmups = []
        for index in range(2):
            response = request_action(observations['rgb'],instruction,[],args.endpoint,manifest['model'],execution_feedback(None))
            verify_step(response)
            assert response['model_called']
            warmups.append(response)
            (args.output/'warmups.json').write_text(json.dumps(warmups,indent=2))
            print('WARMUP',index+1,response['action'],round(response['model_request_s'],3),flush=True)
        # Recreate iterator to ensure all fixed episodes start from pristine resets.
        env.close()
        env, _ = make_env(manifest['episodes'],max_steps=manifest['max_episode_steps'], dataset_path=manifest.get('dataset_path'), split=manifest.get('split','val_seen'))
        for _ in manifest['episodes']:
            observations = env.reset()
            episode = env.current_episode
            initial_metrics = env.get_metrics()
            directory = args.output/('episode_'+episode.episode_id)
            directory.mkdir(exist_ok=False)
            history = []
            rows = []
            positions = [pose(env)['position']]
            started = time.perf_counter()
            while not env.episode_over:
                image_path = directory/('frame_{:04d}.png'.format(len(rows)))
                imageio.imwrite(str(image_path),observations['rgb'])
                before = pose(env)
                feedback = execution_feedback(rows[-1] if rows else None)
                request_started = time.perf_counter()
                try:
                    result = request_action(observations['rgb'],episode.instruction.instruction_text,
                                            history,args.endpoint,manifest['model'],feedback)
                except Exception as error:
                    result = {'action':'STOP','raw':None,'error':type(error).__name__+': '+str(error),
                              'usage':None,'finish_reason':None,'model_request_s':time.perf_counter()-request_started,
                              'decision_s':time.perf_counter()-request_started}
                    result['execution_feedback_input'] = feedback
                action_name = result['action']
                action = next(k for k,v in ACTIONS.items() if v==action_name)
                sim_started = time.perf_counter()
                observations = env.step(action)
                sim_s = time.perf_counter()-sim_started
                after = pose(env)
                positions.append(after['position'])
                row = dict(result,step=len(rows),episode_id=episode.episode_id,
                           before=before,after=after,sim_step_s=sim_s,metrics=env.get_metrics(),
                           frame=str(image_path.relative_to(args.output)))
                rows.append(row)
                all_records.append(row)
                history.append(action_name)
                with (directory/'steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print('MODEL_STEP',episode.episode_id,len(rows),action_name,
                      'distance={:.3f}'.format(row['metrics']['distance_to_goal']),
                      'request_s={:.3f}'.format(result['model_request_s']),flush=True)
                (args.output/'progress.json').write_text(json.dumps({
                    'completed_episodes':len(episode_results),'current_episode':episode.episode_id,
                    'current_step':len(rows),'last_action':action_name,'metrics':row['metrics']}))
            metrics = env.get_metrics()
            length = sum(float(np.linalg.norm(np.asarray(b)-np.asarray(a))) for a,b in zip(positions,positions[1:]))
            shortest = float(initial_metrics['distance_to_goal'])
            expected_spl = float(metrics['success'])*shortest/max(shortest,length)
            assert abs(float(metrics['spl'])-expected_spl)<1e-5
            outcome = {'episode_id':episode.episode_id,'trajectory_id':episode.trajectory_id,
                       'steps':len(rows),'start_distance_m':shortest,'path_length_m':length,
                       'stop_called':history[-1]=='STOP','time_limit_reached':len(rows)>=manifest['max_episode_steps'],
                       'error_count':sum(bool(r['error']) for r in rows),'metrics':metrics,
                       'terminal_stop_source':('error_fallback' if rows[-1]['error'] else 'model') if history[-1]=='STOP' else 'step_limit',
                       'elapsed_s':time.perf_counter()-started}
            episode_results.append(outcome)
            (directory/'result.json').write_text(json.dumps(outcome,indent=2))
            imageio.imwrite(str(directory/'final.png'),observations['rgb'])
            print('MODEL_EPISODE_RESULT',json.dumps(outcome),flush=True)
    finally:
        if env is not None: env.close()
        sampler.terminate()
        sampler.wait(timeout=10)
        memory_stream.close()
    memory = []
    for line in (args.output/'gpu_samples.csv').read_text().splitlines():
        parts = line.split(',')
        if len(parts)>=3:
            try: memory.append(float(parts[1]))
            except ValueError: pass
    request_times = [r['model_request_s'] for r in all_records]
    decision_times = [r['decision_s'] for r in all_records]
    summary = {'scope':manifest['scope'],'model':manifest['model'],'status':'complete',
        'episodes':episode_results,'episode_count':len(episode_results),
        'sr':float(np.mean([r['metrics']['success'] for r in episode_results])),
        'spl':float(np.mean([r['metrics']['spl'] for r in episode_results])),
        'model_stop_success_count':sum(r['metrics']['success']==1 and r['terminal_stop_source']=='model' for r in episode_results),
        'fallback_stop_success_count':sum(r['metrics']['success']==1 and r['terminal_stop_source']=='error_fallback' for r in episode_results),
        'decision_count':len(all_records),'error_count':sum(bool(r['error']) for r in all_records),
        'warmup_requests_excluded':2,
        'model_request_latency_s':{'median':float(np.median(request_times)),'p95':float(np.percentile(request_times,95)),'mean':float(np.mean(request_times))},
        'decision_latency_s':{'median':float(np.median(decision_times)),'p95':float(np.percentile(decision_times,95))},
        'gpu_memory':{'scope':'whole card sampled memory used, includes desktop and model; not exact per-model allocator peak',
                      'interval_ms':200,'sample_count':len(memory),'sampled_peak_mib':max(memory) if memory else None},
        'renderer':'software Mesa llvmpipe; simulated movement time separately recorded'}
    assert len(episode_results)==len(manifest['episodes'])
    summary_path.write_text(json.dumps(summary,indent=2))
    print('NAVIGATION_DIAGNOSTIC_READY',json.dumps(summary),flush=True)


if __name__=='__main__': main()
