"""Validate official SR/SPL with STOP-negative and reference-follower controls."""
import argparse
import json
import time
from pathlib import Path

import imageio
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from closed_loop_env import ROOT, make_env, pose

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--episodes', nargs='+', default=['188'])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
results = []
for controller in ['stop_negative', 'reference_follower']:
    env, config = make_env(args.episodes)
    (args.output / 'habitat_config.yaml').write_text(config.dump())
    try:
        for _ in args.episodes:
            observations = env.reset()
            episode = env.current_episode
            start = pose(env)
            initial_metrics = env.get_metrics()
            imageio.imwrite(str(args.output / ('{}_{}_start.png'.format(controller, episode.episode_id))), observations['rgb'])
            rows = []
            follower = ShortestPathFollower(env.sim, goal_radius=0.3, return_one_hot=False, stop_on_error=False)
            targets = episode.reference_path[1:] or [episode.goals[0].position]
            target_index = 0
            started = time.perf_counter()
            while not env.episode_over:
                if controller == 'stop_negative':
                    action = 0
                else:
                    action = int(follower.get_next_action(targets[target_index]))
                    while action == 0 and target_index + 1 < len(targets):
                        target_index += 1
                        action = int(follower.get_next_action(targets[target_index]))
                before = pose(env)
                observations = env.step(action)
                rows.append({'action': action, 'before': before, 'after': pose(env), 'metrics': env.get_metrics()})
            metrics = env.get_metrics()
            positions = [start['position']] + [r['after']['position'] for r in rows]
            length = sum(float(np.linalg.norm(np.asarray(b)-np.asarray(a))) for a,b in zip(positions, positions[1:]))
            shortest = float(initial_metrics['distance_to_goal'])
            expected_spl = float(metrics['success']) * shortest / max(shortest, length)
            assert abs(float(metrics['spl']) - expected_spl) < 1e-5
            record = {'controller': controller, 'episode_id': episode.episode_id,
                      'trajectory_id': episode.trajectory_id, 'steps': len(rows),
                      'start_distance_m': shortest, 'actual_path_length_m': length,
                      'metrics': metrics, 'independent_spl_check': expected_spl,
                      'elapsed_s': time.perf_counter()-started, 'rows': rows}
            (args.output / ('{}_{}.json'.format(controller, episode.episode_id))).write_text(json.dumps(record, indent=2))
            imageio.imwrite(str(args.output / ('{}_{}_final.png'.format(controller, episode.episode_id))), observations['rgb'])
            results.append({k:v for k,v in record.items() if k!='rows'})
            print('CONTROL_RESULT', json.dumps(results[-1]), flush=True)
            if controller == 'stop_negative':
                assert shortest > 3.0 and metrics['success'] == 0 and metrics['spl'] == 0
            else:
                assert metrics['success'] == 1 and metrics['spl'] > 0
    finally:
        env.close()
(args.output / 'summary.json').write_text(json.dumps({'scope':'controller verification, not model baseline','results':results},indent=2))
print('CLOSED_LOOP_METRICS_CONTROLS_READY', flush=True)
