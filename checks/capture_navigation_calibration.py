"""Render frozen TRAIN follower trajectories; export author-formatted input-only calibration."""
import base64
import hashlib
import json
from pathlib import Path

import imageio
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import simplevln_policy as policy
from simplevln_env import ROOT, make_env, pose

DEST = ROOT / 'data/calibration/navigation_awq_v1'
manifest_path = ROOT / 'data/baseline/navigation_awq_calibration_v1/manifest.json'
manifest = json.loads(manifest_path.read_text())
DEST.mkdir(parents=True, exist_ok=True)
assert not (DEST / 'samples.json').exists(), 'Calibration already captured; preserve frozen inputs'
env, config = make_env(manifest['episodes'], 400, manifest['dataset_path'], 'train')
(DEST / 'habitat_config.yaml').write_text(config.dump())
samples = []
traces = []
try:
    for _ in manifest['episodes']:
        obs = env.reset(); ep = env.current_episode
        directory = DEST / ('episode_' + ep.episode_id)
        directory.mkdir(exist_ok=False)
        follower = ShortestPathFollower(env.sim, goal_radius=0.3, return_one_hot=False, stop_on_error=False)
        targets = ep.reference_path[1:] or [ep.goals[0].position]
        target = 0; frames = []; rows = []
        while not env.episode_over:
            frame = directory / ('frame_%04d.png' % len(frames))
            imageio.imwrite(str(frame), obs['rgb']); frames.append(frame)
            action = int(follower.get_next_action(targets[target]))
            while action == 0 and target + 1 < len(targets):
                target += 1
                action = int(follower.get_next_action(targets[target]))
            before = pose(env)
            obs = env.step(action)
            rows.append(dict(action=action, before=before, after=pose(env), metrics=env.get_metrics()))
        assert env.get_metrics()['success'] == 1 and len(frames) >= 4, ep.episode_id
        indices = [round(i * (len(frames)-1) / 3) for i in range(4)]
        for index in indices:
            agent = policy.AuthorAgent('http://unused', 'calibration', str(DEST / 'agent_runtime'), 25, 15, 200, 1.0, False)
            agent.rgb_list = [policy.Image.open(str(f)).convert('RGB') for f in frames[max(0,index-199):index]]
            agent.predict_inference = lambda: 'stop'  # construct the original request, no model/labels needed
            agent.act({'rgb':imageio.imread(str(frames[index])), 'instruction':{'text':ep.instruction.instruction_text}}, {}, ep.episode_id)
            messages = agent.conversations
            image_index = 0; hashes = []
            for message in messages:
                for item in message['content']:
                    if item['type'] != 'image_url': continue
                    encoded = item['image_url']['url'].split(',',1)[1]
                    jpeg = base64.b64decode(encoded)
                    local = directory / ('sample_%04d_image_%02d.jpg' % (index,image_index))
                    local.write_bytes(jpeg)
                    item.clear(); item.update(type='image', image=local.relative_to(ROOT).as_posix())
                    hashes.append(hashlib.sha256(jpeg).hexdigest()); image_index += 1
            samples.append(dict(id='train_%s_step_%d' % (ep.episode_id,index), scene=ep.scene_id,
                episode_id=ep.episode_id, step=index, messages=messages, image_sha256=hashes))
        (directory / 'reference_trace.json').write_text(json.dumps(rows))
        traces.append(dict(episode_id=ep.episode_id, scene=ep.scene_id, steps=len(rows), sample_steps=indices,
                           metrics=env.get_metrics()))
        print('CALIBRATION_CAPTURE',ep.episode_id,len(rows),indices,flush=True)
finally:
    env.close()
assert len(samples) == 32
(DEST / 'samples.json').write_text(json.dumps(samples,indent=2))
(DEST / 'provenance.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    samples_sha256=hashlib.sha256((DEST/'samples.json').read_bytes()).hexdigest(), traces=traces,
    input_schema='Only system/user text and RGB images; no action targets, distance, poses or labels'),indent=2))
print('NAVIGATION_CALIBRATION_READY',len(samples),flush=True)
