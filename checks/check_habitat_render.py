"""Habitat-Sim 0.1.7 public-scene smoke test, compatible with Python 3.6.

API/source reference:
https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/examples/settings.py
https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#testing
"""
import argparse
import ctypes as ct
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def check_actions(sim, agent, output, initial_rgb, np, imageio):
    """Check fixed action plumbing; this does not evaluate a navigation task."""
    actions = ['turn_left', 'move_forward', 'turn_right']
    result = {'status': 'running', 'scope': 'three fixed action API checks; no SR/SPL',
              'forward_amount_m': 0.25, 'turn_amount_deg': 30.0, 'steps': []}
    previous_rgb = initial_rgb
    for index, action in enumerate(actions, 1):
        before = agent.get_state()
        position_before = np.asarray(before.position).copy()
        rotation_before = np.array([before.rotation.w, before.rotation.x,
                                    before.rotation.y, before.rotation.z])
        observation = sim.step(action)
        after = agent.get_state()
        position_after = np.asarray(after.position).copy()
        rotation_after = np.array([after.rotation.w, after.rotation.x,
                                   after.rotation.y, after.rotation.z])
        rotation_dot = abs(float(np.dot(rotation_before, rotation_after)))
        angle = float(np.degrees(2.0 * np.arccos(np.clip(rotation_dot, 0.0, 1.0))))
        rgb = observation['rgb'][:, :, :3]
        image_name = 'action_{:02d}_{}.png'.format(index, action)
        imageio.imwrite(str(output / image_name), rgb)
        displacement = float(np.linalg.norm(position_after - position_before))
        record = {'action': action, 'position_before': position_before.tolist(),
                  'position_after': position_after.tolist(), 'displacement_m': displacement,
                  'rotation_change_deg': angle, 'image': image_name,
                  'image_changed': bool(np.any(rgb != previous_rgb))}
        result['steps'].append(record)
        # Write evidence before asserting, so partial failures remain inspectable.
        (output / 'action_result.json').write_text(json.dumps(result, indent=2))
        print('Action result:', json.dumps(record), flush=True)
        if not np.isfinite(position_after).all() or not np.isfinite(angle):
            raise RuntimeError('Non-finite pose after ' + action)
        if action.startswith('turn_'):
            if not 29.0 <= angle <= 31.0 or displacement > 0.001:
                raise RuntimeError('Turn did not match configured 30-degree rotation')
            if not record['image_changed']:
                raise RuntimeError('Image did not update after turn')
        previous_rgb = rgb
    result['status'] = 'action_api_checks_passed_pending_visual_review'
    result['forward_translation_observed'] = result['steps'][1]['displacement_m'] > 0.001
    (output / 'action_result.json').write_text(json.dumps(result, indent=2))
    print('HABITAT_ACTION_SMOKE_READY (fixed action API only; no navigation success metric)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--archive', type=Path)
    sources.add_argument('--scene', type=Path)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-mode', choices=['cuda', 'software'], default='cuda')
    parser.add_argument('--action-smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episode = json.loads(args.episode.read_text()) if args.episode else None
    if args.scene:
        scene = args.scene
        if episode and Path(episode['scene_id']).name != scene.name:
            raise ValueError('Episode scene does not match supplied GLB')
    else:
        if episode:
            raise ValueError('--episode requires --scene')
        scene = args.output / 'skokloster-castle.glb'
        with zipfile.ZipFile(str(args.archive)) as archive:
            # Copy only known basenames, never extract archive-controlled paths.
            for name in ('skokloster-castle.glb', 'skokloster-castle.navmesh'):
                matches = [n for n in archive.namelist() if Path(n).name == name]
                if len(matches) != 1:
                    raise RuntimeError('Expected one {} in official archive: {}'.format(name, matches))
                with archive.open(matches[0]) as src, (args.output / name).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    with scene.open('rb') as stream:
        if stream.read(4) != b'glTF':
            raise RuntimeError('Test scene is not a GLB file')

    import habitat_sim
    import habitat_sim.agent
    import imageio
    import numpy as np
    from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_coeffs

    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(scene.resolve())
    # Magnum's unsigned CUDA-device sentinel 0xffffffff disables CUDA matching.
    # Habitat exposes an int; -1 converts to that sentinel in setCudaDevice().
    # This is an opt-in software compatibility test, not a CUDA success.
    config.gpu_device_id = -1 if args.render_mode == 'software' else 0
    config.enable_physics = False
    sensor = habitat_sim.SensorSpec()
    sensor.uuid = 'rgb'
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor.resolution = [480, 640]
    sensor.position = [0.0, 1.5, 0.0]
    sensor.gpu2gpu_transfer = False
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor]
    if args.action_smoke:
        agent_config.action_space = {
            'move_forward': habitat_sim.agent.ActionSpec(
                'move_forward', habitat_sim.agent.ActuationSpec(amount=0.25)),
            'turn_left': habitat_sim.agent.ActionSpec(
                'turn_left', habitat_sim.agent.ActuationSpec(amount=30.0)),
            'turn_right': habitat_sim.agent.ActionSpec(
                'turn_right', habitat_sim.agent.ActuationSpec(amount=30.0)),
        }
    print('Creating simulator / EGL context...', flush=True)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent_config]))
    try:
        egl = ct.CDLL('libEGL.so.1')
        egl.eglGetProcAddress.argtypes = [ct.c_char_p]
        egl.eglGetProcAddress.restype = ct.c_void_p
        address = egl.eglGetProcAddress(b'glGetString')
        if not address:
            raise RuntimeError('Cannot inspect actual GL renderer')
        gl_get_string = ct.CFUNCTYPE(ct.c_char_p, ct.c_uint)(address)
        gl_info = {}
        for name, token in [('vendor', 0x1F00), ('renderer', 0x1F01), ('version', 0x1F02)]:
            value = gl_get_string(token)
            gl_info[name] = value.decode('utf-8', errors='replace') if value else None
        print('Actual OpenGL:', json.dumps(gl_info), flush=True)
        sim.seed(7)
        if not sim.pathfinder.is_loaded:
            raise RuntimeError('Public scene navmesh did not load')
        agent = sim.initialize_agent(0)
        state = habitat_sim.AgentState()
        if episode:
            state.position = np.asarray(episode['start_position'], dtype=np.float32)
            state.rotation = quat_from_coeffs(episode['start_rotation'])
            if not sim.pathfinder.is_navigable(state.position):
                raise RuntimeError('Official episode start is not navigable with this scene/navmesh')
        else:
            state.position = sim.pathfinder.get_random_navigable_point()
            state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
        if not np.isfinite(state.position).all():
            raise RuntimeError('Scene has no finite navigable start point')
        agent.set_state(state)
        actual_state = agent.get_state()
        if not np.allclose(actual_state.position, state.position, atol=1e-5, rtol=0):
            raise RuntimeError('Actual agent start differs from requested position')
        requested_rotation = quat_to_coeffs(state.rotation)
        actual_rotation = quat_to_coeffs(actual_state.rotation)
        if not (np.allclose(actual_rotation, requested_rotation, atol=1e-5, rtol=0)
                or np.allclose(actual_rotation, -requested_rotation, atol=1e-5, rtol=0)):
            raise RuntimeError('Actual agent heading differs from requested rotation')
        rgba = sim.get_sensor_observations()['rgb']
        if rgba.shape != (480, 640, 4) or rgba.dtype != np.uint8:
            raise RuntimeError('Unexpected RGB observation: {} {}'.format(rgba.shape, rgba.dtype))
        rgb = rgba[:, :, :3]
        image_path = args.output / 'rgb.png'
        imageio.imwrite(str(image_path), rgb)
        if not np.array_equal(imageio.imread(str(image_path)), rgb):
            raise RuntimeError('Saved PNG differs from rendered RGB')
        report = {
            'status': 'frame_saved_pending_visual_review',
            'scope': 'one scene RGB preview; no navigation metrics or model',
            'pose_source': 'official CE episode' if episode else 'seeded random navigable point',
            'episode_id': episode['episode_id'] if episode else None,
            'trajectory_id': episode.get('trajectory_id') if episode else None,
            'instruction': episode['instruction']['instruction_text'] if episode else None,
            'preview_sensor_profile': 'custom 640x480 RGB, height 1.5m; not an official evaluation config',
            'seed': 7,
            'requested_render_mode': args.render_mode,
            'cuda_device_selection': config.gpu_device_id,
            'opengl': gl_info,
            'position': np.asarray(actual_state.position).tolist(),
            'rotation_xyzw': quat_to_coeffs(actual_state.rotation).tolist(),
            'sensor_height_m': 1.5,
            'resolution_hw': [480, 640],
            'scene_sha256': hashlib.sha256(scene.read_bytes()).hexdigest(),
            'png_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'pixel_min': int(rgb.min()),
            'pixel_max': int(rgb.max()),
            'pixel_std': float(rgb.std()),
            'numpy': np.__version__,
            'imageio': imageio.__version__,
        }
        with (args.output / 'render_result.json').open('w') as stream:
            json.dump(report, stream, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        if report['pixel_std'] < 1.0:
            raise RuntimeError('Nearly uniform image; inspect the PNG and renderer logs')
        if args.action_smoke:
            check_actions(sim, agent, args.output, rgb, np, imageio)
    finally:
        sim.close()
    print('HABITAT_RENDER_FRAME_READY (PNG saved; visual review pending)', flush=True)


if __name__ == '__main__':
    main()
