"""Official Habitat 0.1.7 task/metric wiring for fixed CE development episodes."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {0: 'STOP', 1: 'MOVE_FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT'}


def make_env(episode_ids=None, max_steps=500):
    import habitat
    # Load only the pinned official CE dataset module, avoiding training extras.
    path = ROOT / 'tmp/habitat_setup/sources/vlnce-task/task.py'
    spec = importlib.util.spec_from_file_location('project2_official_vlnce_task', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = habitat.get_config()
    config.defrost()
    config.SEED = 7
    config.ENVIRONMENT.MAX_EPISODE_STEPS = max_steps
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = -1
    config.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.SIMULATOR.TURN_ANGLE = 15
    config.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR']
    config.SIMULATOR.RGB_SENSOR.WIDTH = 224
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 224
    config.SIMULATOR.RGB_SENSOR.HFOV = 90
    config.SIMULATOR.RGB_SENSOR.POSITION = [0.0, 1.25, 0.0]
    config.TASK.TYPE = 'VLN-v0'
    config.TASK.SENSORS = ['INSTRUCTION_SENSOR']
    config.TASK.POSSIBLE_ACTIONS = list(ACTIONS.values())
    config.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SUCCESS', 'SPL', 'COLLISIONS']
    config.TASK.SUCCESS.SUCCESS_DISTANCE = 3.0
    config.TASK.SUCCESS_DISTANCE = 3.0
    config.DATASET.TYPE = 'VLN-CE-v1'
    config.DATASET.SPLIT = 'val_seen'
    config.DATASET.DATA_PATH = str(ROOT / 'data/datasets/R2R_VLNCE_v1-3_smoke/{split}/{split}.json.gz')
    config.DATASET.SCENES_DIR = str(ROOT / 'data/scene_datasets')
    config.DATASET.EPISODES_ALLOWED = episode_ids if episode_ids is not None else ['*']
    config.freeze()
    dataset = module.VLNCEDatasetV1(config.DATASET)
    if episode_ids is not None:
        assert {e.episode_id for e in dataset.episodes} == set(episode_ids)
        dataset.episodes.sort(key=lambda e: episode_ids.index(e.episode_id))
    env = habitat.Env(config=config, dataset=dataset)
    return env, config


def pose(env):
    from habitat_sim.utils.common import quat_to_coeffs
    state = env.sim.get_agent_state()
    return {'position': state.position.tolist(), 'rotation_xyzw': quat_to_coeffs(state.rotation).tolist()}
