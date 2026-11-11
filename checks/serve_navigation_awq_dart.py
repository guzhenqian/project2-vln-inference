"""Isolated D-group service. Default is offline check; --run requires C complete.

Never takes over a port, kills a process, or changes installed packages. Custom
mapping registration is repeated on spawn, as required by LMDeploy 0.11.1.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import uuid

from navigation_awq_dart_profiles import AWQ_SPEC, profile_config, served_model_name
from navigation_awq_dart_preflight import (ROOT, MODEL, CONFIG_SHA, HEADER_SHA,
                                          check_checkpoint, check_protected_sources,
                                          digest, require, require_c_finished)

MAP_FILE = ROOT / 'checks/navigation_awq_dart_module_map.py'
TARGET = 'navigation_awq_dart_runtime.AwqDartQwen3VLForConditionalGeneration'
SOURCE_FILES = (
    'serve_navigation_awq_dart.py', 'navigation_awq_dart_runtime.py',
    'navigation_awq_dart_module_map.py', 'navigation_awq_dart_profiles.py',
    'navigation_awq_dart_guard.py',
    'navigation_awq_dart_preflight.py', 'navigation_dart_profiles.py',
    'navigation_dart_selection.py', 'navigation_dart_kv_layout.py',
)


def register_project_mapping():
    from lmdeploy.pytorch.models.patch import update_custom_module_map
    update_custom_module_map(str(MAP_FILE))


register_project_mapping()


def spawned_registration_probe(queue):
    from lmdeploy.pytorch.models.patch import CUSTOM_MODULE_MAP
    queue.put(CUSTOM_MODULE_MAP.get('Qwen3VLForConditionalGeneration'))


def make_engine(profile):
    from lmdeploy import PytorchEngineConfig
    limits = profile_config(profile)
    return PytorchEngineConfig(
        dtype='float16', tp=1, dp=1, session_len=limits['session_len'], max_batch_size=1,
        max_prefill_token_num=limits['max_prompt_tokens'], cache_max_entry_count=0.35,
        num_gpu_blocks=64, block_size=64, eager_mode=True, enable_prefix_caching=False,
        enable_microbatch=False, enable_metrics=False, quant_policy=0,
        distributed_executor_backend='mp', enable_mp_engine=True, mp_engine_backend='mp',
    )


def check_spawn():
    import multiprocessing
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    child = context.Process(target=spawned_registration_probe, args=(queue,))
    child.start()
    try:
        require(queue.get(timeout=45) == TARGET, 'Spawned child lost D-group registration')
        child.join(timeout=10)
        require(child.exitcode == 0, 'Registration probe failed to exit')
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        queue.close()


@contextmanager
def exclusive_service_lock():
    path = ROOT / 'logs/navigation_awq_dart_service.active.lock'
    token = uuid.uuid4().hex
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'token': token, 'pid': os.getpid(), 'scope': 'AWQ+DART service'}, stream)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('token') == token:
            path.unlink()


def available_gpu_memory():
    require_c_finished()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        require(probe.connect_ex(('127.0.0.1', 23333)) != 0,
                'Port 23333 is occupied; coordinate release with the current service owner')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True)
    free = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip()]
    require(len(free) == 1 and free[0] >= 10000,
            'Expected released single GPU with at least 10000 MiB free; no service started')
    return free[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-ratio', type=float, choices=(1.0, 0.75), default=1.0)
    parser.add_argument('--profile', choices=('short', 'navigation'), default='navigation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--run', action='store_true', help='Explicitly load the service only after C has completed')
    parser.add_argument('--check-spawn', action='store_true')
    args = parser.parse_args()
    if args.run and args.check_spawn:
        parser.error('--check-spawn is for offline checking only')
    expected_versions = {'lmdeploy': '0.11.1', 'torch': '2.5.1+cu121', 'transformers': '4.57.6'}
    versions = {name: version(name) for name in expected_versions}
    require(versions == expected_versions, 'Verified dependency versions changed: ' + str(versions))
    source_checks = check_protected_sources()
    checkpoint = check_checkpoint()
    from lmdeploy.pytorch.models.patch import CUSTOM_MODULE_MAP
    require(CUSTOM_MODULE_MAP.get('Qwen3VLForConditionalGeneration') == TARGET,
            'D-group adapter registration failed')
    engine = make_engine(args.profile)
    limits = profile_config(args.profile)
    require(engine.num_gpu_blocks == 64 and engine.block_size == 64
            and engine.session_len == 4096 and limits['max_new_tokens'] == 512,
            'D-group cache/generation capacity differs from intended C-group control')
    source_hashes = {name: digest(ROOT / 'checks' / name) for name in SOURCE_FILES}
    print('AWQ_DART_SERVICE_CONFIG_READY', served_model_name(args.keep_ratio, args.profile), flush=True)
    print('W4A16 language projections; FP16 vision/KV; eager; fixed 64 x 64 KV blocks', flush=True)
    if not args.run:
        if args.check_spawn:
            check_spawn()
            print('AWQ_DART_SPAWN_REGISTRATION_OK', flush=True)
        print('CHECK_ONLY: no weights loaded, CUDA initialization, HTTP request or service start', flush=True)
        return

    require_c_finished()
    with exclusive_service_lock():
        run_service(args, engine, limits, versions, source_checks, source_hashes)


def run_service(args, engine, limits, versions, source_checks, source_hashes):
    c_state = require_c_finished()
    # A successful C controller deliberately leaves its service running. Require
    # explicit resource release by its owner; do not stop it from this launcher.
    available_gpu_memory()
    checkpoint = check_checkpoint(full_weights=True)
    free_mib = available_gpu_memory()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'logs' / ('navigation_awq_dart_service_' + stamp)
    output.mkdir(exist_ok=False)
    os.environ['PROJECT2_AWQ_DART_KEEP_RATIO'] = str(args.keep_ratio)
    os.environ['PROJECT2_AWQ_DART_PROFILE'] = args.profile
    os.environ['PROJECT2_AWQ_DART_RUN_DIR'] = str(output)
    existing = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = str(ROOT / 'checks') + (os.pathsep + existing if existing else '')
    launch = {
        'scope': 'AWQ+DART integration candidate; no completed D150 result',
        'profile': args.profile, 'profile_limits': limits, 'full_prefill_required': True,
        'full_prefill_gpu_memory_verified': False,
        'model_source': MODEL.relative_to(ROOT).as_posix(),
        'model_name': served_model_name(args.keep_ratio, args.profile),
        'keep_ratio': args.keep_ratio, 'weight_quantization': AWQ_SPEC,
        'model_config_sha256': CONFIG_SHA, 'weight_header_sha256': HEADER_SHA,
        'checkpoint_verification': checkpoint,
        'engine_config': asdict(engine), 'versions': versions,
        'cache_policy': {'mode': 'fixed_gpu_blocks', 'num_gpu_blocks_configured': 64,
                         'block_size': 64, 'same_for_keep100_and_keep75': True,
                         'pruning_releases_allocated_blocks': False,
                         'scheduler_uses_original_sequence_lengths': True},
        'sources_sha256': source_hashes, 'c_locked_sources_sha256': source_checks,
        'c_completed_status_snapshot': c_state,
        'server': 'http://127.0.0.1:23333', 'gpu_free_mib_before': free_mib,
        'console_log': os.environ.get('PROJECT2_AWQ_DART_CONSOLE_LOG'),
        'limitations': ['GPU combination, full-prefill memory, generation and navigation remain pending at launch.',
                       'Keep100 is AWQ with native forwards, not a token-pruned D-group result.',
                       'Old A/B use graph execution and 1024 prefill; this uses eager/full-prefill3072.',
                       'Allocated KV pool stays fixed; effective token reduction is not allocator savings.'],
    }
    with (output / 'launch.json').open('x', encoding='utf-8') as stream:
        json.dump(launch, stream, ensure_ascii=False, indent=2,
                  default=lambda value: getattr(value, 'value', str(value)))
    print('Run directory:', output.relative_to(ROOT).as_posix(), flush=True)
    from lmdeploy.messages import VisionConfig
    from lmdeploy.serve.openai.api_server import serve
    serve(str(MODEL), model_name=served_model_name(args.keep_ratio, args.profile), backend='pytorch',
          backend_config=engine, vision_config=VisionConfig(1), server_name='127.0.0.1',
          server_port=23333, max_concurrent_requests=1, log_level='INFO', max_log_len=256)


if __name__ == '__main__':
    main()
