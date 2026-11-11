"""Explicit user-run, isolated FP16/DART diagnostic service; no package edits.

This file registers the project mapping at module top level because both
LMDeploy multiprocessing layers use spawn and re-import this main module.
The actual service is started only inside the main guard.
"""

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from navigation_dart50_profiles import profile_config, served_model_name


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
MAP_FILE = ROOT / 'checks/navigation_dart50_module_map.py'
EXPECTED_CONFIG_SHA = '8c129761080666a7970e3bf3612c88e5ec64936a1fa8e92c821acc4e32c6c061'


def register_project_mapping():
    from lmdeploy.pytorch.models.patch import update_custom_module_map
    update_custom_module_map(str(MAP_FILE))


# Intentionally outside main: required in both spawned LMDeploy workers.
register_project_mapping()


def model_name(keep_ratio, profile='short'):
    return served_model_name(keep_ratio, profile)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spawned_registration_probe(queue):
    from lmdeploy.pytorch.models.patch import CUSTOM_MODULE_MAP
    queue.put(CUSTOM_MODULE_MAP.get('Qwen3VLForConditionalGeneration'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-ratio', type=float, choices=(1.0, 0.5), default=0.5)
    parser.add_argument('--profile', choices=('short', 'navigation'), default='navigation',
                        help='short: original two-image lesson; navigation: experimental full prefill for 2-9 images')
    parser.add_argument('--check-only', action='store_true', help='Verify source/config/registration; do not load a model or start a service')
    parser.add_argument('--check-spawn', action='store_true', help='With --check-only, also verify mapping registration in a fresh spawned Python process')
    parser.add_argument('--run', action='store_true', help='Explicit service start after completed C75')
    parser.add_argument('--c75-report', type=Path)
    args = parser.parse_args()
    if args.run and args.check_only:
        parser.error('--run and --check-only conflict')
    args.check_only = not args.run
    if args.check_spawn and not args.check_only:
        parser.error('--check-spawn is only allowed with --check-only')
    limits = profile_config(args.profile)
    if limits['max_prompt_tokens'] + limits['max_new_tokens'] > limits['session_len']:
        raise ValueError('Profile does not reserve the configured generation budget')
    required_versions = {'lmdeploy': '0.11.1', 'torch': '2.5.1+cu121', 'transformers': '4.57.6'}
    installed = {name: version(name) for name in required_versions}
    if installed != required_versions:
        raise RuntimeError('This diagnostic adapter is pinned to the verified environment: ' + str(installed))
    config_path = MODEL / 'config.json'
    if digest(config_path) != EXPECTED_CONFIG_SHA:
        raise ValueError('The unquantized source configuration has changed')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for section in (config, config['text_config'], config['vision_config']):
        if section.get('quantization_config') is not None:
            raise ValueError('Only an unquantized source checkpoint is allowed')
    with (MODEL / 'model.safetensors').open('rb') as stream:
        size_bytes = stream.read(8)
        header_size = struct.unpack('<Q', size_bytes)[0]
        if header_size > 32 * 1024 * 1024:
            raise ValueError('Unexpected weight header size')
        header_bytes = stream.read(header_size)
    header = json.loads(header_bytes)
    if any(name.endswith(('.qweight', '.qzeros', '.weight_packed')) for name in header):
        raise ValueError('Quantized weights are forbidden for the DART-only service')
    from lmdeploy.pytorch.models.patch import CUSTOM_MODULE_MAP
    target = CUSTOM_MODULE_MAP.get('Qwen3VLForConditionalGeneration')
    if target != 'navigation_dart50_runtime.Dart50Qwen3VLForConditionalGeneration':
        raise RuntimeError('Custom model registration did not select the project adapter')
    from lmdeploy import PytorchEngineConfig
    engine = PytorchEngineConfig(
        dtype='float16', tp=1, dp=1, session_len=limits['session_len'], max_batch_size=1,
        max_prefill_token_num=limits['max_prompt_tokens'], cache_max_entry_count=0.35,
        num_gpu_blocks=64 if args.profile == 'navigation' else 0,
        eager_mode=True, enable_prefix_caching=False, enable_microbatch=False,
        enable_metrics=False, quant_policy=0, distributed_executor_backend='mp',
        enable_mp_engine=True, mp_engine_backend='mp',
    )
    if args.profile == 'navigation':
        if (engine.max_prefill_token_num != 3072 or engine.num_gpu_blocks != 64
                or engine.block_size != 64 or engine.session_len != 4096
                or limits['max_new_tokens'] != 512):
            raise ValueError('Navigation profile must use 3072-token full prefill, 64 GPU blocks of 64, and 512 reserved generation tokens')
    elif engine.num_gpu_blocks != 0 or engine.max_prefill_token_num != 1024:
        raise ValueError('Short profile must retain automatic GPU cache allocation and 1024-token prefill')
    cache_policy = {
        'mode': 'fixed_gpu_blocks' if args.profile == 'navigation' else 'automatic_fraction',
        'num_gpu_blocks_configured': engine.num_gpu_blocks,
        'block_size': engine.block_size,
        'token_slot_capacity_configured': engine.num_gpu_blocks * engine.block_size if engine.num_gpu_blocks else None,
        'cache_max_entry_count_used_when_automatic': engine.cache_max_entry_count,
        'same_for_keep100_and_keep50': True,
        'scheduler_uses_original_sequence_lengths': True,
        'pruning_releases_allocated_blocks': False,
        'note': ('Fixed 64 blocks preserve the observed short-service allocation while increasing full prefill; '
                 'new model-load and full-prefill GPU memory validation remain pending'
                 if args.profile == 'navigation' else
                 'Original short-profile automatic cache sizing remains unchanged'),
    }
    print('DART_SERVICE_CONFIG_READY', flush=True)
    print('Source:', MODEL.relative_to(ROOT).as_posix(), flush=True)
    print('Profile:', args.profile, flush=True)
    print('Served model:', model_name(args.keep_ratio, args.profile), flush=True)
    print('FP16 weights/activations; no AWQ; unquantized KV; single request; eager mode', flush=True)
    if args.profile == 'short':
        print('Diagnostic input limit: 1024 tokens; nine-image/chunked input is not supported yet', flush=True)
        print('KV cache: original automatic allocation (num_gpu_blocks=0, cache fraction=0.35)', flush=True)
    else:
        print(f'Experimental navigation profile: {limits["min_images"]}-{limits["max_images"]} images; '
              f'one complete prefill up to {limits["max_prompt_tokens"]} tokens; '
              f'{limits["max_new_tokens"]} generation tokens reserved', flush=True)
        print('KV cache: fixed 64 blocks x 64 tokens = 4096 slots for both keep100 and keep50; '
              'scheduler allocation retains original sequence lengths', flush=True)
        print('Fixed cache size is configured, not a new GPU memory measurement; model-load validation remains pending', flush=True)
        print('Full-prefill GPU memory capacity remains unverified; no formal 150-task result is claimed', flush=True)
    if args.check_only:
        if args.check_spawn:
            import multiprocessing
            spawn_context = multiprocessing.get_context('spawn')
            queue = spawn_context.Queue()
            child = spawn_context.Process(target=spawned_registration_probe, args=(queue,))
            child.start()
            try:
                child_target = queue.get(timeout=45)
                child.join(timeout=10)
                if child.exitcode != 0 or child_target != target:
                    raise RuntimeError('Custom model registration did not survive process spawn')
            finally:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=5)
                queue.close()
            print('DART_SPAWN_REGISTRATION_OK', flush=True)
        print('CHECK_ONLY: no model weights loaded, no service started, no GPU calls', flush=True)
        return

    from navigation_dart50_support import require_c75_complete
    require_c75_complete(args.c75_report)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', 23333)) == 0:
            raise SystemExit('Port 23333 is already in use. Stop the old model in its service terminal, wait for the shell prompt, then retry. No process was stopped by this script.')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True)
    free_values = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip()]
    if len(free_values) != 1 or free_values[0] < 10000:
        raise SystemExit('This FP16 single-GPU check requires at least 10000 MiB free on the one visible GPU. No service was started.')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'logs' / ('navigation_dart50_service_' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    os.environ['PROJECT2_DART50_KEEP_RATIO'] = str(args.keep_ratio)
    os.environ['PROJECT2_DART50_PROFILE'] = args.profile
    os.environ['PROJECT2_DART50_RUN_DIR'] = str(output)
    # Keep imports available to both spawn layers, without changing the installed environment.
    checks_path = str(ROOT / 'checks')
    existing = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = checks_path + (os.pathsep + existing if existing else '')
    manifest = {
        'scope': ('Short-input FP16/DART service integration diagnostic, not a navigation benchmark'
                  if args.profile == 'short' else
                  'Experimental multi-image FP16/DART full-prefill service diagnostic, not a completed navigation benchmark'),
        'profile': args.profile, 'profile_limits': limits,
        'full_prefill_required': True,
        'full_prefill_gpu_memory_verified': False if args.profile == 'navigation' else None,
        'model_source': MODEL.relative_to(ROOT).as_posix(),
        'model_name': model_name(args.keep_ratio, args.profile),
        'keep_ratio': args.keep_ratio, 'weight_quantization': None,
        'model_config_sha256': EXPECTED_CONFIG_SHA,
        'weight_header_sha256': hashlib.sha256(size_bytes + header_bytes).hexdigest(),
        'weight_header_note': 'Header only; full source weights were validated during original download',
        'engine_config': asdict(engine), 'versions': installed,
        'cache_policy': cache_policy,
        'server': 'http://127.0.0.1:23333', 'gpu_free_mib_before': free_values[0],
        'console_log': os.environ.get('PROJECT2_DART50_CONSOLE_LOG'),
        'sources_sha256': {path.name: digest(path) for path in (
            Path(__file__), MAP_FILE, ROOT / 'checks/navigation_dart50_runtime.py',
            ROOT / 'checks/navigation_dart50_support.py',
            ROOT / 'checks/navigation_dart_selection.py', ROOT / 'checks/navigation_dart_kv_layout.py',
            ROOT / 'checks/navigation_dart50_profiles.py')},
        'limitations': [('Two-image/short-input diagnostic only; input above 1024 tokens rejected'
                         if args.profile == 'short' else
                         f'Experimental 2-9-image profile; one complete prefill up to {limits["max_prompt_tokens"]} tokens; GPU memory capacity is not yet verified'),
                       'Split-prefill DART selection is not supported; the complete prompt must fit one prefill',
                       'Eager mode differs from the previous baseline CUDA-graph runtime; do not claim a matched latency speedup',
                       'Only independent single-turn requests; no prefix cache or history continuation',
                       'Cache allocation remains based on original scheduler length'],
    }
    with (output / 'launch.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, default=lambda value: getattr(value, 'value', str(value)))
    print('Run directory:', output.relative_to(ROOT).as_posix(), flush=True)
    print('Starting diagnostic model service; leave this terminal open while it runs.', flush=True)
    from lmdeploy.messages import VisionConfig
    from lmdeploy.serve.openai.api_server import serve
    serve(str(MODEL), model_name=model_name(args.keep_ratio, args.profile), backend='pytorch',
          backend_config=engine, vision_config=VisionConfig(1), server_name='127.0.0.1',
          server_port=23333, max_concurrent_requests=1, log_level='INFO', max_log_len=256)


if __name__ == '__main__':
    main()
