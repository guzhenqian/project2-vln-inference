"""Independent down-only GEMV-over-wide service; original latency entry stays frozen."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path

from navigation_stage2_common import ROOT, SOURCES, read, write, digest
from serve_navigation_repair import recipe
from lmdeploy.pytorch.models.patch import update_custom_module_map

MODULE_MAP = 'navigation_latency_w4_hybrid_module_map_20261120.py'
update_custom_module_map(str(ROOT / 'checks' / MODULE_MAP))

# Import in frontend AND spawned workers, before engine construction.
if os.environ.get('PROJECT2_LATENCY_INPROCESS') == '1':
    import navigation_latency_inprocess_20261120
    navigation_latency_inprocess_20261120.install()
if os.environ.get('PROJECT2_LATENCY_LINEAR_IPC') == '1':
    import navigation_latency_ipc_20261120
    navigation_latency_ipc_20261120.install()
if os.environ.get('PROJECT2_LATENCY_PROFILE') == '1':
    import navigation_latency_frontend_20261120
    navigation_latency_frontend_20261120.install()
if os.environ.get('PROJECT2_LATENCY_SCHEDULE_TRACE') == '1':
    import navigation_latency_schedule_probe_20261120
    navigation_latency_schedule_probe_20261120.install()
if os.environ.get('PROJECT2_LATENCY_VISION_QMAX') == '1':
    import navigation_latency_vision_qmax_20261120
    navigation_latency_vision_qmax_20261120.install()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=('fp16', 'awq_w4a16', 'gptq_w4a16'))
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    config = recipe(args.group)
    assert config['weight_bits'] == 4 and os.environ.get('PROJECT2_LATENCY_HYBRID_TABLE')
    assert not os.environ.get('PROJECT2_LATENCY_W4_TABLE') and not os.environ.get('PROJECT2_LATENCY_W4_PHASE')
    run = args.run_dir.resolve()
    assert run.is_relative_to(ROOT / 'logs') and run.is_dir()
    executor = os.environ.get('PROJECT2_LATENCY_EXECUTOR', 'mp')
    interval = int(os.environ.get('PROJECT2_LATENCY_PREFILL_INTERVAL', '16'))
    assert executor in ('mp', 'uni') and interval in (1, 2, 4, 8, 16)
    assert (os.environ.get('PROJECT2_LATENCY_VISION_QMAX_VALIDATE') != '1'
            or os.environ.get('PROJECT2_LATENCY_VISION_QMAX') == '1')
    versions = {name: version(name) for name in ('lmdeploy', 'torch', 'transformers', 'triton')}
    assert versions == dict(lmdeploy='0.11.1', torch='2.5.1+cu121', transformers='4.57.6', triton='3.1.0')
    os.environ.update(PROJECT2_STAGE2_METHOD=config['method'], PROJECT2_STAGE2_KEEP_RATIO='1.0',
        PROJECT2_STAGE2_PROFILE='navigation', PROJECT2_STAGE2_RUN_DIR=str(run),
        PROJECT2_W4A16_FIX_VARIANT='stock', PROJECT2_W4A16_FIX_GRAPH='on',
        PROJECT2_REPAIR_WBITS=str(config['weight_bits']), PROJECT2_REPAIR_ABITS='16',
        PROJECT2_PRECISION_WBITS=str(config['weight_bits']), PROJECT2_PRECISION_ABITS='16')
    os.environ['PYTHONPATH'] = str(ROOT / 'checks') + os.pathsep + os.environ.get('PYTHONPATH', '')
    from lmdeploy import PytorchEngineConfig
    from lmdeploy.messages import VisionConfig
    engine = PytorchEngineConfig(dtype='float16', tp=1, dp=1, session_len=4096,
        custom_module_map=str(ROOT / 'checks' / MODULE_MAP),
        max_batch_size=1, max_prefill_token_num=3072, cache_max_entry_count=.35,
        num_gpu_blocks=64, eager_mode=False, enable_prefix_caching=False,
        enable_microbatch=False, enable_metrics=False, quant_policy=0,
        prefill_interval=interval, distributed_executor_backend=executor,
        enable_mp_engine=True, mp_engine_backend='mp')
    names = SOURCES + ['navigation_w4a16_fix_runtime.py', 'navigation_repair_runtime.py',
        'navigation_repair_module_map.py', 'serve_navigation_repair.py',
        'serve_navigation_latency_hybrid_20261120.py']
    if config['weight_bits'] == 4:
        names += ['navigation_w4a16_candidate_kernels.py', 'navigation_w4a16_repair_kernels.py']
    if os.environ.get('PROJECT2_LATENCY_PROFILE') == '1':
        names.append('navigation_latency_frontend_20261120.py')
    if os.environ.get('PROJECT2_LATENCY_LINEAR_IPC') == '1':
        names.append('navigation_latency_ipc_20261120.py')
    if os.environ.get('PROJECT2_LATENCY_INPROCESS') == '1':
        names.append('navigation_latency_inprocess_20261120.py')
    if os.environ.get('PROJECT2_LATENCY_SCHEDULE_TRACE') == '1':
        names.append('navigation_latency_schedule_probe_20261120.py')
    if os.environ.get('PROJECT2_LATENCY_VISION_QMAX') == '1':
        names.append('navigation_latency_vision_qmax_20261120.py')
    names += ['navigation_latency_w4_kernels_20261120.py',
              'navigation_latency_w4_gemv_20261120.py',
              'navigation_latency_w4_hybrid_adapter_20261120.py',
              'navigation_latency_w4_hybrid_contract_20261120.py', MODULE_MAP]
    model = ROOT / config['model']
    weight_sha = digest(model / 'model.safetensors')
    expected = ('b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b'
        if args.group == 'fp16' else read(model / 'conversion_audit.json')['sha256'])
    assert weight_sha == expected
    from navigation_latency_w4_hybrid_contract_20261120 import load_hybrid_recipe
    hybrid_path = Path(os.environ['PROJECT2_LATENCY_HYBRID_TABLE']).resolve()
    binding = load_hybrid_recipe(hybrid_path,config['method'],weight_sha)
    baseline = ROOT / binding['recipe']['baseline_table']['path']
    assert digest(baseline) == binding['recipe']['baseline_table']['sha256']
    supplied = os.environ.get('PROJECT2_W4A16_REPAIR_TABLE')
    assert supplied and digest(Path(supplied)) == binding['recipe']['baseline_table']['sha256'], 'Specify the exact frozen fallback table'
    write(run / 'launch.json', dict(group=args.group, recipe=config, versions=versions,
        model_name='navigation-repair-' + args.group, model_weights_sha256=weight_sha,
        model_config_sha256=digest(model / 'config.json'),
        hybrid_recipe_path=str(hybrid_path),hybrid_recipe_sha256=digest(hybrid_path),
        engine=json.loads(json.dumps(asdict(engine), default=lambda x: x.value)),
        sources_sha256={name: digest(ROOT / 'checks' / name) for name in names},
        candidate_environment={k: v for k, v in os.environ.items() if k.startswith('PROJECT2_LATENCY_')},
        scope='Independent hybrid candidate:36 down GEMV decode +108 wide decode; original wide prefill701/2815; native fallback for otherM; full-model output/latency gate pending'))
    from lmdeploy.serve.openai.api_server import serve
    serve(str(model), model_name='navigation-repair-' + args.group, backend='pytorch',
        backend_config=engine, vision_config=VisionConfig(1), server_name='127.0.0.1',
        server_port=args.port, max_concurrent_requests=1, log_level='WARNING', max_log_len=64)


if __name__ == '__main__':
    main()
