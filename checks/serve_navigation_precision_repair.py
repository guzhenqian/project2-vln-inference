"""Same common repair service with explicitly prepared A4/A8/W8 kernels."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path

from navigation_stage2_common import ROOT, GROUPS, SOURCES, read, write, digest
from lmdeploy.pytorch.models.patch import update_custom_module_map

update_custom_module_map(str(ROOT/'checks/navigation_precision_repair_module_map.py'))


def recipe(group):
    if group == 'fp16':
        return dict(method='fp16', model=GROUPS['fp16']['model'], weight_bits=16, activation_bits=16)
    algorithm, precision = group.split('_')
    if precision == 'w4a16':
        return dict(method=algorithm, model=GROUPS[algorithm]['model'], weight_bits=4, activation_bits=16)
    value = dict(read(ROOT/'setup/navigation-precision-20261110.json')['groups'][group])
    value['method'] = value['algorithm']
    from navigation_precision_repair_kernels import ACTIVATION_RECIPES
    value['previous_activation_recipe'] = value.get('activation_recipe')
    value['activation_recipe'] = ACTIVATION_RECIPES[value['activation_bits']]
    value['compute'] = 'int8_mma' if value['activation_bits'] < 16 else 'fp16_mma'
    value['activation_storage_bits'] = value['activation_bits']
    value['native_int4_mma'] = False
    value['kernel_variant'] = 'balanced'
    value['launch_policy'] = 'explicit startup preparation then frozen; no online tuning or cold compilation'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    config = recipe(args.group)
    assert (config['weight_bits'], config['activation_bits']) in ((4,4),(4,8),(8,16))
    run = args.run_dir.resolve()
    assert run.is_relative_to(ROOT/'logs') and run.is_dir()
    versions = {name: version(name) for name in ('lmdeploy', 'torch', 'transformers', 'triton')}
    assert versions == dict(lmdeploy='0.11.1', torch='2.5.1+cu121', transformers='4.57.6', triton='3.1.0')
    os.environ.update(PROJECT2_STAGE2_METHOD=config['method'], PROJECT2_STAGE2_KEEP_RATIO='1.0',
        PROJECT2_STAGE2_PROFILE='navigation', PROJECT2_STAGE2_RUN_DIR=str(run),
        PROJECT2_W4A16_FIX_VARIANT='stock', PROJECT2_W4A16_FIX_GRAPH='on',
        PROJECT2_REPAIR_WBITS=str(config['weight_bits']), PROJECT2_REPAIR_ABITS=str(config['activation_bits']),
        PROJECT2_PRECISION_WBITS=str(config['weight_bits']), PROJECT2_PRECISION_ABITS=str(config['activation_bits']))
    os.environ['PYTHONPATH'] = str(ROOT/'checks') + os.pathsep + os.environ.get('PYTHONPATH', '')
    from lmdeploy import PytorchEngineConfig
    from lmdeploy.messages import VisionConfig
    engine = PytorchEngineConfig(dtype='float16', tp=1, dp=1, session_len=4096,
        custom_module_map=str(ROOT/'checks/navigation_precision_repair_module_map.py'),
        max_batch_size=1, max_prefill_token_num=3072, cache_max_entry_count=.35,
        num_gpu_blocks=64, eager_mode=False, enable_prefix_caching=False,
        enable_microbatch=False, enable_metrics=False, quant_policy=0,
        distributed_executor_backend='mp', enable_mp_engine=True, mp_engine_backend='mp')
    names = SOURCES + ['navigation_w4a16_fix_runtime.py', 'navigation_repair_runtime.py',
        'navigation_precision_repair_module_map.py', 'serve_navigation_precision_repair.py',
        'navigation_precision_repair_runtime.py']
    if config['weight_bits'] == 4 and config['activation_bits'] == 16:
        names += ['navigation_w4a16_candidate_kernels.py', 'navigation_w4a16_repair_kernels.py']
    elif config['weight_bits'] != 16:
        names += ['navigation_precision_guard.py', 'navigation_precision_repair_kernels.py']
    model = ROOT/config['model']
    weight_sha = digest(model/'model.safetensors')
    if args.group == 'fp16':
        assert weight_sha == 'b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b'
    else:
        assert weight_sha == read(model/'conversion_audit.json')['sha256']
    write(run/'launch.json', dict(group=args.group, recipe=config, versions=versions,
        model_name='navigation-repair-'+args.group, model_weights_sha256=weight_sha,
        model_config_sha256=digest(model/'config.json'),
        engine=json.loads(json.dumps(asdict(engine), default=lambda x: x.value)),
        sources_sha256={name:digest(ROOT/'checks'/name) for name in names},
        scope='Native full prefill3072, FP16 KV64x64, decode graph; no token compression'))
    from lmdeploy.serve.openai.api_server import serve
    serve(str(model), model_name='navigation-repair-'+args.group, backend='pytorch',
        backend_config=engine, vision_config=VisionConfig(1), server_name='127.0.0.1',
        server_port=args.port, max_concurrent_requests=1, log_level='WARNING', max_log_len=64)


if __name__ == '__main__':
    main()
