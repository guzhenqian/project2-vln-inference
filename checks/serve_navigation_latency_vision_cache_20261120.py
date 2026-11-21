"""Dedicated CPU visual-cache experiment; frozen primary service stays unchanged.

The same entry supports off and cpu controls. Requires a dedicated reset plan,
two excluded strict numerical warmups, and separate cache-aware replay audit.
This entry is intentionally not compatible with the frozen native speed auditor.
"""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path

from navigation_stage2_common import ROOT, SOURCES, read, write, digest
from serve_navigation_repair import recipe
from lmdeploy.pytorch.models.patch import update_custom_module_map


MODULE_MAP = 'navigation_latency_vision_cpu_cache_module_map_20261120.py'
PRIMARY_SOURCE_SHA = '01c1111d1954d1eec38918f415ce0b81b455f4d8117b9f43b9d602a8e4c88345'
REPAIR_SOURCE_SHA = 'e3ca4c02732d531f85a3a6306966549d31e8217c056a7d5a8cf3d91f51813ae8'
os.environ.setdefault('PROJECT2_LATENCY_EXECUTOR', 'uni')
os.environ.setdefault('PROJECT2_LATENCY_PREFILL_INTERVAL', '2')
os.environ.setdefault('PROJECT2_LATENCY_INPROCESS', '1')
os.environ.setdefault('PROJECT2_LATENCY_VISION_CACHE', 'off')
os.environ.setdefault('PROJECT2_LATENCY_VISION_CACHE_CAPACITY', '96')
os.environ.setdefault('PROJECT2_LATENCY_VISION_CACHE_VALIDATE', '1')
update_custom_module_map(str(ROOT/'checks'/MODULE_MAP))
import navigation_latency_inprocess_20261120
import navigation_latency_vision_cpu_cache_20261120
navigation_latency_inprocess_20261120.install()
navigation_latency_vision_cpu_cache_20261120.install()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=('fp16', 'awq_w4a16'))
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert run.is_relative_to(ROOT/'logs') and run.is_dir()
    assert digest(ROOT/'checks/serve_navigation_latency_20261120.py') == PRIMARY_SOURCE_SHA
    assert digest(ROOT/'checks/navigation_repair_runtime.py') == REPAIR_SOURCE_SHA
    assert os.environ['PROJECT2_LATENCY_EXECUTOR'] == 'uni'
    assert os.environ['PROJECT2_LATENCY_PREFILL_INTERVAL'] == '2'
    assert os.environ['PROJECT2_LATENCY_INPROCESS'] == '1'
    assert os.environ['PROJECT2_LATENCY_VISION_CACHE_VALIDATE'] == '1'
    mode = os.environ['PROJECT2_LATENCY_VISION_CACHE']
    capacity = int(os.environ['PROJECT2_LATENCY_VISION_CACHE_CAPACITY'])
    assert mode in ('off', 'cpu') and capacity in (96, 200)
    assert (run/'cache_request_plan.json').is_file() and (run/'request_plan.json').is_file()
    config = recipe(args.group)
    versions = {name:version(name) for name in ('lmdeploy', 'torch', 'transformers', 'triton')}
    assert versions == dict(lmdeploy='0.11.1', torch='2.5.1+cu121', transformers='4.57.6', triton='3.1.0')
    os.environ.update(PROJECT2_STAGE2_METHOD=config['method'], PROJECT2_STAGE2_KEEP_RATIO='1.0',
        PROJECT2_STAGE2_PROFILE='navigation', PROJECT2_STAGE2_RUN_DIR=str(run),
        PROJECT2_W4A16_FIX_VARIANT='stock', PROJECT2_W4A16_FIX_GRAPH='on',
        PROJECT2_REPAIR_WBITS=str(config['weight_bits']), PROJECT2_REPAIR_ABITS='16',
        PROJECT2_PRECISION_WBITS=str(config['weight_bits']), PROJECT2_PRECISION_ABITS='16')
    os.environ['PYTHONPATH'] = str(ROOT/'checks')+os.pathsep+os.environ.get('PYTHONPATH', '')
    from lmdeploy import PytorchEngineConfig
    from lmdeploy.messages import VisionConfig
    engine = PytorchEngineConfig(dtype='float16', tp=1, dp=1, session_len=4096,
        custom_module_map=str(ROOT/'checks'/MODULE_MAP), max_batch_size=1,
        max_prefill_token_num=3072, cache_max_entry_count=.35, num_gpu_blocks=64,
        eager_mode=False, enable_prefix_caching=False, enable_microbatch=False,
        enable_metrics=False, quant_policy=0, prefill_interval=2,
        distributed_executor_backend='uni', enable_mp_engine=True, mp_engine_backend='mp')
    names = SOURCES + ['navigation_w4a16_fix_runtime.py', 'navigation_repair_runtime.py',
        'navigation_repair_module_map.py', 'serve_navigation_repair.py',
        'serve_navigation_latency_20261120.py', 'navigation_latency_inprocess_20261120.py',
        'navigation_latency_vision_qmax_20261120.py',
        'navigation_latency_vision_cpu_cache_20261120.py', MODULE_MAP,
        'serve_navigation_latency_vision_cache_20261120.py']
    if config['weight_bits'] == 4:
        names += ['navigation_w4a16_candidate_kernels.py', 'navigation_w4a16_repair_kernels.py']
    model = ROOT/config['model']
    weight_sha = digest(model/'model.safetensors')
    expected = ('b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b'
                if args.group == 'fp16' else read(model/'conversion_audit.json')['sha256'])
    assert weight_sha == expected
    contract = dict(schema='navigation_vision_cpu_cache/v1', mode=mode, capacity=capacity,
        storage='cpu_pageable', transfer='blocking original-stream D2H/H2D; included online',
        validation='strict excluded input warmup0/1, four tensors, bytewise failure stops service',
        request_plan_sha256=digest(run/'cache_request_plan.json'),
        processor_config_files_sha256={name:digest(model/name) for name in
            ('config.json', 'preprocessor_config.json', 'processor_config.json') if (model/name).is_file()},
        key='actual decoded RGB plus complete processor semantics/grid/checkpoint/dtype/source identity',
        full_visual_tokens=True, deepstack_arrays=3, prefix_kv_cache=False,
        episode_reset='Actual model LRU clear bound to dedicated request plan/index',
        production_ready=False, requires_cache_aware_audit=True)
    write(run/'launch.json', dict(group=args.group, recipe=config, versions=versions,
        model_name='navigation-repair-'+args.group, model_weights_sha256=weight_sha,
        model_config_sha256=digest(model/'config.json'),
        engine=json.loads(json.dumps(asdict(engine), default=lambda x:x.value)),
        sources_sha256={name:digest(ROOT/'checks'/name) for name in names},
        candidate_environment={k:v for k,v in os.environ.items() if k.startswith('PROJECT2_LATENCY_')},
        vision_cache_contract=contract,
        scope='Independent CPU visual-cache control/candidate; not a frozen native latency result'))
    from lmdeploy.serve.openai.api_server import serve
    serve(str(model), model_name='navigation-repair-'+args.group, backend='pytorch',
        backend_config=engine, vision_config=VisionConfig(1), server_name='127.0.0.1',
        server_port=args.port, max_concurrent_requests=1, log_level='WARNING', max_log_len=64)


if __name__ == '__main__':
    main()
