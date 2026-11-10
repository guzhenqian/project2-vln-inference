"""One owned stage2 service; custom mapping is registered in spawned workers."""
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
import os
import socket
from importlib.metadata import version
from pathlib import Path
import uuid

from navigation_precision_common import ROOT, GROUPS, SOURCES, read, write, digest, model_name


def register():
    from lmdeploy.pytorch.models.patch import update_custom_module_map
    update_custom_module_map(str(ROOT / 'checks/navigation_precision_module_map.py'))


register()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    group = GROUPS[args.group]
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT / 'logs') or run.exists():
        raise ValueError('Expected fresh log directory inside project')
    source = ROOT / group['model']
    kernel_gate = read(ROOT/'reports/navigation_precision_kernel_gate_linked_20261110.json')
    assert kernel_gate['status'] == 'passed'
    assert kernel_gate['kernel_source_sha256'] == digest(ROOT/'checks/navigation_precision_kernels.py')
    config = read(source / 'config.json')
    expected_versions = dict(lmdeploy='0.11.1', torch='2.5.1+cu121', transformers='4.57.6')
    if any(version(name) != value for name, value in expected_versions.items()):
        raise ValueError('Inference environment changed')
    if group['method'] in ('awq', 'gptq'):
        audit = read(source / 'conversion_audit.json')
        if audit['quantized_layer_count'] != 252 or audit['bytes'] != (source / 'model.safetensors').stat().st_size:
            raise ValueError('Converted model incomplete')
        if group['method'] == 'gptq' and audit.get('quantization_algorithm') != 'GPTQ':
            raise ValueError('GPTQ provenance absent')
    elif config.get('quantization_config') is not None:
        raise ValueError('Token-only group must be FP16')
    assert digest(source/'model.safetensors') == audit['sha256'], 'Weight hash mismatch'
    from lmdeploy import PytorchEngineConfig
    engine = PytorchEngineConfig(dtype='float16', tp=1, dp=1, session_len=4096,
        max_batch_size=1, max_prefill_token_num=3072, cache_max_entry_count=.35,
        num_gpu_blocks=64, eager_mode=True, enable_prefix_caching=False,
        enable_microbatch=False, enable_metrics=False, quant_policy=0,
        distributed_executor_backend='mp', enable_mp_engine=True, mp_engine_backend='mp')
    assert engine.block_size == 64
    engine_dict = json.loads(json.dumps(asdict(engine), default=lambda value: value.value))
    if args.check_only:
        print('STAGE2_SERVICE_CONFIG_CHECKED', args.group)
        return
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 23333))
    run.mkdir(parents=True, exist_ok=False)
    os.environ.update(PROJECT2_STAGE2_METHOD=group['method'], PROJECT2_STAGE2_KEEP_RATIO=str(group['keep']),
        PROJECT2_STAGE2_PROFILE='navigation', PROJECT2_STAGE2_RUN_DIR=str(run))
    os.environ['PROJECT2_PRECISION_WBITS'] = str(group['weight_bits'])
    os.environ['PROJECT2_PRECISION_ABITS'] = str(group['activation_bits'])
    os.environ['PYTHONPATH'] = str(ROOT / 'checks') + os.pathsep + os.environ.get('PYTHONPATH', '')
    write(run / 'launch.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        precision_config_sha256=digest(ROOT/'setup/navigation-precision-20261110.json'),
        group=args.group, recipe=group, keep_ratio=group['keep'], profile='navigation',
        model_name=model_name(args.group), model_source=group['model'],
        model_config_sha256=digest(source / 'config.json'),
        model_weights_sha256=audit['sha256'], model_bytes=audit['bytes'],
        conversion_audit_sha256=digest(source/'conversion_audit.json'),
        engine_config=engine_dict, versions=expected_versions,
        sources_sha256={name: digest(ROOT / 'checks' / name) for name in SOURCES},
        instrumentation='Original detailed traces plus actual precision-kernel receipts; included in latency',
        token_variant='PriorTR-v1 single-layer without merging' if group['method']=='sparsevlm' else group['method']))
    print('STAGE2_SERVICE', args.group, run.relative_to(ROOT), flush=True)
    from lmdeploy.messages import VisionConfig
    from lmdeploy.serve.openai.api_server import serve
    serve(str(source), model_name=model_name(args.group), backend='pytorch',
        backend_config=engine, vision_config=VisionConfig(1), server_name='127.0.0.1',
        server_port=23333, max_concurrent_requests=1, log_level='INFO', max_log_len=256)


if __name__ == '__main__':
    main()
