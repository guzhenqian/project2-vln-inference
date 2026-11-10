"""Separate diagnostic service with unchanged eager3072 engine settings."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from importlib.metadata import version

from navigation_stage2_common import ROOT, GROUPS, SOURCES, digest, write
from lmdeploy.pytorch.models.patch import update_custom_module_map
update_custom_module_map(str(ROOT / 'checks/navigation_quant_timing_module_map.py'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['fp16','awq','gptq'], required=True)
    parser.add_argument('--mode', choices=['audited','native'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert run.is_relative_to(ROOT / 'logs') and (run / 'request_plan.json').is_file()
    assert version('lmdeploy') == '0.11.1' and version('torch') == '2.5.1+cu121'
    from lmdeploy import PytorchEngineConfig
    from lmdeploy.messages import VisionConfig
    engine = PytorchEngineConfig(dtype='float16', tp=1, dp=1, session_len=4096,
        max_batch_size=1, max_prefill_token_num=3072, cache_max_entry_count=.35,
        num_gpu_blocks=64, eager_mode=True, enable_prefix_caching=False,
        enable_microbatch=False, enable_metrics=False, quant_policy=0,
        distributed_executor_backend='mp', enable_mp_engine=True, mp_engine_backend='mp')
    os.environ.update(PROJECT2_STAGE2_METHOD=args.group, PROJECT2_STAGE2_KEEP_RATIO='1.0',
        PROJECT2_STAGE2_PROFILE='navigation', PROJECT2_STAGE2_RUN_DIR=str(run),
        PROJECT2_QUANT_TIMING_MODE=args.mode)
    os.environ['PYTHONPATH'] = str(ROOT / 'checks') + os.pathsep + os.environ.get('PYTHONPATH','')
    new_sources = ['serve_navigation_quant_timing.py','navigation_quant_timing_runtime.py','navigation_quant_timing_module_map.py']
    write(run/'launch.json', dict(group=args.group, mode=args.mode, model=GROUPS[args.group]['model'],
        engine=json.loads(json.dumps(asdict(engine), default=lambda x:x.value)),
        sources_sha256={s:digest(ROOT/'checks'/s) for s in SOURCES+new_sources}))
    from lmdeploy.serve.openai.api_server import serve
    serve(str(ROOT/GROUPS[args.group]['model']), model_name='quant-timing-'+args.group,
        backend='pytorch', backend_config=engine, vision_config=VisionConfig(1),
        server_name='127.0.0.1', server_port=args.port, max_concurrent_requests=1,
        log_level='WARNING', max_log_len=64)


if __name__ == '__main__':
    main()
