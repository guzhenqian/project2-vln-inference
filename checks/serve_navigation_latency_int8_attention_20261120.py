"""Independent qkv/o expanded-INT8 prefill experiment and matched off control.

Original W4 decode/MLP, KV64x64 and image inputs remain native. This is a new
mixed-precision candidate; original same-weight auditors must reject its map.
"""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path

from navigation_stage2_common import ROOT,SOURCES,read,write,digest
from serve_navigation_repair import recipe
from lmdeploy.pytorch.models.patch import update_custom_module_map

PRIMARY_SHA='01c1111d1954d1eec38918f415ce0b81b455f4d8117b9f43b9d602a8e4c88345'
os.environ.setdefault('PROJECT2_LATENCY_EXECUTOR','uni')
os.environ.setdefault('PROJECT2_LATENCY_PREFILL_INTERVAL','2')
os.environ.setdefault('PROJECT2_LATENCY_INPROCESS','1')
MODE=os.environ.get('PROJECT2_LATENCY_INT8_ATTENTION','off')
assert MODE in ('off','qkv_o')
MODULE_MAP=('navigation_latency_int8_attention_module_map_20261120.py'
            if MODE=='qkv_o' else 'navigation_repair_module_map.py')
update_custom_module_map(str(ROOT/'checks'/MODULE_MAP))
import navigation_latency_inprocess_20261120
navigation_latency_inprocess_20261120.install()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',required=True,choices=('fp16','awq_w4a16'))
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--port',required=True,type=int)
    args=parser.parse_args()
    run=args.run_dir.resolve()
    assert run.is_relative_to(ROOT/'logs') and run.is_dir()
    assert digest(ROOT/'checks/serve_navigation_latency_20261120.py')==PRIMARY_SHA
    assert os.environ['PROJECT2_LATENCY_EXECUTOR']=='uni'
    assert os.environ['PROJECT2_LATENCY_PREFILL_INTERVAL']=='2'
    assert os.environ['PROJECT2_LATENCY_INPROCESS']=='1'
    for flag in ('PROJECT2_LATENCY_W4_TABLE','PROJECT2_LATENCY_W4_PHASE','PROJECT2_LATENCY_HYBRID_TABLE',
                 'PROJECT2_LATENCY_VISION_QMAX','PROJECT2_LATENCY_VISION_CACHE','PROJECT2_LATENCY_LINEAR_IPC',
                 'PROJECT2_LATENCY_PROFILE','PROJECT2_LATENCY_SCHEDULE_TRACE'):
        assert os.environ.get(flag) in (None,'','0','off'), 'Independent candidate cannot combine '+flag
    config=recipe(args.group)
    assert MODE!='qkv_o' or args.group=='awq_w4a16'
    versions={name:version(name) for name in ('lmdeploy','torch','transformers','triton')}
    assert versions==dict(lmdeploy='0.11.1',torch='2.5.1+cu121',transformers='4.57.6',triton='3.1.0')
    os.environ.update(PROJECT2_STAGE2_METHOD=config['method'],PROJECT2_STAGE2_KEEP_RATIO='1.0',
        PROJECT2_STAGE2_PROFILE='navigation',PROJECT2_STAGE2_RUN_DIR=str(run),
        PROJECT2_W4A16_FIX_VARIANT='stock',PROJECT2_W4A16_FIX_GRAPH='on',
        PROJECT2_REPAIR_WBITS=str(config['weight_bits']),PROJECT2_REPAIR_ABITS='16',
        PROJECT2_PRECISION_WBITS=str(config['weight_bits']),PROJECT2_PRECISION_ABITS='16')
    os.environ['PYTHONPATH']=str(ROOT/'checks')+os.pathsep+os.environ.get('PYTHONPATH','')
    from lmdeploy import PytorchEngineConfig
    from lmdeploy.messages import VisionConfig
    engine=PytorchEngineConfig(dtype='float16',tp=1,dp=1,session_len=4096,
        custom_module_map=str(ROOT/'checks'/MODULE_MAP),max_batch_size=1,max_prefill_token_num=3072,
        cache_max_entry_count=.35,num_gpu_blocks=64,eager_mode=False,enable_prefix_caching=False,
        enable_microbatch=False,enable_metrics=False,quant_policy=0,prefill_interval=2,
        distributed_executor_backend='uni',enable_mp_engine=True,mp_engine_backend='mp')
    names=SOURCES+['navigation_w4a16_fix_runtime.py','navigation_repair_runtime.py','navigation_repair_module_map.py',
        'serve_navigation_repair.py','serve_navigation_latency_20261120.py','navigation_latency_inprocess_20261120.py',
        'navigation_latency_expanded_int8_prefill_20261120.py','probe_navigation_latency_expanded_int8_prefill_20261120.py',
        'navigation_latency_int8_attention_adapter_20261120.py','navigation_latency_int8_attention_module_map_20261120.py',
        'navigation_latency_int8_attention_contract_20261120.py','serve_navigation_latency_int8_attention_20261120.py']
    if config['weight_bits']==4:
        names+=['navigation_w4a16_candidate_kernels.py','navigation_w4a16_repair_kernels.py']
    model=ROOT/config['model'];weight_sha=digest(model/'model.safetensors')
    expected=('b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b'
              if args.group=='fp16' else read(model/'conversion_audit.json')['sha256'])
    assert weight_sha==expected
    proof=None;probe_dir=None
    if MODE=='qkv_o':
        from navigation_latency_int8_attention_contract_20261120 import load_contract
        probe_dir=Path(os.environ['PROJECT2_LATENCY_INT8_PROBE']).resolve()
        assert probe_dir.is_relative_to(ROOT/'logs')
        proof=load_contract(probe_dir,'awq',weight_sha,verify_sources=True)
        assert proof['status']=='verified_micro_contract' and proof['allowed_projections']==['qkv','o']
        assert proof['layout']=='nk_transposed' and not proof['full_model_approval']
        assert digest(os.environ['PROJECT2_W4A16_REPAIR_TABLE'])==read(probe_dir/'protocol.json')['baseline_table_sha256']
    contract=dict(schema='navigation_int8_attention/v1',mode=MODE,layout='nk_transposed' if MODE=='qkv_o' else None,
        enabled_projections=['qkv','o'] if MODE=='qkv_o' else [],enabled_count=72 if MODE=='qkv_o' else 0,
        micro_probe_directory=str(probe_dir) if probe_dir else None,micro_proof=proof,
        arithmetic='New expanded per-channel W8/per-token A8 for qkv/o prefill; original W4 MLP and decode' if MODE=='qkv_o' else 'Original native model',
        actual_smoke_gate='Both input_warmup0/1, every72 enabled modules, finite and relativeL2<=0.02; stop without fallback',
        original_checkpoint_changed=False,kv_blocks=64,kv_block_size=64,whole_model_quality='not_evaluated',
        same_w4_output_equivalence_claim=False,requires_independent_arithmetic_audit=True)
    write(run/'launch.json',dict(group=args.group,recipe=config,versions=versions,
        model_name='navigation-repair-'+args.group,model_weights_sha256=weight_sha,
        model_config_sha256=digest(model/'config.json'),
        engine=json.loads(json.dumps(asdict(engine),default=lambda v:v.value)),
        sources_sha256={name:digest(ROOT/'checks'/name) for name in names},
        candidate_environment={k:v for k,v in os.environ.items() if k.startswith('PROJECT2_LATENCY_')},
        int8_attention_contract=contract,scope='Independent mixed-prefill precision experiment; pending actual errors and navigation quality'))
    from lmdeploy.serve.openai.api_server import serve
    serve(str(model),model_name='navigation-repair-'+args.group,backend='pytorch',backend_config=engine,
        vision_config=VisionConfig(1),server_name='127.0.0.1',server_port=args.port,max_concurrent_requests=1,
        log_level='WARNING',max_log_len=64)


if __name__=='__main__': main()
