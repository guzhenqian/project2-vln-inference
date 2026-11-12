"""Independent stock/candidate W4A16 native eager/graph diagnostic service."""
import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from navigation_stage2_common import ROOT, GROUPS, SOURCES, digest, write
from lmdeploy.pytorch.models.patch import update_custom_module_map

# Python spawn re-executes this module as __mp_main__. Registration therefore
# must occur at module scope, not only in main(). LMDeploy 0.11.1 copies the
# engine field into MiscConfig but its model worker reads ModelConfig instead;
# passing custom_module_map in PytorchEngineConfig alone is insufficient here.
update_custom_module_map(str(ROOT/'checks/navigation_w4a16_fix_module_map.py'))


def _registration_probe(queue):
    """Resolve the worker class without constructing a model or touching CUDA."""
    from types import SimpleNamespace
    import torch
    from lmdeploy.pytorch.models.patch import _get_module_map, _get_model_class
    assert not torch.cuda.is_initialized()
    mapping = _get_module_map()
    config = SimpleNamespace(architectures=['Qwen3VLForConditionalGeneration'])
    selected = _get_model_class(config, mapping)
    name = selected.__module__ + '.' + selected.__name__
    assert name == 'navigation_w4a16_fix_runtime.W4A16FixModel', name
    assert not torch.cuda.is_initialized()
    queue.put(dict(pid=os.getpid(), spawned_main_name=__name__, selected_class=name,
        mapping=mapping['Qwen3VLForConditionalGeneration'], cuda_initialized=False))


def main():
    if sys.argv[1:] == ['--check-spawn-registration']:
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        child = context.Process(target=_registration_probe, args=(queue,))
        child.start()
        child.join(timeout=45)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
            raise TimeoutError('CPU-only module registration probe')
        assert child.exitcode == 0, child.exitcode
        print(json.dumps(queue.get(timeout=2)), flush=True)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=('fp16', 'awq', 'gptq'), required=True)
    parser.add_argument('--variant', choices=('stock', 'phase_tuned', 'decode_nosplit', 'prefill_dequant', 'hybrid'), default='stock')
    parser.add_argument('--graph', choices=('off', 'on'), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assert args.group != 'fp16' or args.variant == 'stock'
    run = args.run_dir.resolve()
    assert run.is_relative_to(ROOT/'logs') and run.is_dir()
    assert version('lmdeploy') == '0.11.1' and version('torch') == '2.5.1+cu121'
    os.environ.update(PROJECT2_STAGE2_METHOD=args.group, PROJECT2_STAGE2_KEEP_RATIO='1.0',
        PROJECT2_STAGE2_PROFILE='navigation', PROJECT2_STAGE2_RUN_DIR=str(run),
        PROJECT2_W4A16_FIX_VARIANT=args.variant, PROJECT2_W4A16_FIX_GRAPH=args.graph)
    os.environ['PYTHONPATH'] = str(ROOT/'checks') + os.pathsep + os.environ.get('PYTHONPATH', '')
    from lmdeploy import PytorchEngineConfig
    from lmdeploy.messages import VisionConfig
    engine = PytorchEngineConfig(dtype='float16', tp=1, dp=1, session_len=4096,
        # Retain the declared mapping in the config receipt. Module-top
        # registration is the working spawn workaround in installed 0.11.1.
        custom_module_map=str(ROOT/'checks/navigation_w4a16_fix_module_map.py'),
        max_batch_size=1, max_prefill_token_num=3072, cache_max_entry_count=.35,
        num_gpu_blocks=64, eager_mode=args.graph == 'off', enable_prefix_caching=False,
        enable_microbatch=False, enable_metrics=False, quant_policy=0,
        distributed_executor_backend='mp', enable_mp_engine=True, mp_engine_backend='mp')
    new_sources = ['serve_navigation_w4a16_fix.py', 'navigation_w4a16_fix_runtime.py',
                   'navigation_w4a16_fix_module_map.py']
    if args.variant != 'stock':
        new_sources.append('navigation_w4a16_candidate_kernels.py')
    model = ROOT/GROUPS[args.group]['model']
    write(run/'launch.json', dict(group=args.group, variant=args.variant, graph=args.graph,
        model=GROUPS[args.group]['model'], model_name='w4a16-fix-'+args.group,
        model_config_sha256=digest(model/'config.json'),
        model_safetensors_size=(model/'model.safetensors').stat().st_size,
        weight_file_hash_scope='Serial client verifies frozen checkpoint before sessions; server audits loaded tensor names/types.',
        engine=json.loads(json.dumps(asdict(engine), default=lambda x:x.value)),
        sources_sha256={name:digest(ROOT/'checks'/name) for name in SOURCES+new_sources},
        preserved='Same native forward, model weights, 3072 full prefill, FP16 KV64x64, prefix off; graph changes decode scheduling only'))
    from lmdeploy.serve.openai.api_server import serve
    serve(str(model), model_name='w4a16-fix-'+args.group, backend='pytorch', backend_config=engine,
        vision_config=VisionConfig(1), server_name='127.0.0.1', server_port=args.port,
        max_concurrent_requests=1, log_level='WARNING', max_log_len=64)


if __name__ == '__main__':
    main()
