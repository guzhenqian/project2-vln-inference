"""Small stdlib-only fake-object QA for the optional vision launch-grid patch.

No torch/LMDeploy modules or GPU code are imported. Installed source is read
and parsed as text through WSL; a synthetic package exercises install/forward.
Fake receipts prove Python control flow only, never real model execution.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType, ModuleType, SimpleNamespace


PATCH = Path(__file__).with_name('navigation_latency_vision_qmax_20261120.py')
SOURCE_ROOT = '/home/guzhenqian/.venvs/project2/lib/python3.12/site-packages/lmdeploy'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject(fn, contains):
    try:
        fn()
    except RuntimeError as error:
        require(contains in str(error), f'Wrong rejection: {error}')
    else:
        raise AssertionError(f'Expected rejection: {contains}')


def load_patch(name):
    spec = importlib.util.spec_from_file_location(name, PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, shape, rows=None, device='cuda:0', dtype='torch.float16'):
        self.shape = tuple(shape)
        self.rows = rows
        self.device = SimpleNamespace(type=device.split(':')[0])
        self.dtype = dtype
        self.tolist_calls = 0

    def numel(self):
        product = 1
        for value in self.shape:
            product *= value
        return product

    def tolist(self):
        require(self.device.type == 'cpu', 'GPU scalar/tolist read attempted')
        self.tolist_calls += 1
        return self.rows

    def item(self):
        raise AssertionError('Scalar read attempted')


def grid(rows, device='cpu', shape=None):
    return Tensor(shape or (len(rows), 3), rows, device=device, dtype='torch.int64')


def packet(rows):
    total = sum(t*h*w for t, h, w in rows)
    segments = sum(t for t, h, w in rows)
    return [Tensor((total, 16, 80)) for _ in range(3)] + [Tensor((segments,), dtype='torch.int32') for _ in range(2)]


def source_audit(expected, source_root):
    script = ('import json,pathlib; root=pathlib.Path(' + repr(source_root) + '); '
              'names=' + repr(list(expected)) + '; '
              'print(json.dumps({n:(root/n).read_text() for n in names}))')
    result = subprocess.run(['wsl', '-e', '/usr/bin/python3', '-c', script],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
    sources = json.loads(result.stdout)
    hashes = {name:sha(text.encode()) for name, text in sources.items()}
    require(hashes == expected, 'Installed source hash differs from guarded candidate')
    kernel = ast.parse(sources['pytorch/kernels/cuda/flashattention.py'])
    functions = {n.name:n for n in kernel.body if isinstance(n, ast.FunctionDef)}
    launch = functions['flash_attention_fwd']
    grid_fn = next(n for n in launch.body if isinstance(n, ast.FunctionDef) and n.name == 'grid')
    ret = next(n.value for n in grid_fn.body if isinstance(n, ast.Return))
    require(isinstance(ret, ast.Tuple) and len(ret.elts) == 3, 'Unexpected launch grid')
    require(ast.unparse(ret.elts[0]) == "triton.cdiv(max_seqlen, args['BLOCK_M'])", 'Grid bound changed')
    names = [n for n in ast.walk(launch) if isinstance(n, ast.Name) and n.id == 'max_seqlen']
    allowed = set(ast.walk(grid_fn))
    for node in launch.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'max_seqlen is None':
            allowed.update(ast.walk(node))
    require(names and all(n in allowed for n in names), 'max_seqlen affects more than grid/default')
    for name, fn in functions.items():
        if name.startswith('_kernel_meta_'):
            require(not any(isinstance(n, ast.Name) and n.id == 'max_seqlen' for n in ast.walk(fn)),
                    'Kernel metadata depends on max_seqlen')
    kernel_fn = functions['_flash_prefill_fwd_kernel']
    require('max_seqlen' not in [n.arg for n in kernel_fn.args.args], 'Bound became a kernel argument')
    require(not any('autotune' in ast.unparse(d) for d in kernel_fn.decorator_list), 'Active autotune found')
    attention = ast.parse(sources['pytorch/nn/attention.py'])
    cls = next(n for n in attention.body if isinstance(n, ast.ClassDef) and n.name == 'FlashAttention')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    names = [n.arg for n in forward.args.args]
    require(names == ['self', 'query', 'key', 'value', 'q_start_loc', 'q_seqlens',
                      'kv_start_loc', 'kv_seqlens', 'max_q_seqlen'], 'FlashAttention signature changed')
    candidate = ast.parse(PATCH.read_text(encoding='utf-8'))
    imports = [n.module for n in ast.walk(candidate) if isinstance(n, ast.ImportFrom)]
    require(not any(n and n.startswith('lmdeploy.pytorch.kernels') for n in imports),
            'Install imports CUDA kernel package')
    compare = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == '_compare_features')
    per_feature_pass = [n.value for n in ast.walk(compare) if isinstance(n, ast.keyword) and n.arg == 'passed'
                        and isinstance(n.value, ast.BoolOp)]
    require(len(per_feature_pass) == 1 and {n.id for n in ast.walk(per_feature_pass[0]) if isinstance(n, ast.Name)} ==
            {'metadata_equal','finite','distinct_storage','equal','bitwise_equal'}, 'Numerical pass criteria changed')
    byte_compare = next(n.value for n in compare.body if isinstance(n, ast.For) for n in n.body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bitwise_equal'
                                                            for t in n.targets))
    require('old.contiguous().view(torch.uint8)' in ast.unparse(byte_compare) and
            'new.contiguous().view(torch.uint8)' in ast.unparse(byte_compare), 'Bitwise check not over storage bytes')
    return dict(source_sha256=hashes, flash_signature=names, max_seqlen_name_lines=sorted(n.lineno for n in
                ast.walk(launch) if isinstance(n, ast.Name) and n.id == 'max_seqlen'),
                effective_change='Launch grid.x only; no kernel argument, autotune, block/warp/stage dependency')


@contextmanager
def fake_package(directory, module):
    root = directory/'lmdeploy'
    model_source = '''class Qwen3VLForConditionalGeneration:
    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None,
                inputs_embeds=None, mrope_position_ids=None, pixel_values=None,
                vis_cu_seqlens=None, vis_pos_emb=None, image_mask=None,
                pos_embeds=None, grid_thw=None, **kwargs):
        return self._execute(input_ids, position_ids, past_key_values,
            attn_metadata=attn_metadata, inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids, pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens, vis_pos_emb=vis_pos_emb,
            image_mask=image_mask, pos_embeds=pos_embeds, grid_thw=grid_thw, **kwargs)
'''
    texts = {name:'# synthetic source only\n' for name in module.SOURCE_SHA256}
    texts['pytorch/models/qwen3_vl.py'] = model_source
    texts['pytorch/kernels/cuda/flashattention.py'] = 'def flash_attention_fwd(*args, **kwargs):\n    raise AssertionError("Fake kernel must never execute")\n'
    for name, text in texts.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    module.SOURCE_SHA256 = {name:sha((root/name).read_bytes()) for name in texts}
    names = ['lmdeploy', 'lmdeploy.pytorch', 'lmdeploy.pytorch.models', 'lmdeploy.pytorch.models.qwen3_vl',
             'lmdeploy.pytorch.nn', 'lmdeploy.pytorch.nn.attention', 'lmdeploy.pytorch.backends',
             'lmdeploy.pytorch.backends.cuda', 'lmdeploy.pytorch.backends.cuda.flash_attention']
    previous = {name:sys.modules.get(name) for name in names}
    modules = {}
    for name in names:
        value = ModuleType(name)
        value.__path__ = []
        modules[name] = sys.modules[name] = value
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(modules[parent], attr, value)
    model_mod = modules['lmdeploy.pytorch.models.qwen3_vl']
    model_mod.__file__ = str(root/'pytorch/models/qwen3_vl.py')
    exec(compile(model_source, model_mod.__file__, 'exec'), model_mod.__dict__)
    kernel_globals = {'__name__':'lmdeploy.pytorch.kernels.cuda.flashattention'}
    exec(compile(texts['pytorch/kernels/cuda/flashattention.py'],
                 str(root/'pytorch/kernels/cuda/flashattention.py'), 'exec'), kernel_globals)
    native_kernel = kernel_globals['flash_attention_fwd']

    class Impl:
        def __init__(self):
            self.causal = False
            self.flash_attention_fwd = native_kernel

    class Attention:
        def __init__(self, index):
            self.index = index
            self.impl = Impl()
            self.seen = []

        def forward(self, query, key, value, q_start_loc, q_seqlens,
                    kv_start_loc=None, kv_seqlens=None, max_q_seqlen=None):
            self.seen.append((query, key, value, q_start_loc, q_seqlens,
                              kv_start_loc, kv_seqlens, max_q_seqlen))
            return query

    modules['lmdeploy.pytorch.nn.attention'].FlashAttention = Attention
    modules['lmdeploy.pytorch.backends.cuda.flash_attention'].TritonFlashAttentionImpl = Impl
    try:
        yield model_mod.Qwen3VLForConditionalGeneration, Attention, Impl, native_kernel
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def execute(self, input_ids, position_ids, past_key_values, **kwargs):
    self.native_calls += 1
    self.native_seen = (input_ids, position_ids, past_key_values, kwargs)
    if self.fail_native:
        raise RuntimeError('synthetic native failure')
    if kwargs['pixel_values'] is not None and kwargs['inputs_embeds'] is None:
        self.last_visual_output = self.visual.forward(kwargs['pixel_values'])
    self.language_calls += 1
    return self.output


def vision_execute(self, *args, **kwargs):
    self.call_count += 1
    if self.fail_on_call == self.call_count:
        raise RuntimeError('synthetic visual failure')
    for block in self.blocks[:self.owner.execute_layers]:
        block.attn.attention.forward(*self.owner.packet)
    result = (object(), [object(), object(), object()])
    self.outputs.append(result)
    return result


def make_model(model_type, attention_type):
    model = model_type()
    model.visual = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(attention=attention_type(i)))
                                          for i in range(24)])
    model._execute = MethodType(execute, model)
    model.visual.owner = model
    model.visual.forward = MethodType(vision_execute, model.visual)
    model.visual.call_count = 0
    model.visual.fail_on_call = None
    model.visual.outputs = []
    model.native_calls = 0
    model.language_calls = 0
    model.fail_native = False
    model.execute_layers = 24
    model.output = object()
    return model


def gate_test(directory, candidate_hash, checks):
    module = load_patch('vision_qmax_gate_fake_qa')
    gate_dir = directory/'gate'
    gate_dir.mkdir()
    receipts = gate_dir/'receipts'
    receipts.mkdir()
    os.environ['PROJECT2_STAGE2_RUN_DIR'] = str(receipts)
    os.environ['PROJECT2_LATENCY_VISION_QMAX_VALIDATE'] = '1'
    comparisons = []
    decision = {'passed':True, 'raise':False}

    def compare(reference, candidate):
        if decision['raise']:
            raise RuntimeError('synthetic comparator failure')
        require(reference is not candidate and all(a is not b for a,b in
                zip([reference[0], *reference[1]], [candidate[0], *candidate[1]])), 'Fake outputs alias')
        comparisons.append((reference, candidate))
        return dict(passed=decision['passed'], features=[dict(feature=label, passed=decision['passed'],
                    metadata_equal=True, finite=True, distinct_storage=True, torch_equal=decision['passed'],
                    bitwise_equal=decision['passed'], max_abs_error=0 if decision['passed'] else 1,
                    relative_l2=0 if decision['passed'] else 1)
                    for label in ('main', 'deepstack0', 'deepstack1', 'deepstack2')])

    module._compare_features = compare
    with fake_package(gate_dir, module) as (model_type, attention_type, impl_type, kernel):
        module.install()
        model = make_model(model_type, attention_type)
        model._fix_plan = [dict(phase='input_warmup', tokens=0, sample=i) for i in range(2)]
        model._fix_dir = receipts
        requests = []
        for index, count in enumerate((2, 9)):
            model._fix_index = index
            rows = [[1,30,40]]*count
            model.packet = packet(rows)
            input_ids = Tensor((700 if index == 0 else 2800,), dtype='torch.int64')
            input_receipt = dict(request_index=index, input_ids={'shape':list(input_ids.shape), 'sha256':'fake'},
                                 pixel_values={'shape':[count*1200,1536], 'sha256':'fake'})
            (receipts/f'input_{input_ids.numel()}.json').write_text(json.dumps(input_receipt), encoding='utf-8')
            args = [input_ids, object(), object()]
            kwargs = dict(pixel_values=Tensor((count*1200,1536)), grid_thw=grid(rows))
            previous = (model.native_calls, model.language_calls, model.visual.call_count)
            seen_before = len(model.visual.blocks[0].attn.attention.seen)
            outer = object()
            token = module._CONTEXT.set(outer)
            try:
                require(model.forward(*args, **kwargs) is model.output, 'Gate changed model output')
                require(module._CONTEXT.get() is outer, 'Nested gate context not restored')
            finally:
                module._CONTEXT.reset(token)
            require((model.native_calls,model.language_calls,model.visual.call_count) ==
                    (previous[0]+1,previous[1]+1,previous[2]+2), 'Gate did not run vision twice/language once')
            require(model.last_visual_output is comparisons[-1][1], 'Native model did not receive candidate features')
            for block in model.visual.blocks:
                seen = block.attn.attention.seen[seen_before:]
                require(len(seen) == 2 and [entry[7] for entry in seen] == [None,1200], 'Baseline/candidate bound differs')
                require(all(a is b for entry in seen for a,b in zip(entry[:5],model.packet)), 'Gate input identities differ')
            requests.append((args,kwargs,rows))
        checks += ['eligible 2/9 warmups: vision twice, native full model and language once',
                   'all 24 baseline calls use native default; all 24 candidate calls use CPU qmax',
                   'candidate visual features returned; nested reference and outer contexts restored']
        before = (model.visual.call_count,len(comparisons))
        args, kwargs, rows = requests[1]
        model.forward(*args, **kwargs)
        require((model.visual.call_count,len(comparisons)) == (before[0]+1,before[1]), 'Repeat shape reran gate')
        model._fix_index = 2
        rows = [[1,30,40]]*3
        model.packet = packet(rows)
        model.forward(Tensor((1000,)),object(),object(), pixel_values=Tensor((3600,1536)),grid_thw=grid(rows))
        require(len(comparisons) == before[1], 'Later unseen shape ran gate')
        checks.append('repeated shape and later unseen shape perform one visual forward without numerical gate')
        model._fix_index = 0
        context = module._grid_context(grid([[1,30,40]]*2),Tensor((2400,1536)),model)
        ids = Tensor((700,))
        eligible, _ = module._validation_eligibility(model,context,ids)
        require(eligible, 'Saved warmup should be eligible')
        model._fix_plan[0]['phase'] = 'measured'
        require(module._validation_eligibility(model,context,ids)[0] is False,'Measured phase eligible')
        model._fix_plan[0]['phase'] = 'input_warmup'
        require(module._validation_eligibility(model,context,Tensor((999,)))[0] is False,'Missing SHA receipt eligible')
        bad_context = dict(context,rows=[[1,30,40]]*3)
        require(module._validation_eligibility(model,bad_context,ids)[0] is False,'Wrong image count eligible')
        checks.append('numerical gate eligibility rejects measured phase, missing SHA receipt and wrong image count')
        numeric = [json.loads(p.read_text(encoding='utf-8')) for p in receipts.glob('vision_qmax_numeric_*.json')]
        require(len(numeric)==2 and all(d['passed'] and d['status']=='passed' for d in numeric), 'Gate receipts differ')
        require(all(len(d['native_layer_calls']) == len(d['candidate_layer_calls']) == 24 for d in numeric),'Gate layer receipts differ')
        require(all(d['patch_module_sha256']==candidate_hash for d in numeric), 'Gate patch hash differs')
        require(all(d['validation_enabled_env']=='1' and len(d['features'])==4 and
                    all(feature['bitwise_equal'] for feature in d['features']) for d in numeric), 'Numerical feature schema differs')
        checks.append('simulated numerical receipts include native/candidate 24-layer evidence and passed comparison')

    # A new synthetic service permits the same first shape to exercise failure.
    failed_module = load_patch('vision_qmax_gate_failure_fake_qa')
    failure_dir = directory/'gate_failure'
    failure_dir.mkdir()
    failed_receipts = failure_dir/'receipts'
    failed_receipts.mkdir()
    os.environ['PROJECT2_STAGE2_RUN_DIR'] = str(failed_receipts)
    failed_module._compare_features = compare
    with fake_package(failure_dir,failed_module) as (model_type,attention_type,impl_type,kernel):
        failed_module.install()
        model = make_model(model_type,attention_type)
        model._fix_index = 0
        model._fix_plan = [dict(phase='input_warmup',tokens=0,sample=0)]
        model._fix_dir = failed_receipts
        model.packet = packet([[1,30,40]]*2)
        (failed_receipts/'input_700.json').write_text(json.dumps(dict(request_index=0,input_ids={},pixel_values={})),encoding='utf-8')
        args = [Tensor((700,)),object(),object()]
        kwargs = dict(pixel_values=Tensor((2400,1536)),grid_thw=grid([[1,30,40]]*2))
        decision['passed'] = False
        outer = object()
        token = failed_module._CONTEXT.set(outer)
        try:
            reject(lambda:model.forward(*args,**kwargs),'numerical gate failed')
            require(failed_module._CONTEXT.get() is outer and model.language_calls==0,'Failed gate leaked context or ran language')
            failed = json.loads(next(failed_receipts.glob('vision_qmax_numeric_*.json')).read_text(encoding='utf-8'))
            require(failed['status']=='failed' and failed['passed'] is False,'Failed gate receipt missing')
            for offset in (1,2):
                model.visual.fail_on_call = model.visual.call_count+offset
                reject(lambda:model.forward(*args,**kwargs),'synthetic visual failure')
                require(failed_module._CONTEXT.get() is outer,'Visual failure leaked context')
            model.visual.fail_on_call = None
            decision['raise'] = True
            reject(lambda:model.forward(*args,**kwargs),'synthetic comparator failure')
            require(failed_module._CONTEXT.get() is outer,'Comparator failure leaked context')
        finally:
            failed_module._CONTEXT.reset(token)
        checks += ['failed numerical gate writes failed receipt and stops before language forward',
                   'baseline/candidate visual exceptions and comparator exception restore both contexts']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default=SOURCE_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output must be new')
    require('torch' not in sys.modules, 'This probe must start without torch')
    initial_hash = sha(PATCH.read_bytes())
    module = load_patch('vision_qmax_fake_qa')
    static = source_audit(module.SOURCE_SHA256, args.source_root)
    checks = ['installed hashes and AST: grid-only bound, no active autotune, unchanged metadata rules',
              'no direct CUDA kernel package import in candidate install',
              'AST: numerical passed requires metadata/finite/distinct storage/value and uint8 bitwise equality']
    env_names = ['PROJECT2_LATENCY_VISION_QMAX', 'PROJECT2_STAGE2_RUN_DIR', 'PROJECT2_LATENCY_VISION_QMAX_VALIDATE']
    previous_env = {name:os.environ.get(name) for name in env_names}
    cases = []
    try:
        os.environ.pop('PROJECT2_LATENCY_VISION_QMAX', None)
        os.environ['PROJECT2_LATENCY_VISION_QMAX_VALIDATE'] = '0'
        require(module.install() is False, 'Disabled install mutated state')
        checks.append('disabled install is a no-op without importing LMDeploy')
        with tempfile.TemporaryDirectory(prefix='project2_vision_qmax_cpu_') as temp:
            directory = Path(temp)
            receipts = directory/'receipts'
            receipts.mkdir()
            os.environ['PROJECT2_LATENCY_VISION_QMAX'] = '1'
            os.environ['PROJECT2_STAGE2_RUN_DIR'] = str(receipts)
            with fake_package(directory, module) as (model_type, attention_type, impl_type, native_kernel):
                require(module.install() is True and module.install() is True, 'Install not idempotent')
                model = make_model(model_type, attention_type)
                for label, rows in [('two', [[1, 30, 40]]*2), ('nine', [[1, 30, 40]]*9),
                                    ('mixed', [[1, 20, 60], [1, 8, 16], [1, 40, 30]])]:
                    image_grid = grid(rows)
                    model.packet = packet(rows)
                    pixels = Tensor((sum(t*h*w for t,h,w in rows), 1536))
                    inputs = [Tensor((3072,),dtype='torch.int64'), object(), object()]
                    kwargs = dict(pixel_values=pixels, grid_thw=image_grid, attn_metadata=object(),
                                  mrope_position_ids=object(), vis_cu_seqlens=object(),
                                  vis_pos_emb=object(), image_mask=object(), pos_embeds=object(), retained=object())
                    result = model.forward(*inputs, **kwargs)
                    require(result is model.output and module._CONTEXT.get() is None, 'Output/context changed')
                    require(image_grid.tolist_calls == 1, 'Expected one CPU grid conversion per prefill')
                    require(all(a is b for a,b in zip(model.native_seen[:3], inputs)), 'Base positional identity changed')
                    require(all(model.native_seen[3][key] is value for key,value in kwargs.items()), 'Base kwargs identity changed')
                    qmax = max(h*w for t,h,w in rows)
                    for block in model.visual.blocks:
                        seen = block.attn.attention.seen[-1]
                        require(all(a is b for a,b in zip(seen[:5], model.packet)), 'Q/K/V/metadata identity changed')
                        require(seen[5:7] == (None, None) and seen[7] == qmax, 'Changed more than max bound')
                    cases.append(dict(label=label, total_patches=pixels.shape[0], segments=len(rows),
                                      max_q_seqlen=qmax, verified_layers=24))
                checks += ['2/9/mixed image grids: one CPU tolist, all 24 layers verified',
                           'all base inputs, Q/K/V and segment tensor identities preserved',
                           'only native attention max_q_seqlen changes; output identity preserved']
                before = len(list(receipts.glob('*.json')))
                model.forward(*inputs, **kwargs)
                require(len(list(receipts.glob('*.json'))) == before, 'Repeated shape emitted duplicate receipt')
                checks.append('repeat shape receipt suppression and model wrapping idempotence')
                attention = model.visual.blocks[0].attn.attention
                reject(lambda:attention.forward(*model.packet), 'outside its CPU-grid context')
                context = module._grid_context(grid([[1,30,40]]*2), Tensor((2400,1536)), model)
                token = module._CONTEXT.set(context)
                valid_packet = packet([[1,30,40]]*2)
                try:
                    reject(lambda:attention.forward(*valid_packet, max_q_seqlen=2400), 'unexpectedly supplied')
                    reject(lambda:attention.forward(*valid_packet, None, None, 2400), 'unexpectedly supplied')
                    reject(lambda:attention.forward(Tensor((2399,16,80)), *valid_packet[1:]), 'query shape')
                    reject(lambda:attention.forward(valid_packet[0], Tensor((2400,15,80)), *valid_packet[2:]), 'non-self')
                    reject(lambda:attention.forward(*valid_packet[:3], Tensor((1,)), valid_packet[4]), 'segment metadata')
                finally:
                    module._CONTEXT.reset(token)
                checks.append('external calls, explicit positional/keyword max, bad query/KV/segment shapes reject')
                gpu_grid = grid([[1,30,40]], device='cuda')
                reject(lambda:module._grid_context(gpu_grid, Tensor((1200,1536)), model), 'CPU grid')
                require(gpu_grid.tolist_calls == 0, 'GPU grid was read before rejection')
                for bad_grid, pixels, message in [
                    (grid([[1,30,40]], shape=(1,4)), Tensor((1200,1536)), 'shape'),
                    (grid([]), Tensor((0,1536)), 'values'),
                    (grid([[1,0,40]]), Tensor((0,1536)), 'values'),
                    (grid([[1,30.0,40]]), Tensor((1200,1536)), 'values'),
                    (grid([[2,30,40]]), Tensor((2400,1536)), 'static images'),
                    (grid([[1,30,40]]), Tensor((1199,1536)), 'patch count')]:
                    reject(lambda g=bad_grid,p=pixels:module._grid_context(g,p,model), message)
                checks.append('CPU grid requirement, positive integer/static-grid and total-patch validation')
                outer = {'outer':'retained'}
                token = module._CONTEXT.set(outer)
                try:
                    model.fail_native = True
                    reject(lambda:model.forward(*inputs, **kwargs), 'synthetic native failure')
                    require(module._CONTEXT.get() is outer, 'Context not restored after native error')
                    model.fail_native = False
                    model.forward(*inputs, **kwargs)
                    require(module._CONTEXT.get() is outer, 'Context not restored after success')
                finally:
                    module._CONTEXT.reset(token)
                count = sum(len(b.attn.attention.seen) for b in model.visual.blocks)
                require(model.forward(*inputs) is model.output, 'Decode bypass changed')
                require(model.forward(*inputs, pixel_values=Tensor((1,)), inputs_embeds=object()) is model.output,
                        'Existing embeddings bypass changed')
                require(sum(len(b.attn.attention.seen) for b in model.visual.blocks) == count, 'Bypass ran vision')
                checks += ['nested ContextVar restored after native success and exception',
                           'decode and externally supplied embeddings bypass unchanged']
                model.packet = packet([[1,8,8]])
                model.execute_layers = 23
                reject(lambda:model.forward(*inputs, pixel_values=Tensor((64,1536)), grid_thw=grid([[1,8,8]])),
                       'each original layer once')
                require(module._CONTEXT.get() is None, 'Missing-layer rejection leaked context')
                checks.append('first unseen shape requires exactly all 24 layers')
                for change, message in [('causal', 'noncausal'), ('kernel', 'already been replaced'), ('type', 'module/backend')]:
                    fresh = attention_type(0)
                    if change == 'causal': fresh.impl.causal = True
                    if change == 'kernel': fresh.impl.flash_attention_fwd = lambda:None
                    if change == 'type': fresh.impl = object()
                    reject(lambda f=fresh:module._wrap_attention(f,0,id(model),attention_type,impl_type,native_kernel),message)
                checks.append('incompatible backend, causal mode, and replaced kernel fail closed')
                data = [json.loads(p.read_text(encoding='utf-8')) for p in receipts.glob('*.json')]
                installs = [d for d in data if d['kind'] == 'install_receipt']
                executions = [d for d in data if d['kind'] == 'first_shape_execution_receipt']
                require(len(installs) == 1 and len(executions) == 3, 'Receipt count differs')
                for item in data:
                    require(item['patch_module_sha256'] == initial_hash and item['original_source_sha256'] == module.SOURCE_SHA256,
                            'Receipt source hashes differ')
                for item in executions:
                    require(item['cpu_grid_reused'] and item['regular_path_device_scalar_reads_added'] == 0 and item['regular_path_added_cuda_synchronizations'] == 0,
                            'Receipt scope differs')
                    require([c['layer'] for c in item['actual_layer_calls']] == list(range(24)), 'Receipt layers differ')
                checks.append('simulated receipt schema: one install plus one 24-layer execution per distinct grid')
            gate_test(directory,initial_hash,checks)
    finally:
        for name, value in previous_env.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
    require('torch' not in sys.modules, 'Torch was imported')
    require(sha(PATCH.read_bytes()) == initial_hash, 'Candidate changed during probe')
    report = dict(passed=True, scope='Stdlib fake-object control-flow QA; no real model, kernel or torch import',
                  candidate_sha256=initial_hash, probe_sha256=sha(Path(__file__).read_bytes()),
                  torch_imported=False, cuda_executed=False, checks=checks, cases=cases, installed_source_audit=static,
                  limitations=['Synthetic package replaces source guards only inside the fake fixture; real installed hashes are audited separately.',
                               'Fake receipts are temporary and not real execution receipts.',
                               'The torch feature comparator is replaced by a controlled stub; its pass criteria are checked statically only.',
                               'Real numerical equivalence and performance require the separately authorized GPU study.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, checks=len(checks), output=str(args.output), candidate_sha256=initial_hash)))


if __name__ == '__main__':
    main()
