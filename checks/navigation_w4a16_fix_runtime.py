"""Native FP16/W4A16 forwards with optional decode graphs and external timing.

The original model, packed checkpoints, KV allocation, image preparation and
mRoPE buffers are retained. No DART state or per-token audit hooks enter graphs.
All CUDA events surrounding a graph replay are recorded outside its capture.
Request timing is flushed at the next image prefill: clients must send one tail
request after their final measured request. Initial input SHA work is warmup only.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import time
from types import MethodType

import torch
from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner
from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLForConditionalGeneration
from navigation_stage2_runtime import Stage2Qwen3VLForConditionalGeneration
from navigation_awq_dart_guard import execution_receipt

VARIANT = os.environ.get('PROJECT2_W4A16_FIX_VARIANT', 'stock')
GRAPH = os.environ.get('PROJECT2_W4A16_FIX_GRAPH', 'off')
assert GRAPH in ('off', 'on')


def _install_candidate():
    if VARIANT == 'stock':
        return
    import lmdeploy.pytorch.backends.cuda.awq_modules as backend
    from navigation_w4a16_candidate_kernels import VARIANTS, precision_linear
    assert VARIANT in VARIANTS

    def candidate(x, qweight, qzeros, scales, w_bit=4, group_size=128,
                  bias=None, out_features=0):
        assert w_bit == 4 and group_size == 128 and x.dtype == torch.float16
        assert out_features == scales.size(1)
        shape = x.shape[:-1] + (out_features,)
        result = precision_linear(x.flatten(0, -2).contiguous(), qweight, scales, qzeros,
                                  variant=VARIANT)
        if bias is not None:
            result = result + bias
        result = result.reshape(shape)
        return result.unsqueeze(0) if result.ndim == 2 else result

    backend.wq_gemm_forward = candidate


_install_candidate()
_original_graph_call = CUDAGraphRunner.__call__


def _observed_graph_call(runner, **kwargs):
    model = runner.model
    if not isinstance(model, W4A16FixModel):
        return _original_graph_call(runner, **kwargs)
    graph_enabled = bool(runner.enable_graph(**kwargs))
    before_keys = set(runner._runner_map)
    active = model._fix_before_step(kwargs)
    if active:
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        started = time.perf_counter()
    output = _original_graph_call(runner, **kwargs)
    mode = ('capture' if set(runner._runner_map) != before_keys else 'replay') if graph_enabled else 'eager'
    model._fix_graph_totals[mode] += 1
    if active:
        end.record()
        step = model._fix_current_step
        step['timers']['model'] = (begin, end, (time.perf_counter()-started)*1000)
        step['execution_mode'] = mode
        step['graph_keys'] = [list(k) for k in runner._runner_map]
    elif mode == 'capture':
        model._fix_write('startup_graph_capture.json', dict(graph_keys=[list(k) for k in runner._runner_map],
            actual_capture_observed=True, counts=dict(model._fix_graph_totals)))
    return output


CUDAGraphRunner.__call__ = _observed_graph_call


class W4A16FixModel(Stage2Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        self._fix_graph = GRAPH
        self._fix_variant = VARIANT
        super().__init__(*args, **kwargs)
        assert self._dart_keep == 1.0 and self._method in ('fp16', 'awq', 'gptq')
        assert self._is_quant or VARIANT == 'stock'
        self._fix_dir = Path(os.environ['PROJECT2_STAGE2_RUN_DIR'])
        plan = self._fix_dir/'request_plan.json'
        self._fix_plan = json.loads(plan.read_text()) if plan.exists() else []
        self._fix_index = -1
        self._fix_steps = []
        self._fix_current_step = None
        self._fix_seen_inputs = set()
        self._fix_graph_totals = Counter()
        self._fix_python_forwards = 0
        self._fix_execution_verified = False
        for name, module in [('vision', self.visual), ('language', self.language_model)]:
            original = module.forward

            def wrapped(part, *a, _name=name, _original=original, **kw):
                step = self._fix_current_step
                if step is None or step['phase'] != 'prefill':
                    return _original(*a, **kw)
                return self._fix_time(_name, lambda: _original(*a, **kw))

            module.forward = MethodType(wrapped, module)

    def _write_json(self, stem, data):
        # Stage2 construction retains its metadata/loader audit but its stated
        # execution scope must reflect this independent native diagnostic.
        if data.get('event') == 'runtime_initialized':
            data = dict(data, scope='native full-prefill3072; optional native decode CUDA graph',
                        graph_requested=GRAPH, variant=VARIANT,
                        native_forward_without_stage2_state=True)
        return super()._write_json(stem, data)

    def _fix_write(self, filename, value):
        with (self._fix_dir/filename).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')

    def support_cuda_graph(self, *args, **kwargs):
        return self._fix_graph == 'on' and Qwen3VLForConditionalGeneration.support_cuda_graph(self, *args, **kwargs)

    def update_model_metas(self, past_key_values, inputs_embeds=None, context=None):
        result = Qwen3VLForConditionalGeneration.update_model_metas(self, past_key_values, inputs_embeds, context)
        context.model_metas = result
        return result

    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None, **kwargs):
        self._fix_python_forwards += 1
        first = not self._fix_execution_verified
        if first and self._is_quant:
            assert self._awq_loaded_weights == self._awq_expected_weights
            self._awq_calls = {}
        result = Qwen3VLForConditionalGeneration.forward(self, input_ids, position_ids, past_key_values,
            attn_metadata=attn_metadata, **kwargs)
        if first:
            receipt = None
            if self._is_quant:
                receipt = execution_receipt(self._awq_calls, self._awq_inventory, self._awq_loaded_weights,
                    self._awq_expected_weights, input_ids.numel(), input_ids.numel())
            for hook in self._awq_hooks:
                hook.remove()
            self._awq_hooks = []
            self._awq_calls = None
            self._fix_execution_verified = True
            self._fix_write('execution_audit.json', dict(group=self._method, variant=VARIANT,
                graph_requested=GRAPH, first_native_forward_verified=True,
                scope='One-time synthetic engine warmup; per-token hooks removed before graph capture',
                native_awq_receipt=receipt, native_forward_preserved=True))
        return result

    def _fix_before_step(self, kwargs):
        context = self.ctx_mgr.current_context()
        if not context.is_decoding:
            if kwargs.get('pixel_values') is None:
                return False  # synthetic engine startup, not a client request
            self._fix_flush()
            self._fix_index += 1
            length = kwargs['input_ids'].numel()
            if length not in self._fix_seen_inputs:
                evidence = {}
                for name in ('input_ids', 'pixel_values'):
                    raw = kwargs[name].detach().cpu().contiguous()
                    evidence[name] = dict(shape=list(raw.shape), dtype=str(raw.dtype),
                        sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())
                self._fix_write(f'input_{length}.json', dict(request_index=self._fix_index,
                    scope='First occurrence only; client must classify this request as warmup', **evidence))
                self._fix_seen_inputs.add(length)
        if self._fix_index < 0:
            return False
        self._fix_current_step = dict(phase='decode' if context.is_decoding else 'prefill',
            rows=kwargs['input_ids'].numel(), timers={})
        self._fix_steps.append(self._fix_current_step)
        return True

    def _fix_time(self, name, fn):
        step = self._fix_current_step
        if step is None:
            return fn()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        started = time.perf_counter()
        value = fn()
        end.record()
        step['timers'][name] = (begin, end, (time.perf_counter()-started)*1000)
        return value

    def get_logits(self, hidden_states):
        result = self._fix_time('logits', lambda: Qwen3VLForConditionalGeneration.get_logits(self, hidden_states))
        self._fix_current_step = None
        return result

    def _fix_flush(self):
        if not self._fix_steps:
            return
        # This is the next request, after the previous HTTP response; never a
        # synchronization inside decode or a CUDA-graph capture.
        torch.cuda.synchronize()
        rows = []
        for step in self._fix_steps:
            row = {key:value for key,value in step.items() if key != 'timers'}
            row['timers'] = {key:dict(cuda_interval_ms=a.elapsed_time(b), host_wall_ms=wall)
                for key,(a,b,wall) in step['timers'].items()}
            if 'logits' in step['timers']:
                row['model_through_logits_ms'] = step['timers']['model'][0].elapsed_time(step['timers']['logits'][1])
            rows.append(row)
        labels = self._fix_plan[self._fix_index] if isinstance(self._fix_plan, list) and self._fix_index < len(self._fix_plan) else None
        self._fix_write(f'timing_{self._fix_index:03d}.json', dict(index=self._fix_index, plan=labels,
            group=self._method, variant=VARIANT, graph_requested=GRAPH, steps=rows,
            logical_forward_counts=dict(Counter(r['phase'] for r in rows)),
            actual_execution_modes=dict(Counter(r['execution_mode'] for r in rows)),
            graph_counts_since_startup=dict(self._fix_graph_totals),
            python_forward_invocations_since_startup=self._fix_python_forwards,
            capture_replay_observed_outside_graph=True,
            timing_scope='CUDA intervals include dispatch/idle; graph decode measured around actual replay, logits separately'))
        self._fix_steps = []

