"""Separate fixed-input diagnostic; never modifies the frozen stage2 adapter.

CUDA event intervals include GPU idle gaps caused by host launch/validation work.
They are not sums of kernel execution time. Events are resolved at the next
request, after the preceding HTTP response has completed (no per-token sync).
"""
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType

import torch
from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLForConditionalGeneration
from navigation_stage2_runtime import Stage2Qwen3VLForConditionalGeneration
from navigation_awq_dart_guard import inspect_awq_modules


class QuantTimingModel(Stage2Qwen3VLForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._qt_dir = Path(os.environ['PROJECT2_STAGE2_RUN_DIR'])
        self._qt_mode = os.environ['PROJECT2_QUANT_TIMING_MODE']
        self._qt_plan = json.loads((self._qt_dir / 'request_plan.json').read_text())
        self._qt_index, self._qt_steps, self._qt_active = -1, [], None
        self._qt_profiler = None
        if self._qt_mode == 'native':
            for hook in self._awq_hooks:
                hook.remove()
        for name, module in [('vision', self.visual), ('language', self.language_model)]:
            original = module.forward
            def wrapped(part, *a, _name=name, _original=original, **kw):
                return self._qt_timed(_name, lambda: _original(*a, **kw))
            module.forward = MethodType(wrapped, module)

    def _qt_timed(self, name, fn):
        if self._qt_active is None:
            return fn()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        with torch.profiler.record_function('quant_diag/' + self._qt_active['phase'] + '/' + name):
            result = fn()
        end.record()
        self._qt_active['timers'][name] = (begin, end, (time.perf_counter() - start) * 1000)
        return result

    def _qt_flush(self):
        if not self._qt_steps:
            return
        torch.cuda.synchronize()
        if self._qt_profiler is not None:
            self._qt_profiler.stop()
            events = self._qt_profiler.events()
            kernels = {}
            for event in events:
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    item = kernels.setdefault(event.name, {'count': 0, 'total_us': 0.0})
                    item['count'] += 1
                    item['total_us'] += event.time_range.elapsed_us()
            self._qt_write(f'profile_{self._qt_index:03d}.json', dict(
                kernels=sorted([dict(name=k, **v) for k, v in kernels.items()], key=lambda x: -x['total_us']),
                table=self._qt_profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)))
            self._qt_profiler.export_chrome_trace(str(self._qt_dir / f'profile_{self._qt_index:03d}.trace.json'))
            self._qt_profiler = None
        rows = []
        for step in self._qt_steps:
            row = {key: value for key, value in step.items() if key != 'timers'}
            row['timers'] = {key: dict(cuda_interval_ms=a.elapsed_time(b), host_wall_ms=wall)
                             for key, (a, b, wall) in step['timers'].items()}
            if 'logits' in step['timers']:
                row['model_through_logits_ms'] = step['timers']['model'][0].elapsed_time(step['timers']['logits'][1])
            rows.append(row)
        self._qt_write(f'timing_{self._qt_index:03d}.json', dict(index=self._qt_index,
            plan=self._qt_plan[self._qt_index], mode=self._qt_mode, steps=rows))
        self._qt_steps = []

    def _qt_write(self, filename, value):
        with (self._qt_dir / filename).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)

    def update_model_metas(self, past_key_values, inputs_embeds=None, context=None):
        if self._qt_mode == 'native':
            result = Qwen3VLForConditionalGeneration.update_model_metas(self, past_key_values, inputs_embeds, context)
            context.model_metas = result
            return result
        return super().update_model_metas(past_key_values, inputs_embeds, context)

    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None, **kwargs):
        ctx = self.ctx_mgr.current_context()
        native = Qwen3VLForConditionalGeneration.forward if self._qt_mode == 'native' else Stage2Qwen3VLForConditionalGeneration.forward
        # LMDeploy performs synthetic text-only engine warmup before HTTP readiness.
        # It is not one of the frozen image requests and must not consume a plan row.
        if self._qt_index == -1 and kwargs.get('pixel_values') is None:
            return native(self, input_ids, position_ids, past_key_values, attn_metadata=attn_metadata, **kwargs)
        if not ctx.is_decoding:
            self._qt_flush()
            self._qt_index += 1
            assert self._qt_index < len(self._qt_plan)
            if self._is_quant:
                assert self._awq_loaded_weights == self._awq_expected_weights
            plan = self._qt_plan[self._qt_index]
            assert input_ids.numel() == (701 if plan['sample'] == 0 else 2815)
            if plan['phase'] == 'profile':
                self._qt_profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                self._qt_profiler.start()
            if plan['phase'] == 'warmup' and plan['repeat'] == 0:
                evidence = {}
                for name, tensor in [('input_ids', input_ids), ('pixel_values', kwargs['pixel_values'])]:
                    raw = tensor.detach().cpu().contiguous()
                    evidence[name] = dict(shape=list(raw.shape), dtype=str(raw.dtype),
                        sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())
                self._qt_write(f'input_{plan["sample"]}.json', evidence)
                if self._is_quant:
                    inspect_awq_modules(self)
        self._qt_active = dict(phase='decode' if ctx.is_decoding else 'prefill',
            rows=input_ids.numel(), timers={})
        result = self._qt_timed('model', lambda: native(self, input_ids, position_ids, past_key_values,
            attn_metadata=attn_metadata, **kwargs))
        self._qt_steps.append(self._qt_active)
        return result

    def get_logits(self, hidden_states):
        result = self._qt_timed('logits', lambda: Qwen3VLForConditionalGeneration.get_logits(self, hidden_states))
        self._qt_active = None
        return result
