"""Opt-in exact-M W4 launch adapter for independent latency services.

Synthetic or otherwise unmeasured M retains the historical frozen baseline.
All-phase mode uses candidates for measured M=1/701/2815. Prefill-only mode
keeps the original baseline for M1 and uses candidates for M701/2815.
Python call counts never pretend to count CUDA graph replayed kernel calls;
capture coverage and the existing outside-graph replay evidence are separate.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch

from navigation_repair_runtime import NavigationRepairModel


MANDATORY_ROWS = (1, 701, 2815)


class NavigationLatencyW4Model(NavigationRepairModel):
    def __init__(self, *args, **kwargs):
        self._wide_phase = os.environ.get('PROJECT2_LATENCY_W4_PHASE', 'all')
        if self._wide_phase not in ('all', 'prefill'):
            raise ValueError('PROJECT2_LATENCY_W4_PHASE must be all or prefill')
        self._wide_installed = False
        self._wide_calls = Counter()
        self._wide_capture_calls = Counter()
        self._wide_request_start = Counter()
        self._wide_request_started = False
        super().__init__(*args, **kwargs)
        if not self._is_quant or self._method not in ('awq', 'gptq'):
            raise ValueError('Wide W4 adapter requires the existing AWQ/GPTQ W4A16 model')
        if (int(os.environ.get('PROJECT2_REPAIR_WBITS', '0')),
                int(os.environ.get('PROJECT2_REPAIR_ABITS', '0'))) != (4, 16):
            raise ValueError('Wide adapter does not implement activation quantization')

    def forward(self, *args, **kwargs):
        first = not self._fix_execution_verified
        # Let the original first synthetic forward install and verify the
        # original frozen baseline. The candidate replaces it before the engine
        # performs decode warmup/capture, without changing old runtime code.
        result = super().forward(*args, **kwargs)
        if first:
            if self._wide_installed:
                raise RuntimeError('Wide adapter installed more than once')
            self._install_wide()
        return result

    def _install_wide(self):
        import navigation_latency_w4_kernels_20261120 as wide
        import lmdeploy.pytorch.backends.cuda.awq_modules as backend

        table_path = Path(os.environ['PROJECT2_LATENCY_W4_TABLE']).resolve()
        launch = json.loads((self._fix_dir/'launch.json').read_text(encoding='utf-8'))
        receipt = wide.load_table(table_path, self._method, launch['model_weights_sha256'])
        table_shapes = {(row['m'], row['k'], row['n']) for row in receipt['shapes']}
        objects = self._awq_objects
        if len(objects) != 144 or self._awq_loaded_weights != self._awq_expected_weights:
            raise ValueError('Require all 144 verified frozen language projections')
        by_pointer = {}
        actual_shapes = set()
        for name, module in objects.items():
            k, n = module.qweight.shape[0], module.scales.shape[1]
            actual_shapes.add((k, n))
            for m in MANDATORY_ROWS:
                if (m, k, n) not in table_shapes:
                    raise ValueError(f'Missing required measured configuration: {name}, M={m}')
            pointer = module.qweight.data_ptr()
            if pointer in by_pointer:
                raise ValueError('Expected independently allocated frozen projection weights')
            by_pointer[pointer] = name
            wide.prepare_projection(module.qweight, module.scales, module.qzeros)
        if len(actual_shapes) != 4:
            raise ValueError('Expected exactly four fused projection layouts')
        frozen = wide.freeze()
        fallback = backend.wq_gemm_forward

        def dispatch(x, qweight, qzeros, scales, w_bit=4, group_size=128,
                     bias=None, out_features=0):
            if w_bit != 4 or group_size != 128 or out_features != scales.shape[1]:
                raise ValueError('Native W4A16 protocol differs')
            name = by_pointer.get(qweight.data_ptr())
            if name is None:
                raise ValueError('Unregistered projection in wide adapter')
            flat = x.flatten(0, -2).contiguous()
            m, k, n = flat.shape[0], flat.shape[1], scales.shape[1]
            key = (m, k, n)
            baseline_decode = m == 1 and self._wide_phase == 'prefill'
            use_candidate = key in table_shapes and not baseline_decode
            if m in MANDATORY_ROWS and not use_candidate and not baseline_decode:
                raise RuntimeError('Measured M must never silently fall back')
            route = 'candidate' if use_candidate else 'baseline_decode' if baseline_decode else 'baseline_fallback'
            self._wide_calls[(route, m, name)] += 1
            if m == 1 and torch.cuda.is_current_stream_capturing():
                self._wide_capture_calls[(route, name)] += 1
            if not use_candidate:
                return fallback(x, qweight, qzeros, scales, w_bit=w_bit,
                                group_size=group_size, bias=bias, out_features=out_features)
            shape = x.shape[:-1] + (out_features,)
            result = wide.precision_linear(flat, qweight, scales, qzeros)
            if bias is not None:
                result = result + bias
            result = result.reshape(shape)
            return result.unsqueeze(0) if result.ndim == 2 else result

        backend.wq_gemm_forward = dispatch
        self._wide_installed = True
        self._wide_module_names = set(objects)
        self._fix_write('wide_launch_install.json', dict(
            algorithm=self._method, checkpoint_sha256=launch['model_weights_sha256'],
            candidate_table_path=str(table_path),
            candidate_table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
            phase=self._wide_phase, mandatory_table_m=list(MANDATORY_ROWS),
            mandatory_candidate_m=list(MANDATORY_ROWS if self._wide_phase == 'all' else (701,2815)),
            decode_route='candidate' if self._wide_phase == 'all' else 'baseline_decode',
            actual_projection_count=len(objects),
            fallback_scope='Only M absent from saved exact-M table; mandatory M cannot fall back',
            original_first_synthetic_forward='baseline, before candidate install',
            cold_compile_guard_enabled=True, table=frozen))

    def _fix_before_step(self, kwargs):
        # The parent flushes the previous request before starting a new prefill.
        value = super()._fix_before_step(kwargs)
        if value and self._fix_current_step['phase'] == 'prefill':
            if not self._wide_installed:
                raise RuntimeError('Client prefill before wide adapter install')
            self._wide_request_start = self._wide_calls.copy()
            self._wide_request_started = True
        return value

    def _fix_flush(self):
        if not self._fix_steps:
            return super()._fix_flush()
        index = self._fix_index
        steps = list(self._fix_steps)
        delta = self._wide_calls - self._wide_request_start
        prefill_ms = [step['rows'] for step in steps if step['phase'] == 'prefill']
        expected_capture_route = 'candidate' if self._wide_phase == 'all' else 'baseline_decode'
        captures = {name: count for (route, name), count in self._wide_capture_calls.items()
                    if route == expected_capture_route}
        bad_capture = {name: count for (route, name), count in self._wide_capture_calls.items()
                       if route != expected_capture_route}
        for m in prefill_ms:
            if m in MANDATORY_ROWS:
                expected_route='baseline_decode' if m==1 and self._wide_phase=='prefill' else 'candidate'
                covered = {name for (route, rows, name), count in delta.items()
                           if route == expected_route and rows == m and count > 0}
                if covered != self._wide_module_names:
                    raise RuntimeError(f'Incomplete {expected_route} prefill coverage for M={m}')
                if any(route != expected_route and rows == m for route, rows, name in delta):
                    raise RuntimeError('Measured prefill used unexpected route')
        modes = Counter(step.get('execution_mode') for step in steps)
        if modes['replay']:
            if set(captures) != self._wide_module_names or bad_capture:
                raise RuntimeError(f'Graph replay lacks full {expected_capture_route} capture coverage')
        super()._fix_flush()
        self._fix_write(f'wide_execution_{index:03d}.json', dict(
            index=index, prefill_m=prefill_ms, phase=self._wide_phase,
            python_projection_calls=[dict(route=route, m=m, module=name, calls=count)
                                     for (route,m,name),count in sorted(delta.items())],
            candidate_capture_projection_calls={name:count for (route,name),count in self._wide_capture_calls.items() if route=='candidate'},
            baseline_decode_capture_projection_calls={name:count for (route,name),count in self._wide_capture_calls.items() if route=='baseline_decode'},
            expected_decode_route=expected_capture_route,
            unexpected_capture_projection_calls=bad_capture,
            outside_graph_execution_modes=dict(modes),
            graph_replay_python_counters_increment=False,
            graph_evidence='Capture coverage proves stated decode route in graph; replay count comes from existing outside-graph observer',
            mandatory_prefill_route_coverage_passed=True))

