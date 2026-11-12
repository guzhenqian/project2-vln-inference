"""Numerical/PTX/performance screening for isolated precision repair kernels.

This is NOT a navigation-quality gate. Inputs are deterministic synthetic
activations with frozen real checkpoint projections; A4's new scale recipe is
declared, never selected using the20-case navigation acceptance set. The old
kernel is timed only as a local implementation baseline. Defaults validate
three layers but time layer0 only. All artifacts go to a new output directory.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
FUSED = {
    'qkv': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
    'o': ['self_attn.o_proj'],
    'gate_up': ['mlp.gate_proj', 'mlp.up_proj'],
    'down': ['mlp.down_proj'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    # This new directory belongs to this process. Repeated progress snapshots
    # are atomic; historical experiment directories are never opened for writing.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def unpack(q, bits, torch):
    shifts = [0, 16, 4, 20, 8, 24, 12, 28] if bits == 4 else [0, 8, 16, 24]
    return torch.stack([(q >> i) & ((1 << bits) - 1) for i in shifts], -1).reshape(q.shape[0], -1)


def pack(q, bits, torch):
    order = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else [0, 1, 2, 3]
    grouped = q.reshape(q.shape[0], -1, len(order))[:, :, order]
    shift = torch.arange(len(order), device=q.device, dtype=torch.int32) * bits
    return (grouped << shift).sum(-1, dtype=torch.int32)


def activation_reference(x, bits, kernel, torch):
    if bits == 16:
        return x.float(), dict(packing='not_applicable')
    m, k = x.shape
    limit = 2 ** (bits - 1) - 1
    shaped = x.float().reshape(m, k // 128, 128) if bits == 4 else x.float()
    # Use multiplication to preserve the declared FP32 reciprocal convention.
    scale = (shaped.abs().amax(-1) * (1.0 / limit)).clamp_min(1.e-8)
    integers = (shaped / scale[..., None]).round().clamp(-limit, limit)
    actual, actual_scale = kernel.quantize_activations(x, bits)
    if bits == 4:
        actual = torch.stack([(actual >> i) & 15 for i in range(0, 32, 4)], -1).reshape(x.shape)
        actual = torch.where(actual >= 8, actual - 16, actual)
    if not torch.equal(actual.reshape(shaped.shape).float(), integers):
        raise AssertionError('Activation integer/packing mismatch')
    if not torch.equal(actual_scale, scale):
        raise AssertionError('Activation scale mismatch')
    dequantized = (integers * scale[..., None]).reshape(x.shape)
    error = float((dequantized - x.float()).norm() / x.float().norm().clamp_min(1.e-8))
    return dequantized, dict(packing='exact', scales='exact', relative_l2_vs_fp16_input=error,
                             zero_fraction=float((integers == 0).float().mean()),
                             scale_shape=list(scale.shape))


def reference_output(x, qw, scales, qz, wb, ab, kernel, torch):
    a, activation = activation_reference(x, ab, kernel, torch)
    integers = unpack(qw, wb, torch) - unpack(qz, wb, torch).repeat_interleave(128, 0)
    if ab == 16:
        dense = (integers.half() * scales.repeat_interleave(128, 0)).float()
    else:
        # Integer group partials are scaled in FP32, with no FP16 weight-rounding.
        dense = integers.float() * scales.float().repeat_interleave(128, 0)
    expected = a @ dense
    del dense, integers, a
    return expected, activation


def timed(fn, torch, repeats, warmups=3):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    events = []
    start = time.perf_counter()
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        events.append((begin, end))
    torch.cuda.synchronize()
    values = [begin.elapsed_time(end) for begin, end in events]
    return dict(median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values),
                samples_ms=values, wall_mean_ms=1000 * (time.perf_counter() - start) / repeats)


def graph_timed(fn, torch, repeats):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    graph.replay()
    torch.cuda.synchronize()
    result = timed(graph.replay, torch, repeats, warmups=2)
    del captured, graph
    return result


def ptx_evidence(x, qw, scales, qz, wb, ab, variant, kernel, torch):
    """Read the compiled handle from the exact launch, not an unrelated cache entry."""
    import triton
    m, k = x.shape
    n = scales.shape[1]
    a, act_scales = (x, x) if ab == 16 else kernel.quantize_activations(x, ab)
    cfg = kernel.launch_config(m, k, variant)
    split = cfg['split_k']
    output = torch.empty((split, m, n) if split > 1 else (m, n), device=x.device,
                         dtype=torch.float32 if split > 1 else torch.float16)
    compiled = kernel._linear_repair[(triton.cdiv(m, cfg['bm']) * triton.cdiv(n, cfg['bn']), split)](
        a, act_scales, qw, qz, scales, output, m, k, n, wb, ab, cfg['bm'], cfg['bn'], split,
        num_warps=cfg['num_warps'], num_stages=cfg['num_stages'])
    ptx = compiled.asm.get('ptx', '')
    required = '.s8.s8.' if ab < 16 else '.f16.f16.'
    if required not in ptx:
        raise AssertionError('Required actual MMA instruction missing: ' + required)
    return dict(required_instruction=required, verified=True,
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                mma_instructions=sorted({line.strip() for line in ptx.splitlines() if 'mma.sync' in line}),
                launch_config=cfg)


def summarize_speed(records, variants):
    summaries = []
    groups = sorted({r['group'] for r in records if 'timings' in r})
    for group in groups:
        for variant in variants:
            ratios = {'decode': [], 'prefill': []}
            for row in records:
                if row['group'] != group or 'timings' not in row:
                    continue
                phase = 'decode' if row['rows'] <= 16 else 'prefill'
                mode = 'graph' if phase == 'decode' else 'eager'
                times = row['timings']
                if mode not in times['old'] or mode not in times[variant]['full']:
                    continue
                ratios[phase].append(times[variant]['full'][mode]['median_ms'] /
                                     times['old'][mode]['median_ms'])
            phase_result = {phase: dict(count=len(values), median_ratio=statistics.median(values),
                                       max_ratio=max(values)) for phase, values in ratios.items() if values}
            # Predeclared screening only: avoid carrying a clearly slower kernel
            # into fullmodel testing; no statistical speedup claim is made.
            complete = set(phase_result) == {'decode', 'prefill'}
            acceptable = complete and all(v['median_ratio'] <= 1.05 and v['max_ratio'] <= 1.20
                                          for v in phase_result.values())
            summaries.append(dict(group=group, variant=variant, phase=phase_result,
                local_speed_screen_pass=acceptable, complete_two_phase_screen=complete,
                same_activation_recipe_as_old=not group.endswith('w4a4'),
                caveat='A4 recipe changed; ratios compare implementations, not identical arithmetic.'
                       if group.endswith('w4a4') else 'Hot isolated projection screen; not fullmodel speed.'))
    return summaries


def prepare_all(args, groups, kernel, safe_open, torch):
    """Prepare selected geometry families once, then prohibit private JIT misses."""
    prepared, seen = [], set()
    modes = sorted({(groups[g]['weight_bits'], groups[g]['activation_bits']) for g in args.groups})
    for wb, ab in modes:
        qw = torch.zeros((640, 192 // (32 // wb)), device='cuda', dtype=torch.int32)
        qz = torch.zeros((5, 192 // (32 // wb)), device='cuda', dtype=torch.int32)
        scales = torch.ones((5, 192), device='cuda', dtype=torch.float16)
        for variant in args.variants:
            prepared.append(kernel.prepare_projection(qw, scales, qz, wb, ab, variant))
        del qw, qz, scales
    for group in args.groups:
        spec = groups[group]
        wb, ab = spec['weight_bits'], spec['activation_bits']
        with safe_open(str(ROOT / spec['model'] / 'model.safetensors'), framework='pt', device='cpu') as reader:
            for layer in args.layers:
                for projection in args.projections:
                    parts = FUSED[projection]
                    prefixes = [f'model.language_model.layers.{layer}.{part}' for part in parts]
                    shapes = [reader.get_slice(prefix + '.scales').get_shape() for prefix in prefixes]
                    geometry = (wb, ab, shapes[0][0] * 128, sum(shape[1] for shape in shapes))
                    if geometry in seen:
                        continue
                    seen.add(geometry)
                    qw, qz, scales = [torch.cat([reader.get_tensor(prefix + '.' + suffix)
                                                for prefix in prefixes], dim=1).contiguous().cuda()
                                      for suffix in ('qweight', 'qzeros', 'scales')]
                    for variant in args.variants:
                        prepared.append(kernel.prepare_projection(qw, scales, qz, wb, ab, variant))
                    print(json.dumps(dict(stage='prepare', group=group, layer=layer,
                                          projection=projection, geometry=geometry)), flush=True)
                    del qw, qz, scales
    frozen = kernel.freeze()
    if not frozen['cache_unchanged_since_freeze']:
        raise AssertionError('Unexpected cache mutation while freezing')
    return dict(preparations=prepared, frozen_snapshot=frozen)


def rejection_checks(kernel, torch, wb, ab, variant):
    """Verify bad pointers/geometries and a forced private cache miss fail closed."""
    import triton
    m, k, n = 1, 640, 192
    qw = torch.zeros((k, n // (32 // wb)), device='cuda', dtype=torch.int32)
    qz = torch.zeros((k // 128, n // (32 // wb)), device='cuda', dtype=torch.int32)
    scales = torch.ones((k // 128, n), device='cuda', dtype=torch.float16)
    cases = {}
    misaligned = torch.empty(k + 1, device='cuda', dtype=torch.float16)[1:].view(m, k)
    try:
        kernel.precision_linear(misaligned, qw, scales, qz, wb, ab, variant)
    except ValueError as error:
        if 'aligned' not in str(error):
            raise
        cases['misaligned_pointer_rejected'] = True
    else:
        raise AssertionError('Misaligned pointer was accepted')
    x = torch.zeros((m, k), device='cuda', dtype=torch.float16)
    unknown_qw = torch.zeros((768, n // (32 // wb)), device='cuda', dtype=torch.int32)
    unknown_qz = torch.zeros((6, n // (32 // wb)), device='cuda', dtype=torch.int32)
    unknown_scales = torch.ones((6, n), device='cuda', dtype=torch.float16)
    unknown_x = torch.zeros((m, 768), device='cuda', dtype=torch.float16)
    try:
        kernel.precision_linear(unknown_x, unknown_qw, unknown_scales, unknown_qz, wb, ab, variant)
    except RuntimeError as error:
        if 'Unprepared' not in str(error):
            raise
        cases['unknown_geometry_rejected'] = True
    else:
        raise AssertionError('Unknown frozen geometry was accepted')
    strided_qw = torch.zeros((k, qw.shape[1] + 1), device='cuda', dtype=torch.int32)[:, :qw.shape[1]]
    try:
        kernel.precision_linear(x, strided_qw, scales, qz, wb, ab, variant)
    except ValueError as error:
        if 'contract' not in str(error):
            raise
        cases['unprepared_stride_rejected'] = True
    else:
        raise AssertionError('Noncontiguous weights were accepted')
    cfg = kernel.launch_config(m, k, variant)
    split = cfg['split_k']
    a, act_scales = (x, x) if ab == 16 else kernel.quantize_activations(x, ab)
    output = torch.empty((split, m, n), device='cuda', dtype=torch.float32)
    try:
        # Stage3 was deliberately never prepared for the small-M bucket. The
        # hook must raise before compiler work, even for direct private calls.
        kernel._linear_repair[(triton.cdiv(m, cfg['bm']) * triton.cdiv(n, cfg['bn']), split)](
            a, act_scales, qw, qz, scales, output, m, k, n, wb, ab, cfg['bm'], cfg['bn'], split,
            num_warps=cfg['num_warps'], num_stages=3)
    except RuntimeError as error:
        if 'rejected a cold JIT' not in str(error):
            raise
        cases['private_cold_compile_rejected_before_compilation'] = True
    else:
        raise AssertionError('Direct private cold compile was accepted')
    if not kernel.snapshot()['cache_unchanged_since_freeze']:
        raise AssertionError('Rejection checks changed the frozen private cache')
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', default=['awq_w4a4', 'awq_w4a8', 'awq_w8a16',
                                                        'gptq_w4a4', 'gptq_w4a8', 'gptq_w8a16'])
    parser.add_argument('--variants', nargs='+', default=['balanced', 'wide_m', 'wide_n', 'split8'])
    parser.add_argument('--layers', nargs='+', type=int, default=[0, 18, 35])
    parser.add_argument('--timing-layers', nargs='*', type=int, default=[0])
    parser.add_argument('--projections', nargs='+', choices=list(FUSED), default=list(FUSED))
    parser.add_argument('--rows', nargs='+', type=int, default=[1, 17, 701, 2815])
    parser.add_argument('--cache-check-rows', nargs='+', type=int, default=[2, 32, 64, 65, 704, 720, 3072])
    parser.add_argument('--timing-modes', nargs='+', choices=['eager', 'graph'], default=['eager', 'graph'])
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--numeric-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'logs') or out.exists():
        raise ValueError('Choose a new directory within project/logs')
    if (not args.groups or not args.layers or not args.rows or min(args.rows) < 1 or args.repeats < 3
            or not set(args.timing_layers).issubset(args.layers)):
        raise ValueError('Invalid rows/repeats/layers')
    config_path = ROOT / 'setup/navigation-precision-20261110.json'
    groups = json.loads(config_path.read_text(encoding='utf-8'))['groups']
    if any(group not in groups for group in args.groups):
        raise ValueError('Unknown group')
    import torch
    from safetensors import safe_open
    import navigation_precision_repair_kernels as kernel
    from navigation_precision_kernels import precision_linear as old_linear
    if not args.variants or any(v not in kernel.VARIANTS for v in args.variants):
        raise ValueError('Unknown variant')
    if (max(args.rows) > kernel.MAX_ROWS or not args.cache_check_rows or min(args.cache_check_rows) < 1
            or max(args.cache_check_rows) > kernel.MAX_ROWS
            or any(m in kernel.PREPARE_ROWS for m in args.cache_check_rows)
            or not any(m % 16 == 0 for m in args.cache_check_rows)):
        raise ValueError('Cache checks must include unseen16-divisible M within1..3072')
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    out.mkdir(parents=True)
    protocol = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        sources_sha256={name: digest(ROOT / 'checks' / name) for name in
                       ('navigation_precision_repair_kernels.py', 'probe_navigation_precision_repair.py',
                        'navigation_precision_kernels.py')},
        config_sha256=digest(config_path), gpu=torch.cuda.get_device_name(),
        activation_recipes=kernel.ACTIVATION_RECIPES, relative_l2_tolerance=.005,
        kernel_preparation=dict(rows=list(kernel.PREPARE_ROWS), explicit_before_measurement=True,
                                freeze_rejects_cold_compilation=True, unseen_rows=args.cache_check_rows),
        speed_gate=dict(max_phase_median_repair_over_old=1.05, max_individual_ratio=1.20,
                        decode_timing='graph', prefill_timing='eager'),
        scope='Synthetic deterministic activations, frozen real weights; not real activation fidelity or navigation quality.',
        timing_caveat='Hot microbench CUDA-event intervals; graph capture and JIT excluded. Not HTTP/fullmodel timing.',
        a4_caveat='Group128 activation scale recipe differs from old whole-row A4; no identical-algorithm speed claim.',
        model_files={g: dict(path=groups[g]['model'],
                            config_sha256=digest(ROOT / groups[g]['model'] / 'config.json'),
                            safetensors_size=(ROOT / groups[g]['model'] / 'model.safetensors').stat().st_size)
                     for g in args.groups})
    write(out / 'protocol.json', protocol)
    records, synthetic, cache_checks = [], [], []
    cache_checked_geometries = set()
    timings = {'eager': timed, 'graph': graph_timed}
    try:
        with torch.inference_mode():
            preparation = prepare_all(args, groups, kernel, safe_open, torch)
            first_spec = groups[args.groups[0]]
            preparation['rejection_checks'] = rejection_checks(
                kernel, torch, first_spec['weight_bits'], first_spec['activation_bits'], args.variants[0])
            write(out / 'preparation.json', preparation)
            # Tail-group and tail-M checks; real checkpoints can have groups not
            # divisible by split-K. Zero rows and isolated outliers are explicit.
            for wb, ab in sorted({(groups[g]['weight_bits'], groups[g]['activation_bits']) for g in args.groups}):
                k, n = 640, 192
                qw = pack(torch.randint(0, 2 ** wb, (k, n), dtype=torch.int32, device='cuda'), wb, torch)
                qz = pack(torch.randint(0, 2 ** wb, (k // 128, n), dtype=torch.int32, device='cuda'), wb, torch)
                scales = torch.rand((k // 128, n), dtype=torch.float16, device='cuda') * .01 + .001
                for m in (1, 17, 65):
                    x = torch.randn((m, k), device='cuda', dtype=torch.float16) * .1
                    x[0, 0] = 4
                    if m > 1:
                        x[1] = 0
                    expected, activation = reference_output(x, qw, scales, qz, wb, ab, kernel, torch)
                    for variant in args.variants:
                        actual = kernel.precision_linear(x, qw, scales, qz, wb, ab, variant)
                        error = float((actual.float() - expected).norm() / expected.norm().clamp_min(1.e-8))
                        if not bool(torch.isfinite(actual).all()) or error >= .005:
                            raise AssertionError(('synthetic', wb, ab, m, variant, error))
                        synthetic.append(dict(weight_bits=wb, activation_bits=ab, rows=m, variant=variant,
                                              relative_l2=error, activation=activation,
                                              ptx=ptx_evidence(x, qw, scales, qz, wb, ab, variant, kernel, torch)))
                    del x, expected, actual
                del qw, qz, scales
            write(out / 'synthetic.json', synthetic)
            for group in args.groups:
                spec = groups[group]
                wb, ab = spec['weight_bits'], spec['activation_bits']
                with safe_open(str(ROOT / spec['model'] / 'model.safetensors'), framework='pt', device='cpu') as reader:
                    for layer in args.layers:
                        for projection in args.projections:
                            parts = FUSED[projection]
                            qw, qz, scales = [torch.cat([
                                reader.get_tensor(f'model.language_model.layers.{layer}.{part}.{suffix}')
                                for part in parts], dim=1).contiguous().cuda()
                                for suffix in ('qweight', 'qzeros', 'scales')]
                            tensors_sha256 = {name: hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                              for name, value in [('qweight', qw), ('qzeros', qz), ('scales', scales)]}
                            geometry = (wb, ab, qw.shape[0], scales.shape[1])
                            if geometry not in cache_checked_geometries:
                                for unseen_m in args.cache_check_rows:
                                    check_x = torch.randn((unseen_m, qw.shape[0]), device='cuda', dtype=torch.float16) * .1
                                    for variant in args.variants:
                                        check_y = kernel.precision_linear(check_x, qw, scales, qz, wb, ab, variant)
                                        if not bool(torch.isfinite(check_y).all()):
                                            raise AssertionError(('Unseen M nonfinite output', geometry, unseen_m, variant))
                                        if not kernel.snapshot()['cache_unchanged_since_freeze']:
                                            raise AssertionError(('Unseen M grew cache', geometry, unseen_m, variant))
                                        cache_checks.append(dict(group=group, layer=layer, projection=projection,
                                                                 rows=unseen_m, variant=variant,
                                                                 finite_output=True, cache_growth=0,
                                                                 scope='Dynamic shape/cache execution check, not numeric reference comparison'))
                                        del check_y
                                    del check_x
                                cache_checked_geometries.add(geometry)
                                write(out / 'cache_checks.json', cache_checks)
                            for m in args.rows:
                                x = torch.randn((m, qw.shape[0]), device='cuda', dtype=torch.float16) * .1
                                expected, activation = reference_output(x, qw, scales, qz, wb, ab, kernel, torch)
                                item = dict(group=group, layer=layer, projection=projection, rows=m,
                                            k=qw.shape[0], n=scales.shape[1], tensors_sha256=tensors_sha256,
                                            activation=activation, variants={})
                                measure = not args.numeric_only and layer in args.timing_layers
                                if measure:
                                    old_fn = lambda: old_linear(x, qw, scales, qz, wb, ab)
                                    item['timings'] = {'old': {mode: timings[mode](old_fn, torch, args.repeats)
                                                               for mode in args.timing_modes}}
                                a, act_scales = (x, x) if ab == 16 else kernel.quantize_activations(x, ab)
                                for variant in args.variants:
                                    fn = lambda v=variant: kernel.precision_linear(x, qw, scales, qz, wb, ab, v)
                                    actual = fn()
                                    error = float((actual.float() - expected).norm() / expected.norm().clamp_min(1.e-8))
                                    if not bool(torch.isfinite(actual).all()) or error >= .005:
                                        raise AssertionError((group, layer, projection, m, variant, error))
                                    item['variants'][variant] = dict(relative_l2=error,
                                        ptx=ptx_evidence(x, qw, scales, qz, wb, ab, variant, kernel, torch))
                                    if measure:
                                        gemm = lambda v=variant: kernel.linear_from_quantized(
                                            a, act_scales, qw, scales, qz, m, qw.shape[0], wb, ab, v)
                                        torch.cuda.reset_peak_memory_stats()
                                        measurements = {'full': {mode: timings[mode](fn, torch, args.repeats)
                                                                 for mode in args.timing_modes},
                                                        'gemm_and_split_reduction': {mode: timings[mode](gemm, torch, args.repeats)
                                                                                   for mode in args.timing_modes}}
                                        if ab < 16:
                                            quant = lambda: kernel.quantize_activations(x, ab)
                                            measurements['activation_quantization'] = {
                                                mode: timings[mode](quant, torch, args.repeats) for mode in args.timing_modes}
                                        measurements['process_allocated_peak_bytes'] = torch.cuda.max_memory_allocated()
                                        measurements['memory_scope'] = 'Probe process incl. references; not model VRAM'
                                        item['timings'][variant] = measurements
                                    del actual
                                records.append(item)
                                write(out / 'measurements.json', records)
                                print(json.dumps(dict(group=group, layer=layer, projection=projection, rows=m,
                                                      numerical='passed', measured=measure)), flush=True)
                                del x, expected, a, act_scales, fn
                            del qw, qz, scales
                            gc.collect()
                            torch.cuda.empty_cache()
        summary = dict(status='complete', numerical_gate='passed', protocol=protocol,
                       preparation=preparation, unseen_shape_cache_checks=cache_checks,
                       final_kernel_snapshot=kernel.snapshot(),
                       synthetic=synthetic, measurements=records,
                       local_speed_screen=summarize_speed(records, args.variants),
                       acceptance='Requires separate fullmodel fixed-input speed and natural-EOS quality gates before20cases.')
        if not summary['final_kernel_snapshot']['cache_unchanged_since_freeze']:
            raise AssertionError('Private kernel cache grew after freeze')
        write(out / 'result.json', summary)
        print('PRECISION_REPAIR_PROBE_COMPLETE', len(records), flush=True)
    except BaseException as error:
        write(out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                        completed_rows=len(records), completed_synthetic=len(synthetic)))
        raise


if __name__ == '__main__':
    main()
