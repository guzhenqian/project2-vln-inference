"""Numerical/hot-timing gate for a frozen W4A16 navigation launch table."""
import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from navigation_stage2_common import ROOT, GROUPS, digest, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', choices=('awq', 'gptq'), default=['awq', 'gptq'])
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--replay-table', type=Path,
                        help='Child validation mode: warm this exact table without autotuning; exactly one matching group.')
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'logs') and not out.exists()
    out.mkdir(parents=True)
    import torch
    from safetensors import safe_open
    import navigation_w4a16_repair_kernels as repair
    from navigation_w4a16_candidate_kernels import candidate_linear
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    projections = {'qkv': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        'o': ['self_attn.o_proj'], 'gate_up': ['mlp.gate_proj', 'mlp.up_proj'], 'down': ['mlp.down_proj']}
    sizes = (1, 701, 2815, 713, 1737, 3001, 16, 17, 33, 65, 704, 720, 3072)
    from run_navigation_w4a16_fixes import FROZEN
    models = {}
    for group in args.groups:
        path = ROOT/GROUPS[group]['model']
        weight_sha = digest(path/'model.safetensors')
        assert weight_sha == FROZEN[group], 'Frozen W4 checkpoint identity changed'
        models[group] = dict(path=GROUPS[group]['model'], config_sha256=digest(path/'config.json'),
                             weights_sha256=weight_sha)
    if args.replay_table:
        assert len(args.groups) == 1, 'Frozen replay belongs to exactly one checkpoint identity'
        repair.load_launch_table(args.replay_table, args.groups[0], models[args.groups[0]]['weights_sha256'])
    meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), groups=args.groups,
        layers=[0], fused_projections=projections, row_counts=list(sizes), repeats=args.repeats,
        relative_l2_limit=0.005, model_files=models,
        explicit_replay_table=str(args.replay_table.resolve()) if args.replay_table else None,
        sources_sha256={name: digest(ROOT/'checks'/name) for name in
            ('navigation_w4a16_repair_kernels.py', 'probe_navigation_w4a16_repair.py',
             'navigation_w4a16_candidate_kernels.py')},
        scope='Same real frozen quantized matrices; random FP16 inputs. Kernel regression and hot timing, not FP16 model quality or navigation acceptance.',
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,temperature.gpu',
            '--format=csv,noheader'], text=True).strip())
    write(out/'protocol.json', meta)

    def load(group, parts):
        with safe_open(str(ROOT/GROUPS[group]['model']/'model.safetensors'), framework='pt', device='cpu') as reader:
            return tuple(torch.cat([reader.get_tensor('model.language_model.layers.0.'+p+'.'+suffix)
                for p in parts], dim=1).cuda() for suffix in ('qweight', 'scales', 'qzeros'))

    def unpack(packed):
        return torch.stack([(packed >> shift) & 15 for shift in [0,16,4,20,8,24,12,28]], -1).reshape(packed.shape[0], -1)

    def timed(fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        events = []
        start = time.perf_counter()
        for _ in range(args.repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            events.append((a, b))
        torch.cuda.synchronize()
        return dict(cuda_interval_ms=statistics.median(a.elapsed_time(b) for a,b in events),
            host_wall_mean_ms=(time.perf_counter()-start)*1000/args.repeats)

    def graph_timed(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fn()
        result = timed(graph.replay)
        del graph, output
        return result

    rows = []
    try:
        with torch.inference_mode():
            # Prepare on one algorithm's real weights; launch choice depends on
            # shape, not algorithm identity. Both algorithms use this one table.
            for parts in projections.values():
                qw, s, qz = load(args.groups[0], parts)
                repair.prepare_projection(qw, s, qz)
                del qw, s, qz
                gc.collect()
                torch.cuda.empty_cache()
            receipt = repair.freeze(args.groups[0], models[args.groups[0]]['weights_sha256'])
            write(out/'launch_table.json', receipt)
            repair.validate_table(receipt, args.groups[0], models[args.groups[0]]['weights_sha256'])
            rejected_wrong_identity = False
            try:
                repair.validate_table(receipt, 'gptq' if args.groups[0] == 'awq' else 'awq', models[args.groups[0]]['weights_sha256'])
            except ValueError:
                rejected_wrong_identity = True
            assert rejected_wrong_identity
            if args.replay_table:
                assert receipt['selection_mode'] == 'replayed_frozen_table'
                assert all(count == 0 for count in receipt['autotuner_cache_entries'].values())
            initial_cache = receipt['autotuner_cache_entries']
            initial_compiled = receipt['compiled_entries']
            for group in args.groups:
                for label, parts in projections.items():
                    qw, s, qz = load(group, parts)
                    dense = ((unpack(qw)-unpack(qz).repeat_interleave(128, 0)).half()
                             * s.repeat_interleave(128, 0)).T.contiguous()
                    for m in sizes:
                        x = torch.randn((m, qw.shape[0]), device='cuda', dtype=torch.float16)*0.1
                        ref = torch.nn.functional.linear(x, dense)
                        funcs = {'phase_tuned': lambda: candidate_linear(x, qw, s, qz, 'phase_tuned'),
                                 'frozen_launch': lambda: repair.precision_linear(x, qw, s, qz)}
                        row = dict(group=group, layer=0, projection=label, m=m, k=qw.shape[0],
                            n=s.shape[1], representative=repair.representative(m), variants={})
                        outputs = {}
                        for name, fn in funcs.items():
                            actual = fn()
                            torch.cuda.synchronize()
                            error = float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-8))
                            assert bool(torch.isfinite(actual).all()) and error < 0.005, (name, label, m, error)
                            row['variants'][name] = dict(relative_l2=error, eager=timed(fn))
                            if m == 1:
                                row['variants'][name]['cuda_graph_replay'] = graph_timed(fn)
                            outputs[name] = actual
                        row['frozen_vs_phase_relative_l2'] = float((outputs['frozen_launch'].float()-outputs['phase_tuned'].float()).norm()/outputs['phase_tuned'].float().norm().clamp_min(1e-8))
                        assert row['frozen_vs_phase_relative_l2'] < 0.005
                        assert repair.snapshot()['autotuner_cache_entries'] == initial_cache
                        assert repair.snapshot()['compiled_entries'] == initial_compiled
                        rows.append(row)
                        write(out/'measurements.json', rows, exclusive=False)
                        print(json.dumps(dict(group=group, projection=label, m=m,
                            relative_l2=row['variants']['frozen_launch']['relative_l2'],
                            timing_ms={k:round(v['eager']['cuda_interval_ms'], 4) for k,v in row['variants'].items()})), flush=True)
                        del x, ref, outputs, actual, funcs
                    del qw, s, qz, dense
                    gc.collect()
                    torch.cuda.empty_cache()
        replay_verification = None
        if not args.replay_table:
            # A fresh worker proves a saved table can be replayed across service
            # restarts. This is separate from the unchanged primary104 checks.
            child_output = out/'frozen_table_replay'
            with (out/'frozen_table_replay.log').open('x', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                    '--output', str(child_output), '--groups', args.groups[0],
                    '--repeats', str(args.repeats), '--replay-table', str(out/'launch_table.json')],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            replay_result = json.loads((child_output/'result.json').read_text())
            assert replay_result['status'] == 'passed'
            replay_table = replay_result['launch_table']
            assert replay_table['shapes'] == receipt['shapes']
            assert all(count == 0 for count in replay_table['autotuner_cache_entries'].values())
            replay_verification = dict(status='passed', path=child_output.relative_to(ROOT).as_posix(),
                result_sha256=digest(child_output/'result.json'), cases=len(replay_result['measurements']),
                exact_selected_configs_preserved=True, startup_autotune_calls=0)
        write(out/'result.json', dict(status='passed', numerical_gate_passed=True,
            performance_gate='Descriptive timings only; full fixed-input service gate required before 20 cases.',
            runtime_autotuner_cache_unchanged=True, runtime_compiled_cache_unchanged=True,
            wrong_algorithm_identity_rejected=True, frozen_table_restart_verification=replay_verification,
            protocol=meta, launch_table=receipt, measurements=rows))
        print('W4A16_REPAIR_KERNEL_GATE_PASSED', len(rows), flush=True)
    except BaseException as error:
        write(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), completed_rows=len(rows)))
        raise


if __name__ == '__main__':
    main()
