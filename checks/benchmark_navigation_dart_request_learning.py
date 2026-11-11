"""User-run repeated-request timing for one frozen two-image DART sample.

This is a small eager-mode diagnostic, not a navigation or general speed benchmark.
It reuses the identity-checked single-request script and its POST-only timer.
--check-only validates local input; it never contacts or starts a model service.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from check_navigation_dart_request_learning import ROOT, model_name, prepare_input


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-ratio', type=float, choices=(1.0, 0.75), required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    sample, settings, images, image_hashes, _ = prepare_input(args.keep_ratio, 0)
    warmup, repeats = 3, 10
    expected = model_name(args.keep_ratio)
    request_script = Path(__file__).with_name('check_navigation_dart_request_learning.py')
    launch_files = sorted((ROOT / 'logs').glob('navigation_dart_service_learning_*/launch.json'),
                          reverse=True)
    launch_path = next((p for p in launch_files if read_json(p).get('model_name') == expected), None)
    if launch_path is None:
        raise SystemExit('No matching service launch record; start that mode first.')
    launch = read_json(launch_path)
    runtime_dir = launch_path.parent
    if (launch['keep_ratio'] != args.keep_ratio or launch['weight_quantization'] is not None
            or launch['engine_config']['dtype'] != 'float16'
            or launch['engine_config']['quant_policy'] != 0
            or not launch['engine_config']['eager_mode']
            or launch['engine_config']['enable_prefix_caching']):
        raise SystemExit('Unexpected precision, keep ratio, or runtime configuration.')
    print(f'Mode: {expected}; sample: {sample["id"]}; images: {len(images)}', flush=True)
    print(f'Protocol: {warmup} warmup + {repeats} measured fresh requests; no retries', flush=True)
    if args.check_only:
        print('DART_REPEAT_INPUT_READY: local checks only; no HTTP, model, GPU or output writes')
        return

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'logs' / ('navigation_dart_repeat_learning_' + run_id + '.json')
    report = {
        'scope': 'Same-input eager-mode repeated POST timing; not navigation or general speed evidence',
        'status': 'running', 'keep_ratio': args.keep_ratio, 'model': expected,
        'sample_id': sample['id'], 'settings': settings, 'image_sha256': image_hashes,
        'warmup_count': warmup, 'measured_count': repeats,
        'launch_path': relative(launch_path), 'launch_sha256': sha256(launch_path),
        'script_sha256': sha256(Path(__file__)),
        'request_script_sha256': sha256(request_script),
        'timer': 'request_s from the single-request checker: chat POST until response body read',
        'runtime_evidence': 'Exactly one new prefill and matching first-decode log per serial request',
        'limitations': ['One fixed two-image input, batch1, eager mode, prefix cache disabled',
                        'Includes diagnostic logging and the current CPU-based DART selector',
                        'Excludes model startup; warmup requests excluded from timing summary',
                        'No GPU-memory saving or navigation-quality conclusion'],
        'requests': [], 'summary': None, 'error': None,
    }
    start = time.perf_counter()
    # Reserve a unique file. Persist evidence after every request, including failure.
    with output.open('x', encoding='utf-8') as stream:
        def save():
            stream.seek(0)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.truncate()
            stream.flush()

        save()
        try:
            for index in range(warmup + repeats):
                phase = 'warmup' if index < warmup else 'measured'
                number = index + 1 if phase == 'warmup' else index + 1 - warmup
                before_prefill = set(runtime_dir.glob('prefill_*.json'))
                before_decode = set(runtime_dir.glob('first_decode_*.json'))
                done = subprocess.run(
                    [sys.executable, '-u', str(request_script), '--keep-ratio', str(args.keep_ratio),
                     '--sample-index', '0'], cwd=ROOT, capture_output=True, text=True,
                    encoding='utf-8', errors='replace')
                saved = [line[len('Saved: '):] for line in done.stdout.splitlines()
                         if line.startswith('Saved: ')]
                if done.returncode != 0 or len(saved) != 1:
                    raise RuntimeError('Single request failed; no retry.\n' + done.stdout[-6000:]
                                       + done.stderr[-6000:])
                request_path = Path(saved[0]).resolve()
                if not request_path.is_relative_to(ROOT / 'logs'):
                    raise ValueError('Unexpected request log location')
                record = read_json(request_path)
                row = {'phase': phase, 'number': number, 'request_log': relative(request_path),
                       'request_sha256': sha256(request_path), 'request_s': record['request_s'],
                       'reply': record['reply'], 'usage': record['usage']}
                report['requests'].append(row)
                save()
                if (record['status'] != 'functional_request_verified'
                        or record['served_model_id'] != expected
                        or record['response_model_id'] != expected or record['settings'] != settings
                        or record['image_sha256'] != image_hashes
                        or record['source_messages'] != sample['messages']):
                    raise ValueError('Request identity, input or response validation mismatch')
                if record['script_sha256'] != report['request_script_sha256']:
                    raise ValueError('Single-request script changed during this run')
                if sha256(launch_path) != report['launch_sha256']:
                    raise ValueError('Service launch record changed during this run')
                new_prefill = set(runtime_dir.glob('prefill_*.json')) - before_prefill
                new_decode = set(runtime_dir.glob('first_decode_*.json')) - before_decode
                if len(new_prefill) != 1 or len(new_decode) != 1:
                    raise ValueError('Expected one new prefill/decode pair; do not run other requests concurrently')
                prefill_path, decode_path = next(iter(new_prefill)), next(iter(new_decode))
                p, d = read_json(prefill_path), read_json(decode_path)
                kept = p['kept_original_positions']
                expected_image = 600 if args.keep_ratio == 1.0 else 450
                expected_total = 101 + expected_image
                expected_mode = ('native_control' if args.keep_ratio == 1.0
                                 else 'dart_layer_internal_multi_image_exact_budget_deterministic_v1')
                image_positions = {i for lo, hi in p['image_spans'] for i in range(lo, hi)}
                non_image = set(range(701)) - image_positions
                per_image = [sum(lo <= i < hi for i in kept) for lo, hi in p['image_spans']]
                if (p['trace_id'] != d['trace_id'] or p['keep_ratio'] != args.keep_ratio
                        or p['actual_mode'] != expected_mode
                        or p['original_length'] != 701 or p['original_image_tokens'] != 600
                        or p['retained_length'] != expected_total
                        or p['retained_image_tokens'] != expected_image
                        or p['removed'] != 701 - expected_total
                        or len(kept) != expected_total or kept != sorted(set(kept))
                        or any(i < 0 or i >= 701 for i in kept)
                        or not non_image.issubset(kept) or len(non_image) != 101
                        or per_image != [expected_image // 2] * 2 or kept[-1] != 700
                        or p['lower_layer_kv_length'] != 701
                        or p['upper_layer_kv_length'] != expected_total
                        or p['weight_dtype'] != 'float16' or p['awq'] or p['kv_quant_policy'] != 0
                        or d['lower_layer_kv_length'] != 702
                        or d['upper_layer_kv_length'] != expected_total + 1
                        or d['scheduler_positions'] != [[701]]
                        or d['mrope_positions'] != [[141], [141], [141]]
                        or not d['original_positions_preserved']):
                    raise ValueError('Actual runtime token/cache/position evidence mismatch')
                row.update(prefill_log=relative(prefill_path), prefill_sha256=sha256(prefill_path),
                           decode_log=relative(decode_path), decode_sha256=sha256(decode_path),
                           runtime_evidence_verified=True)
                save()
                print(f'{phase} {number}: {record["request_s"]:.3f}s; '
                      f'output={record["usage"]["completion_tokens"]}; runtime OK', flush=True)
            measured = [row for row in report['requests'] if row['phase'] == 'measured']
            times = [row['request_s'] for row in measured]
            report['summary'] = {
                'samples': len(times), 'median_s': statistics.median(times),
                'mean_s': statistics.mean(times), 'min_s': min(times), 'max_s': max(times),
                'distinct_replies': len({row['reply'] for row in measured}),
                'output_token_counts': sorted({row['usage']['completion_tokens'] for row in measured}),
            }
            report['status'] = 'complete'
        except KeyboardInterrupt:
            report.update(status='interrupted', error='Interrupted by the user; no retry')
        except Exception as error:
            report.update(status='failed', error=str(error))
        finally:
            report['elapsed_s'] = time.perf_counter() - start
            report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    print('Saved:', output, flush=True)
    if report['status'] != 'complete':
        raise SystemExit('DART_REPEAT_FAILED: ' + report['error'])
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)
    print('DART_REPEAT_TIMING_OK', flush=True)
    print('Compare only with the same 3-warmup/10-measured protocol in the other mode.')


if __name__ == '__main__':
    main()
