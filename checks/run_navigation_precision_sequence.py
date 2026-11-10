"""Serial calibration handoff, real runtime gates, then six frozen same100 runs."""
import argparse
import json
import subprocess
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

from navigation_precision_common import ROOT, RUN, GROUPS, SOURCES, read, write, digest
from run_navigation_precision_validation import OwnedSession, status, PYTHON, stop_owned
from audit_navigation_precision import audit_service
from collect_navigation_stage2 import collect

QPY = str(Path.home()/'.venvs/project2-quant312/bin/python')


def now():
    return datetime.now(timezone.utc).isoformat()


def command(argv, label):
    with (RUN/(label+'_'+time.strftime('%Y%m%dT%H%M%S')+'.log')).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(active_step=label, active_child_pid=process.pid)
        try:
            code = process.wait()
        except BaseException:
            stop_owned(process)
            raise
    if code:
        raise RuntimeError(f'{label} exited {code}; see retained log')


def export_progress():
    result_rows = []
    old = ROOT/'logs/navigation_stage2_sequence_20261109/results'
    for algorithm in ('awq','gptq'):
        data = read(old/(algorithm+'_100.json'))
        result_rows.append(dict(group=algorithm+'_w4a16', status='complete_reused', recipe='stock W4A16',
                               source=(old/(algorithm+'_100.json')).relative_to(ROOT).as_posix(), metrics=data))
    for group, recipe in GROUPS.items():
        path = RUN/'results'/(group+'_100.json')
        result_rows.append(dict(group=group, status='complete' if path.exists() else 'pending',
                               recipe=recipe, metrics=read(path) if path.exists() else None))
    report = dict(updated_utc=now(), rows=result_rows,
                  status='complete' if all(row['metrics'] is not None for row in result_rows) else 'in_progress',
                  scope='same100 development cases; experimental precision kernels; no Token compression',
                  caveats=['W4A4 packed operands widen to INT8 MMA; not native INT4 acceleration.',
                           'New kernels differ from the stock W4A16 control: configuration comparison, not pure bitwidth effect.',
                           'Original weights reused for W4+A4/A8; activation quantization is dynamic per-token.',
                           'A4/A8 applies to quantized language projection inputs; other operations/intermediate tensors remain FP16.'])
    write(RUN/'comparison.json', report, exclusive=False)
    lines = ['# 八组量化精度比较（进行中）', '', '仅填写完成并审计通过的组。W4A4为打包四位数值、INT8整数计算；新组采用独立实验算子，旧W4A16沿用原生算子。', '',
             '| 配置 | 状态 | 成功数/100 | SPL | P50/s | P95/s | 峰值/MiB |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in result_rows:
        m = row['metrics']
        if m:
            lines.append(f"| {row['group']} | {row['status']} | {m['success_count']} | {m['spl']:.4f} | {m['request_median_s']:.3f} | {m['request_p95_s']:.3f} | {m['sampled_whole_card_peak_mib']:.0f} |")
        else:
            lines.append(f"| {row['group']} | 待验证/运行 | — | — | — | — | — |")
    (RUN/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-reviewed', action='store_true')
    args = parser.parse_args()
    if args.resume_reviewed:
        assert RUN.exists() and read(RUN/'status.json')['status'] == 'failed'
    else:
        RUN.mkdir(exist_ok=False)
        write(RUN/'status.json', dict(status='preparing', created_utc=now(), formal_started=False))
    for name in ('validation','formal','results'):
        (RUN/name).mkdir(exist_ok=True)
    controller_snapshot = RUN/('controller_start_'+time.strftime('%Y%m%dT%H%M%S')+'.json')
    write(controller_snapshot, dict(utc=now(), precision_config_sha256=digest(ROOT/'setup/navigation-precision-20261110.json'),
        sources_sha256={name:digest(ROOT/'checks'/name) for name in (
            'run_navigation_precision_sequence.py','run_navigation_precision_validation.py',
            'audit_navigation_precision.py','audit_navigation_stage2.py','collect_navigation_stage2.py',
            'run_simplevln_diagnostic.py','run_navigation_precision_diagnostic.py',
            'navigation_precision_quality.py','simplevln_policy.py','simplevln_env.py')}))
    export_progress()
    try:
        status(status='waiting_for_calibration', active_step='calibration', formal_started=False, failure=None)
        while True:
            calibration = read(ROOT/'logs/navigation_precision_w8_sequence_20261110/status.json')
            if calibration['status'] == 'failed':
                raise RuntimeError('W8 calibration failed: '+str(calibration))
            if calibration['status'] == 'complete':
                break
            time.sleep(15)
        for algorithm in ('awq','gptq'):
            source = f'models/SimpleVLN_Qwen3VL_4B_RL_{algorithm.upper()}_W8_CT_20261110'
            target = GROUPS[algorithm+'_w8a16']['model']
            if not (ROOT/target/'conversion_audit.json').exists():
                command([QPY,'-B','-u',f'checks/convert_navigation_{algorithm}_w8_20261110.py',
                         '--source',source,'--output',target], algorithm+'_conversion')
            assert read(ROOT/target/'conversion_audit.json')['quantization_algorithm']==algorithm.upper()
            gate = f'reports/navigation_{algorithm}_w8_kernel_gate_20261110.json'
            if not (ROOT/gate).exists():
                command([PYTHON,'-B','-u','checks/check_navigation_precision_kernels.py',
                         '--w8-model',target,'--report',gate], algorithm+'_kernel_gate')
            assert read(ROOT/gate)['status']=='passed'
        for group in GROUPS:
            evidence = RUN/'validation'/(group+'_passed.json')
            if evidence.exists():
                audit_service(ROOT/read(evidence)['service'], group)
                continue
            directory = RUN/'validation'/(group+'_'+time.strftime('%Y%m%dT%H%M%S'))
            directory.mkdir()
            owner = OwnedSession(group, directory)
            try:
                status(status='validating', active_group=group, formal_started=False)
                owner.start()
                owner.smoke(0)
                owner.smoke(1)
                single = owner.episode_batch(1, single=True)
                write(evidence, dict(passed=True, group=group, service=owner.service.relative_to(ROOT).as_posix(),
                      controller_snapshot_sha256=digest(controller_snapshot),
                      single=single.relative_to(ROOT).as_posix(), single_audit_sha256=digest(single/'runtime_audit.json'),
                      sources_sha256=read(owner.service/'launch.json')['sources_sha256'], utc=now()))
            finally:
                owner.close()
        # Freeze validated source snapshots before any formal case is launched.
        for group in GROUPS:
            evidence = read(RUN/'validation'/(group+'_passed.json'))
            audit_service(ROOT/evidence['service'], group)
        for group in GROUPS:
            result_path = RUN/'results'/(group+'_100.json')
            if result_path.exists():
                assert read(result_path)['status']=='complete'
                continue
            attempts = list((RUN/'formal').glob(group+'_*'))
            if attempts:
                raise RuntimeError('Unfinished formal cases exist; explicit case-level recovery required, no automatic rerun: '+str(attempts))
            directory = RUN/'formal'/(group+'_'+time.strftime('%Y%m%dT%H%M%S'))
            directory.mkdir()
            owner = OwnedSession(group, directory)
            try:
                status(status='running100', active_group=group, formal_started=True)
                owner.start()
                batches = [owner.episode_batch(i) for i in (1,2)]
                result = collect(group,batches,100)
                audits = [read(p/'runtime_audit.json') for p in batches]
                assert all(a['passed'] and a['infrastructure_errors'] == 0 for a in audits)
                assert result['error_count'] == sum(a['quality_error_count'] for a in audits)
                result['quality_failures_retained_in_denominator'] = True
                result.update(batch_directories=[p.relative_to(ROOT).as_posix() for p in batches],
                    controller_snapshot_sha256=digest(controller_snapshot),
                    runtime_audit_sha256={p.relative_to(ROOT).as_posix():digest(p/'runtime_audit.json') for p in batches},
                    precision_config_sha256=digest(ROOT/'setup/navigation-precision-20261110.json'),
                    source_hashes={name:digest(ROOT/'checks'/name) for name in SOURCES})
                write(result_path,result)
                export_progress()
                print('PRECISION_SAME100_COMPLETE',group,result['success_count'],flush=True)
            finally:
                owner.close()
        export_progress()
        status(status='complete', active_group=None, active_step=None, completed_utc=now())
    except BaseException as error:
        failure = dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),utc=now())
        write(RUN/('failure_'+time.strftime('%Y%m%dT%H%M%S')+'.json'),failure)
        status(status='failed',failure=failure)
        raise


if __name__ == '__main__':
    main()
