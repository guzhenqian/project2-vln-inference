"""Owned serial continuation; no failed/incomplete20 can enter formal80."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

from navigation_stage2_common import ROOT, read, write, digest
from run_navigation_w4a16_fixes import PYTHON, stop_child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'logs/navigation_repair_20261112') and not out.exists()
    lock = (ROOT/'tmp/navigation_repair_continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir()
    prefix = ROOT/'logs/navigation_repair_20261112'
    screens = prefix/'screen_six_balanced_v1'
    initial = prefix/'initial_sequence_v1'
    baseline = prefix/'screen_fp16_v1/fp16'
    groups = ['fp16']+[a+'_'+p for a in ('awq','gptq') for p in ('w4a16','w4a4','w4a8','w8a16')]
    dirs = {g: baseline if g=='fp16' else initial/'w4_screens'/g if g.endswith('_w4a16') else screens/g for g in groups}
    report_prefix = 'navigation_repair_20261114_continuation'
    acceptance = ROOT/'reports'/(report_prefix+'_20.json')
    child = None
    completed = []

    def status(state, **extra):
        write(out/'status.json', dict(status=state, updated_utc=datetime.now(timezone.utc).isoformat(),
            completed100=completed, **extra), exclusive=False)

    def execute(label, command):
        nonlocal child
        with (out/(label+'.log')).open('x') as stream:
            child = subprocess.Popen([PYTHON,'-B','-u']+list(map(str,command)), cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status('running', active_step=label, child_pid=child.pid)
        code = child.wait()
        child = None
        assert code == 0, label+' failed; inspect retained log'

    write(out/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(__file__), wait_for=str(screens.relative_to(ROOT)),
        purpose='Finish owned six screens; fresh FP16 fixed repeat; independent20 audit; only accepted groups append80; collect and summarize.',
        scope='Failures remain visible; no invented100 quality threshold or historical run reuse.'))
    try:
        status('waiting_for_owned_six_screens')
        while True:
            state = read(screens/'status.json')['status']
            assert state != 'failed', 'Six-screen controller failed; repair before continuing'
            if state == 'complete':
                break
            time.sleep(15)
        execute('fp16_repeat', ['checks/run_navigation_repair_checked.py','--output',out/'fp16_repeat',
            '--manifest','data/baseline/navigation_repair_20261112/subset20.json',
            '--groups','fp16','--base-port','23900','--fixed-only'])
        execute('audit20', ['checks/audit_navigation_repair.py','--directories',*dirs.values(),
            '--kernel-gates',initial/'w4_kernel_gate/result.json',prefix/'precision_gate_six_balanced_v1/result.json',
            '--fp16-repeat',out/'fp16_repeat/fp16','--output',acceptance])
        audit = read(acceptance)
        rows = {r['group']:r for r in audit['table']}
        assert rows['fp16']['passed'] and not audit['fp16_repeat']['speed_review_required'], 'Baseline/drift requires review'
        rejected = [g for g in groups if not rows[g]['passed']]
        reports = []
        execute('inventory20', ['checks/summarize_navigation_repair100.py','--screen-audit',acceptance,
            '--output', ROOT/'reports'/(report_prefix+'_inventory20.json')])
        for i,g in enumerate(groups):
            if not rows[g]['passed']:
                continue
            assert digest(ROOT/'checks/audit_navigation_repair.py') == audit['audit_source_sha256'], 'Acceptance implementation changed'
            assert digest(dirs[g]/'result.json') == audit['groups'][g]['result_sha256'], 'Accepted20 result changed'
            run = out/('remaining80_'+g)
            command = ['checks/run_navigation_repair_checked.py','--output',run,
                '--manifest','data/baseline/navigation_repair_20261112/remaining80.json',
                '--groups',g,'--base-port',str(23910+i)]
            if g != 'fp16':
                command += ['--baseline',baseline/'result.json']
            if g.endswith('_w4a16'):
                command += ['--frozen-launch-table',dirs[g]/'service/repair_execution_audit.json']
            execute('remaining80_'+g, command)
            result = read(run/g/'result.json')
            assert result['case_count']==80 and result['status']=='complete', 'Formal80 was not completed'
            report = ROOT/'reports'/(report_prefix+'_'+g+'_100.json')
            execute('collect_'+g, ['checks/collect_navigation_repair100.py','--screen-dir',dirs[g],
                '--remaining-dir',run/g,'--screen-audit',acceptance,'--output',report])
            reports.append(report)
            completed.append(g)
            execute('inventory_'+g, ['checks/summarize_navigation_repair100.py','--screen-audit',acceptance,
                '--reports',*reports,'--output',ROOT/'reports'/(report_prefix+'_inventory_after_'+g+'.json')])
        status('accepted_groups_finished_pending_review', rejected20=rejected,
            note='Requires human-readable final review and GitHub publication; failed groups remain unresolved.')
    except BaseException as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        status('failed')
        raise
    finally:
        stop_child(child)


if __name__=='__main__':
    main()
