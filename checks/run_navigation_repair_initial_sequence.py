"""Continue the authorized initial screens after the separately owned FP16 run."""
import argparse
from datetime import datetime, timezone
import subprocess
import time
import traceback

from navigation_stage2_common import ROOT, read, write, digest
from run_navigation_w4a16_fixes import PYTHON, stop_child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-run', required=True)
    args = parser.parse_args()
    output, baseline_run = ROOT/args.output, ROOT/args.baseline_run
    assert not output.exists() and output.is_relative_to(ROOT/'logs')
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        controller_sha256=digest(__file__), baseline_run=args.baseline_run,
        scope='Wait for current FP16; W4 numerical gate; independent two quant screens; repeat fixed FP16'))
    child = None

    def status(state, **extra):
        write(output/'status.json', dict(status=state, updated_utc=datetime.now(timezone.utc).isoformat(), **extra), exclusive=False)

    def execute(label, argv):
        nonlocal child
        compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
            '--format=csv,noheader'], text=True).strip()
        assert not compute, 'Another GPU compute process remains: '+compute
        with (output/(label+'.log')).open('x') as stream:
            child = subprocess.Popen([PYTHON, '-B', '-u']+argv, cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status('running', active_step=label, child_pid=child.pid)
        code = child.wait()
        child = None
        assert code == 0, label+' failed; inspect retained log'

    try:
        status('waiting_for_owned_fp16')
        while True:
            state = read(baseline_run/'status.json')['status']
            if state == 'complete':
                break
            assert state != 'failed', 'Baseline controller failed'
            time.sleep(10)
        baseline = baseline_run/'fp16/result.json'
        assert read(baseline)['passed_stability'], 'Baseline stability gate failed'
        probe = output/'w4_kernel_gate'
        execute('w4_kernel_gate', ['checks/probe_navigation_w4a16_repair.py', '--output', str(probe)])
        assert read(probe/'result.json')['numerical_gate_passed']
        execute('w4_screens', ['checks/run_navigation_repair_checked.py', '--output', str(output/'w4_screens'),
            '--manifest', 'data/baseline/navigation_repair_20261112/subset20.json',
            '--groups', 'awq_w4a16', 'gptq_w4a16', '--base-port', '23720', '--baseline', str(baseline)])
        execute('fp16_fixed_repeat', ['checks/run_navigation_repair_checked.py', '--output', str(output/'fp16_fixed_repeat'),
            '--manifest', 'data/baseline/navigation_repair_20261112/subset20.json',
            '--groups', 'fp16', '--base-port', '23730', '--fixed-only'])
        status('initial_screens_finished_pending_independent_audit')
    except BaseException as error:
        write(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        status('failed')
        raise
    finally:
        stop_child(child)


if __name__ == '__main__':
    main()
