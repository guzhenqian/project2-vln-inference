"""One-shot handoff from audited initial50 to the authorized second50."""
from pathlib import Path
import subprocess
import sys
from navigation_stage2_common import ROOT, RUN, read, write, digest

decision = read(RUN / 'rescreen_decision.json')
assert decision['status'] == 'selected_for_cumulative100'
assert read(RUN / 'status.json')['status'] == 'initial50_complete'
assert digest(ROOT / 'reports/navigation_stage2_same50_20261109.json') == decision['initial50_report_sha256']
assert digest(ROOT / 'reports/navigation_stage2_delivery_audit_50_20261109.json') == decision['independent_audit_sha256']
assert not (RUN / 'screen_stage2').exists(), 'Never silently restart an incomplete stage'
write(RUN / 'rescreen_sequence_claim.json', dict(
    decision_sha256=digest(RUN / 'rescreen_decision.json'),
    controller_sha256=digest(ROOT / 'checks/run_navigation_stage2_screen.py'),
    groups=decision['new_groups']))
with (RUN / 'rescreen_sequence.log').open('x', encoding='utf-8') as stream:
    process = subprocess.Popen([str(Path.home() / '.venvs/project2/bin/python'), '-B', '-u',
        'checks/run_navigation_stage2_screen.py', '--stage', '2', '--groups', *decision['new_groups']],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        stream.write(line)
        stream.flush()
        print(line, end='', flush=True)
    code = process.wait()
write(RUN / 'rescreen_sequence_exit.json', dict(returncode=code))
sys.exit(code)
