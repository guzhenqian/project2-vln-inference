"""Continue this one authorized validation job into initial50; never schedule retries."""
import os
from pathlib import Path
import subprocess
import time
from navigation_stage2_common import ROOT, RUN, GROUPS, read, write, digest
from run_navigation_dart50_night import processes, now


def main():
    claim=RUN/'initial_sequence_claim.json'
    if claim.exists():
        raise RuntimeError('Initial sequence already claimed; inspect it instead of duplicating')
    expected=digest(ROOT/'setup/navigation-stage2-execution-20261109.json')
    assert read(ROOT/'setup/navigation-stage2-execution-20261109.json')['execution_enabled']
    deadline=time.monotonic()+4*3600
    print('STAGE2_CONTINUATION_WAITING_FOR_ALL_ACCEPTANCE',flush=True)
    while True:
        state=read(RUN/'status.json')
        if state['status'] in ('validation_failed','failed','cancelled'):
            raise RuntimeError('Validation stopped; continuation will not retry or start formal runs')
        if state['status']=='validated_ready':
            active=[pid for pid,item in processes().items() if item['cwd']==str(ROOT)
                and 'checks/run_navigation_stage2_validation.py' in item['words']]
            if not active:
                break
        if time.monotonic()>deadline:
            raise TimeoutError('Validation did not finish within the bounded continuation window')
        time.sleep(5)
    assert digest(ROOT/'setup/navigation-stage2-execution-20261109.json')==expected
    assert all(read(RUN/'validation'/(group+'_passed.json'))['passed'] for group in GROUPS)
    write(claim,dict(pid=os.getpid(),created_utc=now(),automatic_retry=False,
        execution_contract_sha256=expected,screening_controller_sha256=digest(ROOT/'checks/run_navigation_stage2_screen.py')))
    command=[str(Path.home()/'.venvs/project2/bin/python'),'-B','-u','checks/run_navigation_stage2_screen.py','--stage','1']
    print('STAGE2_CONTINUATION_STARTING_INITIAL50',flush=True)
    result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
    write(RUN/'initial_sequence_exit.json',dict(returncode=result.returncode,finished_utc=now()))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
