"""One FP16 attribution pass, using the already verified two/nine-image inputs."""
import json
import os
import subprocess
import time
from run_navigation_quant_timing import PYTHON,OPENER,request
from navigation_stage2_common import ROOT,read,write,digest
from check_navigation_awq_dart_request import prepare_input
from run_navigation_dart50_night import stop_owned


def main():
    run=ROOT/'logs/navigation_quant_timing_20261110T120639Z'
    assert read(run/'status.json')['status']=='complete'
    directory=run/'attribution_fp16'
    service=directory/'service'
    service.mkdir(parents=True,exist_ok=False)
    plan=[dict(phase='warmup',sample=s,repeat=r) for s in (0,1) for r in range(3)]
    plan += [dict(phase='measured',sample=s,repeat=r) for r in range(4) for s in (0,1)]
    plan += [dict(phase='flush',sample=0,repeat=0)]
    write(service/'request_plan.json',plan)
    write(directory/'sources.json',{name:digest(ROOT/'checks'/name) for name in ['navigation_quant_attribution.py','navigation_quant_attribution_module_map.py','serve_navigation_quant_attribution.py','run_navigation_quant_attribution.py']})
    payloads={s:json.loads(prepare_input(1.,s,'navigation')[-1]) for s in (0,1)}
    child=None
    try:
        with (directory/'console.log').open('x') as out:
            child=subprocess.Popen([PYTHON,'-B','-u','checks/serve_navigation_quant_attribution.py','--group','fp16',
                '--mode','audited','--run-dir',str(service),'--port','23460'],cwd=ROOT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        write(directory/'ownership.json',dict(pid=child.pid,port=23460))
        deadline=time.monotonic()+240
        while True:
            assert child.poll() is None
            try:
                with OPENER.open('http://127.0.0.1:23460/v1/models',timeout=2) as response:
                    assert json.load(response)['data'][0]['id']=='quant-timing-fp16'
                break
            except OSError:
                if time.monotonic()>deadline: raise TimeoutError('attribution startup')
                time.sleep(2)
        rows=[]
        for spec in plan:
            start=time.perf_counter()
            result=request(23460,'fp16',spec,payloads)
            result.update(client_outer_start=start,client_outer_end=time.perf_counter())
            rows.append(result)
            write(directory/'requests.json',rows,exclusive=False)
        print('ATTRIBUTION_REQUESTS_COMPLETE',flush=True)
    finally:
        stop_owned(child)


if __name__=='__main__':
    main()
