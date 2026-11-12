"""Same-process FP16 idle ablation after differing cross-run prefill timings."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from navigation_stage2_common import ROOT, read, write, digest
from run_navigation_w4a16_fixes import PYTHON, OPENER, request, stop_child
from check_navigation_awq_dart_request import prepare_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=23910)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'logs') and not out.exists()
    service = out/'service'
    service.mkdir(parents=True)
    payloads = {s: json.loads(prepare_input(1., s, 'navigation')[-1]) for s in (0, 1)}
    plan = [dict(phase='warmup', sample=s, tokens=t, repeat=r, idle_s=1.)
            for s in (0, 1) for t in (18, 64) for r in range(2)]
    # ABBA twice with alternating blocks; equal counts and one model process.
    plan += [dict(phase='measured', sample=1, tokens=18, repeat=i, idle_s=float(idle))
             for i, idle in enumerate([0,1,1,0,1,0,0,1,0,1,1,0])]
    plan.append(dict(phase='flush', sample=0, tokens=18, repeat=0, idle_s=1.))
    write(service/'request_plan.json', plan)
    confirmation = ROOT/'logs/navigation_w4a16_fix_20261112/confirmation'
    prior = read(confirmation/'protocol.json')
    model = ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy/model.safetensors'
    assert model.stat().st_mtime <= (confirmation/'protocol.json').stat().st_mtime
    write(out/'protocol.json', dict(scope='One unchanged FP16 eager model process; alternating 0/1 second untimed idle. Diagnose run variability, not a new optimization.',
        prior_full_weight_sha256=prior['actual_weight_sha256']['fp16'],
        weight_identity_scope='Full SHA already verified by preceding confirmation controller; model modification time predates that receipt. No new weight write or full rehash.',
        preceding_protocol_sha256=digest(confirmation/'protocol.json'),
        model_file_size=model.stat().st_size, source_sha256=digest(Path(__file__)), plan=plan))
    child = None
    try:
        with (out/'console.log').open('x') as stream:
            child = subprocess.Popen([PYTHON, '-B', '-u', 'checks/serve_navigation_w4a16_fix.py',
                '--group','fp16','--variant','stock','--graph','off','--run-dir',str(service),
                '--port',str(args.port)], cwd=ROOT, env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write(out/'ownership.json',dict(pid=child.pid,port=args.port))
        deadline = time.monotonic()+300
        while True:
            if child.poll() is not None:
                raise RuntimeError('Probe service exited')
            try:
                with OPENER.open(f'http://127.0.0.1:{args.port}/v1/models',timeout=2) as response:
                    assert json.load(response)['data'][0]['id']=='w4a16-fix-fp16'
                break
            except OSError:
                if time.monotonic()>deadline:
                    raise TimeoutError('Probe startup')
                time.sleep(2)
        assert read(service/'execution_audit.json')['first_native_forward_verified']
        rows=[]
        for spec in plan:
            time.sleep(spec['idle_s'])
            row=request(args.port,'fp16',spec,payloads)
            rows.append(row)
            write(out/'requests.json',rows,exclusive=False)
        stop_child(child)
        child=None
        measured=[]
        for i,row in enumerate(rows[:-1]):
            timing=read(service/f'timing_{i:03d}.json')
            assert timing['plan']==row['spec']
            if row['spec']['phase']!='measured':
                continue
            pre=timing['steps'][0]
            measured.append(dict(idle_s=row['spec']['idle_s'], repeat=row['spec']['repeat'],
                language_ms=pre['timers']['language']['cuda_interval_ms'],
                language_host_ms=pre['timers']['language']['host_wall_ms'],
                vision_ms=pre['timers']['vision']['cuda_interval_ms'],
                http_s=row['total_s'], output_tokens=row['output_tokens']))
        assert all(r['output_tokens']==measured[0]['output_tokens'] for r in measured)
        summary=[]
        for idle in (0.,1.):
            selected=[r for r in measured if r['idle_s']==idle]
            summary.append(dict(idle_s=idle,n=len(selected),
                medians={k:statistics.median(r[k] for r in selected) for k in
                         ('language_ms','language_host_ms','vision_ms','http_s')},
                language_range_ms=[min(r['language_ms'] for r in selected),max(r['language_ms'] for r in selected)]))
        result=dict(status='complete',summary=summary,measurements=measured,output_tokens_identical=True,
            caveat='Single FP16 process; cannot by itself diagnose cross-process kernel selection, cache or system-state changes.')
        write(out/'result.json',result)
        print(json.dumps(dict(status='complete',summary=summary)),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        stop_child(child)


if __name__=='__main__':
    main()
