"""Owned, sequential same-input FP16/AWQ/GPTQ timing diagnostic (no navigation)."""
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from urllib.request import Request, build_opener, ProxyHandler

from navigation_stage2_common import ROOT, read, write, digest
from check_navigation_awq_dart_request import prepare_input
from run_navigation_dart50_night import stop_owned

PYTHON = str(Path.home()/'.venvs/project2/bin/python')
OPENER = build_opener(ProxyHandler({}))


def gpu():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv,noheader'], text=True).strip()


def request(port, group, spec, payloads):
    payload = dict(payloads[spec['sample']])
    count = 8 if spec['phase'] == 'profile' else 64
    payload.update(model='quant-timing-'+group, max_tokens=count, min_new_tokens=count,
        ignore_eos=True, temperature=0, top_p=1, top_k=1, seed=7, stream=True,
        stream_options={'include_usage':True}, return_token_ids=True, skip_special_tokens=False)
    encoded = json.dumps(payload,ensure_ascii=False).encode()
    times, tokens, text, usage, reasons = [], [], '', None, []
    start = time.perf_counter()
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=encoded,
        headers={'Content-Type':'application/json'}, method='POST')
    with OPENER.open(req,timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            value = line[6:].strip()
            if value == b'[DONE]':
                break
            chunk = json.loads(value)
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices',[]):
                delta = choice.get('delta') or {}
                new = delta.get('gen_tokens') or []
                if new:
                    now = time.perf_counter()-start
                    times.extend([now]*len(new))
                    tokens.extend(new)
                text += delta.get('content') or ''
                if choice.get('finish_reason'):
                    reasons.append(choice['finish_reason'])
    elapsed = time.perf_counter()-start
    assert len(tokens) == count and usage['completion_tokens'] == count, (len(tokens),usage)
    assert usage['prompt_tokens'] == (701 if spec['sample']==0 else 2815), usage
    assert reasons == ['length'], reasons
    return dict(spec=spec, request_sha256=hashlib.sha256(encoded).hexdigest(), usage=usage,
        ttft_s=times[0], decode_s=times[-1]-times[0], decode_ms_per_token=1000*(times[-1]-times[0])/(count-1),
        total_s=elapsed, token_arrival_s=times, output_tokens=tokens, text=text, finish_reasons=reasons)


def summarize(run, sessions):
    all_rows, inputs = [], {}
    for session in sessions:
        directory = run/session['directory']
        for sample in (0,1):
            value = read(directory/'service'/f'input_{sample}.json')
            if sample in inputs:
                assert value == inputs[sample], 'Actual input tensors differ'
            inputs[sample] = value
        rows = read(directory/'requests.json')
        for i,row in enumerate(rows[:-1]):
            timing = read(directory/'service'/f'timing_{i:03d}.json')
            assert timing['index']==i and timing['plan']==row['spec']
            steps = timing['steps']
            assert steps[0]['phase']=='prefill' and all(s['phase']=='decode' and s['rows']==1 for s in steps[1:])
            assert steps[0]['rows']==row['usage']['prompt_tokens']
            if row['spec']['phase'] != 'measured':
                continue
            # All measured 64-token requests have 65 actual forwards. The
            # separate profiler run can change asynchronous lookahead and is
            # never mixed into latency measurements.
            assert len(steps) == row['usage']['completion_tokens'] + 1, (i,len(steps))
            pre,dec = steps[0],steps[1:]
            item = dict(group=session['group'], mode=session['mode'], block=session['block'],
                sample=row['spec']['sample'], http_total_s=row['total_s'], ttft_s=row['ttft_s'],
                returned_tokens=row['usage']['completion_tokens'], actual_decode_forwards=len(dec),
                stream_decode_ms_per_token=row['decode_ms_per_token'],
                prefill_model_ms=pre['timers']['model']['cuda_interval_ms'],
                prefill_vision_ms=pre['timers']['vision']['cuda_interval_ms'],
                prefill_language_ms=pre['timers']['language']['cuda_interval_ms'],
                prefill_through_logits_ms=pre['model_through_logits_ms'],
                decode_model_ms_per_token=statistics.mean(s['timers']['model']['cuda_interval_ms'] for s in dec),
                decode_through_logits_ms_per_token=statistics.mean(s['model_through_logits_ms'] for s in dec),
                decode_language_ms_per_token=statistics.mean(s['timers']['language']['cuda_interval_ms'] for s in dec),
                timing_source=(directory/'service'/f'timing_{i:03d}.json').relative_to(ROOT).as_posix())
            all_rows.append(item)
    summary = []
    for mode in ('audited','native'):
        for group in ('fp16','awq','gptq'):
            for sample in (0,1):
                selected=[x for x in all_rows if (x['mode'],x['group'],x['sample'])==(mode,group,sample)]
                assert len(selected)==(8 if mode=='audited' else 4)
                keys=[k for k in selected[0] if k.endswith(('_s','_ms','_per_token'))]
                summary.append(dict(mode=mode,group=group,sample=sample,n=len(selected),
                    medians={k:statistics.median(x[k] for x in selected) for k in keys},
                    ranges={k:[min(x[k] for x in selected),max(x[k] for x in selected)] for k in keys}))
    result=dict(status='complete',run=run.relative_to(ROOT).as_posix(), actual_input_tensors=inputs,
        summary=summary, measurements=all_rows, sessions=sessions,
        scope='Fixed 2/9 images,701/2815 prompt tokens,64 generated tokens; greedy output IDs may differ. Batch1 eager3072, no prefix/KV quant/token pruning.',
        timing_scope='CUDA event intervals include GPU idle/host dispatch gaps; stream decode includes scheduler/sampling/detokenization. Instrumented diagnostics are not a replacement for original closed-loop timings.',
        instrumentation='Audited mode preserves original stage2 adapter; native mode bypasses only diagnostic validation/hooks/logging using native forward and metadata; separate ablation.',
        counting_note='Every measured 64-token response has one prefill plus 64 observed decode forwards (65 total), including one extra backend forward. Event interval means include all 64 actual decode forwards; stream average is over 63 returned-token gaps. Separate profiler requests can have more asynchronous lookahead and are excluded. Initial exporter assumed 64 total and was corrected offline without rerunning any requests.',
        sources_sha256={p.name:digest(p) for p in [Path(__file__),ROOT/'checks/navigation_quant_timing_runtime.py',ROOT/'checks/serve_navigation_quant_timing.py',ROOT/'checks/navigation_quant_timing_module_map.py']})
    write(run/'result.json',result)
    print(json.dumps(dict(status='complete',run=result['run'],summary=summary),ensure_ascii=False),flush=True)


def main():
    name='navigation_quant_timing_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'logs'/name
    run.mkdir(exist_ok=False)
    payloads, input_records={},{}
    for sample in (0,1):
        source,settings,paths,hashes,encoded=prepare_input(1.,sample,'navigation')
        payloads[sample]=json.loads(encoded)
        input_records[sample]=dict(id=source['id'],images=paths,sha256=hashes)
    orders=[('audited',0,['fp16','awq','gptq']),('audited',1,['gptq','awq','fp16']),('native',0,['gptq','awq','fp16'])]
    write(run/'protocol.json',dict(inputs=input_records,orders=orders,warmups_per_sample=3,measured_per_sample_per_block=4,
        output_tokens=64,profile_output_tokens=8,started_utc=datetime.now(timezone.utc).isoformat()))
    sessions=[]
    print('QUANT_TIMING_RUN',run.relative_to(ROOT),flush=True)
    child=None
    try:
        for mode,block,groups in orders:
            for group in groups:
                index=len(sessions)
                directory=run/f'{index:02d}_{mode}_{group}'
                service=directory/'service'
                service.mkdir(parents=True)
                plan=[dict(phase='warmup',sample=s,repeat=r) for s in (0,1) for r in range(3)]
                plan += [dict(phase='measured',sample=s,repeat=r) for r in range(4) for s in (0,1)]
                if mode=='audited' and block==0:
                    plan += [dict(phase='profile',sample=1,repeat=0)]
                plan += [dict(phase='flush',sample=0,repeat=0)]
                write(service/'request_plan.json',plan)
                port=23440+index
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1',port))
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
                write(run/'status.json',dict(status='running',mode=mode,block=block,group=group,completed_sessions=len(sessions)),exclusive=False)
                with (directory/'console.log').open('x') as console:
                    child=subprocess.Popen([PYTHON,'-B','-u','checks/serve_navigation_quant_timing.py',
                        '--group',group,'--mode',mode,'--run-dir',str(service),'--port',str(port)],
                        cwd=ROOT,env=env,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
                write(directory/'ownership.json',dict(pid=child.pid,port=port))
                deadline=time.monotonic()+600
                while True:
                    if child.poll() is not None:
                        raise RuntimeError(f'Service exited: {directory}')
                    try:
                        with OPENER.open(f'http://127.0.0.1:{port}/v1/models',timeout=2) as response:
                            assert json.load(response)['data'][0]['id']=='quant-timing-'+group
                        break
                    except OSError:
                        if time.monotonic()>deadline:
                            raise TimeoutError('Service startup')
                        time.sleep(2)
                rows=[]
                before=gpu()
                for spec in plan:
                    row=request(port,group,spec,payloads)
                    rows.append(row)
                    write(directory/'requests.json',rows,exclusive=False)
                    if spec['phase']=='measured':
                        print('MEASURED',mode,block,group,spec['sample'],spec['repeat'],
                            round(row['ttft_s'],4),round(row['decode_ms_per_token'],3),flush=True)
                after=gpu()
                stop_owned(child)
                child=None
                session=dict(group=group,mode=mode,block=block,directory=directory.relative_to(run).as_posix(),gpu_before=before,gpu_after=after)
                sessions.append(session)
                write(run/'sessions.json',sessions,exclusive=False)
                print('SESSION_COMPLETE',mode,block,group,flush=True)
        summarize(run,sessions)
        write(run/'status.json',dict(status='complete',completed_sessions=len(sessions)),exclusive=False)
    except BaseException as error:
        write(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        write(run/'status.json',dict(status='failed',completed_sessions=len(sessions)),exclusive=False)
        raise
    finally:
        stop_owned(child)


if __name__=='__main__':
    main()
