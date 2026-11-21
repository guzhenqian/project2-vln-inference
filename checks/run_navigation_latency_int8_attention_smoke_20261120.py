"""Owned single-service qkv/o new-precision smoke; two natural inputs and tail.

No formal measurements or quality claims. Existing native fixed evidence binds
inputs/engine/fallback provenance only. All HTTP errors retain their raw bodies.
Run explicitly with project WSL Python, outside all other GPU work.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback
from urllib.error import HTTPError,URLError
from urllib.request import Request,build_opener,ProxyHandler

import audit_navigation_latency_int8_attention_smoke_20261120 as audit
from navigation_latency_int8_attention_contract_20261120 import load_contract,CHECKPOINT

ROOT=Path(__file__).resolve().parents[1]
OPENER=build_opener(ProxyHandler({}))
H=None


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w' if replace else 'x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def load_helpers():
    # Deliberately deferred: no preparation/old runner imports in --help or AST QA.
    from types import SimpleNamespace
    from run_navigation_latency_navigation_20261120 import normalize_sessions,fixed_evidence,session_key
    from check_navigation_awq_dart_request import prepare_input
    from run_navigation_latency_study_20261120 import PYTHON,stop_owned,gpu
    return SimpleNamespace(**locals())


def project_path(value):
    path=Path(value);path=path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    audit.require(path.is_relative_to(ROOT),'Project path required');return path


def native_binding(config,studies):
    sessions=H.normalize_sessions(config,'checks/serve_navigation_latency_20261120.py')
    matches=[s for s in sessions if s['group']=='awq_w4a16' and s.get('env',{}).get('PROJECT2_LATENCY_EXECUTOR')=='uni' and
        s.get('env',{}).get('PROJECT2_LATENCY_INPROCESS')=='1' and s.get('env',{}).get('PROJECT2_LATENCY_PREFILL_INTERVAL')=='2' and
        not any(s.get('env',{}).get(k) not in (None,'','0','off') for k in ('PROJECT2_LATENCY_W4_TABLE','PROJECT2_LATENCY_HYBRID_TABLE',
            'PROJECT2_LATENCY_VISION_QMAX','PROJECT2_LATENCY_VISION_CACHE','PROJECT2_LATENCY_LINEAR_IPC','PROJECT2_LATENCY_PROFILE'))]
    audit.require(len(matches)==1,'Require one AWQ native inprocess2 reference configuration')
    reference=matches[0];audit.require(reference.get('frozen_table'),'Native AWQ frozen fallback table is required')
    evidence=H.fixed_evidence(studies,reference)
    audit.require(not evidence['wide_evidence'] and not evidence['vision_qmax_evidence'],'Reference must use original native kernels and vision')
    launch_path=ROOT/evidence['fixed_session']/'service/launch.json';launch=audit.read(launch_path)
    audit.require(launch['model_weights_sha256']==evidence['model_weights_sha256']==CHECKPOINT and
        launch['engine']==evidence['requested_engine'],'Bound native launch differs from audited provenance')
    return dict(reference_session=reference,configuration_key=H.session_key(reference),evidence=evidence,
        native_launch_sha256=audit.digest(launch_path),native_model_config_sha256=launch['model_config_sha256'],
        scope='Native input/engine/fallback provenance only; this is not approval of mixed W8A8-prefill/W4-decode precision')


def retain_http_error(run,index,phase,error):
    body=error.read();path=run/f'http_error_{phase}_{index:03d}.bin'
    with path.open('wb') as stream:stream.write(body)
    record=dict(index=index,phase=phase,status=error.code,reason=str(error.reason),
        body_path=path.name,body_sha256=hashlib.sha256(body).hexdigest(),body_bytes=len(body),
        body_utf8=body.decode('utf-8',errors='replace'),headers=dict(error.headers.items()),created_utc=utc())
    write(path.with_suffix('.json'),record,replace=path.with_suffix('.json').exists())
    return record


def post(run,port,index,payload):
    spec=audit.PLAN[index];value=dict(payload,model='navigation-repair-awq_w4a16',seed=7,stream=False,return_token_ids=True)
    if spec['tokens']:
        value.update(max_tokens=18,min_new_tokens=18,ignore_eos=True,temperature=0,top_p=1,top_k=1)
    raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8');semantic=dict(value);semantic.pop('model')
    write(run/f'request_payload_{index:03d}.json',value)
    start=time.perf_counter()
    try:
        with OPENER.open(Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'}),timeout=600) as response:
            body=response.read();status=response.status
    except HTTPError as error:
        retain_http_error(run,index,'request',error);raise
    elapsed=time.perf_counter()-start
    # Persist the exact body even if JSON decoding or semantic checks fail.
    (run/f'response_{index:03d}.bin').write_bytes(body)
    try:answer=json.loads(body)
    except Exception:
        write(run/f'response_decode_failed_{index:03d}.json',dict(status=status,body_sha256=hashlib.sha256(body).hexdigest(),body_bytes=len(body)));raise
    tokens=answer.get('choices',[{}])[0].get('message',{}).get('gen_tokens') or []
    row=dict(index=index,spec=spec,http_mode='nonstream',idle_seconds_outside_timer=0,http_total_s=elapsed,
        diagnostic_timing_excluded_from_benchmark=True,request_sha256=hashlib.sha256(raw).hexdigest(),semantic_payload_sha256=audit.native.sha_object(semantic),
        response=answer,response_body_utf8=body.decode('utf-8'),response_bytes_sha256=hashlib.sha256(body).hexdigest(),
        usage=answer.get('usage',{}),output_tokens=tokens,output_tokens_sha256=audit.native.sha_object(tokens))
    write(run/f'response_record_{index:03d}.json',row)
    return row


def snapshot_sources(run,binding):
    names={'checks/'+name for name in binding['evidence']['source_sha256']}
    names.update({'checks/'+name for name in (audit.ENTRY,audit.ADAPTER,audit.MODULE_MAP,
        'navigation_latency_int8_attention_contract_20261120.py','navigation_latency_expanded_int8_prefill_20261120.py',
        'probe_navigation_latency_expanded_int8_prefill_20261120.py','run_navigation_latency_int8_attention_smoke_20261120.py',
        'audit_navigation_latency_int8_attention_smoke_20261120.py','navigation_repair_output_gate.py',
        'run_navigation_latency_navigation_20261120.py','run_navigation_latency_study_20261120.py','summarize_navigation_latency_20261120.py',
        'check_navigation_awq_dart_request.py','simplevln_policy.py')})
    names.add(audit.AUTHOR);result={}
    for name in sorted(names):
        target=run/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);result[name]=audit.digest(target)
    return result


def run_smoke(args):
    global H
    audit.require(os.name=='posix','Use project WSL Python')
    run=project_path(args.output);probe=project_path(args.probe)
    audit.require(run.is_relative_to(ROOT/'logs') and not run.exists(),'Use a fresh logs directory')
    audit.require(1024<=args.port<65536,'Invalid service port')
    import fcntl
    locks=[];child=None;rows=[];samples=[];failure=None
    try:
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock',ROOT/'tmp/navigation_repair_gpu.lock'):
            lock=path.open('a+')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:lock.close();raise
            locks.append(lock)
        run.mkdir(parents=True);service=run/'service';service.mkdir()
        write(run/'status.json',dict(status='preparing',created_utc=utc()))
        H=load_helpers();binding=native_binding(project_path(args.native_reference_config),[project_path(v) for v in args.fixed_study])
        proof=load_contract(probe,'awq',CHECKPOINT,verify_sources=True)
        env=dict(PROJECT2_LATENCY_EXECUTOR='uni',PROJECT2_LATENCY_PREFILL_INTERVAL='2',PROJECT2_LATENCY_INPROCESS='1',
            PROJECT2_LATENCY_INT8_ATTENTION='qkv_o',PROJECT2_LATENCY_INT8_PROBE=str(probe))
        sources=snapshot_sources(run,binding)
        write(run/'native_reference_binding.json',binding);write(run/'micro_evidence_binding.json',proof)
        write(service/'request_plan.json',audit.PLAN)
        write(run/'protocol.json',dict(schema='navigation_int8_attention_actual_smoke/v1',created_utc=utc(),
            native_reference_binding=binding,candidate_environment=env,source_sha256=sources,
            measured_requests=0,excluded_input_warmups=2,tail_requests=1,no_latency_claim=True,no_quality_claim=True,
            original_w4_equivalence_claim=False,navigation_success_claim=False,outputs_fed_back=False,
            input_scope='Original fixed two/nine images; natural responses, two exact input tensor receipts; excluded tail flush',
            numerical_scope='72 same-candidate-activation local references per smoke at unchanged2%; not full-model accuracy'))
        payloads={i:json.loads(H.prepare_input(1.,i,'navigation')[-1]) for i in (0,1)}
        for i,payload in payloads.items():write(run/f'prepared_input_{i}.json',payload)
        process_env={k:v for k,v in os.environ.items() if not k.startswith('PROJECT2_')}
        process_env.update(env);process_env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        process_env['PATH']='/usr/lib/wsl/lib:'+process_env.get('PATH','')
        process_env['PROJECT2_W4A16_REPAIR_TABLE']=str(project_path(binding['reference_session']['frozen_table']))
        command=[H.PYTHON,'-B','-u',str(ROOT/'checks'/audit.ENTRY),'--group','awq_w4a16','--run-dir',str(service),'--port',str(args.port)]
        write(run/'command.json',dict(argv=command,candidate_environment=env,frozen_table_path=process_env['PROJECT2_W4A16_REPAIR_TABLE'],
            frozen_table_sha256=audit.digest(process_env['PROJECT2_W4A16_REPAIR_TABLE'])))
        with socket.socket() as check:check.bind(('127.0.0.1',args.port))
        samples.append(dict(phase='before_start',at_utc=utc(),value=H.gpu()));write(run/'gpu_samples.json',samples)
        with (run/'service_console.log').open('x',encoding='utf-8') as console:
            child=subprocess.Popen(command,cwd=ROOT,env=process_env,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
        write(run/'ownership.json',dict(controller_pid=os.getpid(),service_pid=child.pid,service_process_group=child.pid,port=args.port,created_utc=utc()))
        write(run/'status.json',dict(status='starting',service_pid=child.pid,updated_utc=utc()),replace=True)
        deadline=time.monotonic()+args.startup_timeout;startup_errors=0
        while True:
            audit.require(child.poll() is None,'Owned service exited during startup; inspect retained console and failure receipts')
            try:
                with OPENER.open(f'http://127.0.0.1:{args.port}/v1/models',timeout=3) as response:models=json.load(response)
                audit.require(models['data'][0]['id']=='navigation-repair-awq_w4a16','Unexpected service model');break
            except HTTPError as error:
                retain_http_error(run,startup_errors,'startup',error);startup_errors+=1
            except (URLError,ConnectionError,TimeoutError):pass
            audit.require(time.monotonic()<deadline,'Startup timed out');time.sleep(2)
        audit.audit_launch(service,binding,env)
        samples.append(dict(phase='after_start',at_utc=utc(),value=H.gpu()));write(run/'gpu_samples.json',samples,replace=True)
        for index,spec in enumerate(audit.PLAN):
            row=post(run,args.port,index,payloads[spec['sample']]);rows.append(row);write(run/'requests.json',rows,replace=True)
            gate=audit.audit_response(row,index);write(run/f'output_gate_{index:03d}.json',gate)
            if index<2:
                m=(701,2815)[index];actual=audit.read(service/f'input_{m}.json')
                audit.require(actual['request_index']==index and {k:actual[k] for k in ('input_ids','pixel_values')}==binding['evidence']['actual_input_tensors'][str(m)],
                    'Smoke actual input tensors differ from native provenance')
                audit.audit_smoke(audit.read(service/f'int8_attention_smoke_{index:03d}.json'),index,actual)
            samples.append(dict(phase=f'after_request_{index}',at_utc=utc(),value=H.gpu()));write(run/'gpu_samples.json',samples,replace=True)
        for name,value in sources.items():audit.require(audit.digest(ROOT/name)==value,'Runtime changed during smoke: '+name)
        write(run/'status.json',dict(status='requests_complete_pending_audit',updated_utc=utc()),replace=True)
    except BaseException as error:
        failure=error
        if run.exists():
            write(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_requests=len(rows),at_utc=utc()),replace=True)
            write(run/'status.json',dict(status='failed',completed_requests=len(rows),updated_utc=utc()),replace=True)
    finally:
        if child is not None:
            H.stop_owned(child)
        if run.exists():
            write(run/'cleanup.json',dict(service_pid=child.pid if child else None,owned_process_exited=child is None or child.poll() is not None,
                returncode=child.returncode if child else None,completed_utc=utc()),replace=True)
        for lock in locks:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if run.exists():
        checked=audit.audit_run(run);write(run/'audit.json',checked)
        if checked['status']!='verified_actual_smoke' and failure is None:
            failure=RuntimeError(checked['audit_error'])
            write(run/'failure.json',dict(error=repr(failure),completed_requests=len(rows),at_utc=utc()))
            write(run/'status.json',dict(status='failed',completed_requests=len(rows),updated_utc=utc()),replace=True)
        if failure is None:
            write(run/'result.json',dict(status='smoke_only',actual_smoke_audit_status=checked['status'],audit_sha256=audit.digest(run/'audit.json'),
                excluded_input_warmups=2,tail_requests=1,measured_requests=0,benefit_claim_allowed=False,no_latency_claim=True,no_quality_claim=True,
                full_model_accuracy_approved=False,original_w4_output_equivalence_claim=False))
            write(run/'status.json',dict(status='smoke_only',completed_requests=3,updated_utc=utc()),replace=True)
            print(json.dumps(dict(status='smoke_only',audit=checked['status'],output=str(run))))
    if failure is not None:raise failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True);parser.add_argument('--native-reference-config',type=Path,required=True)
    parser.add_argument('--fixed-study',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=24830);parser.add_argument('--startup-timeout',type=int,default=1800)
    run_smoke(parser.parse_args())


if __name__=='__main__':main()
