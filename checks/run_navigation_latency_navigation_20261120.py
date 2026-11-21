"""New latency-candidate navigation checks on an externally frozen roster.

Uses the original evaluator and author policy unchanged. It never appends to
the old repair100 results or issues an old repair acceptance receipt. Complete
matching four-cell fixed-input evidence is required before a GPU run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import traceback

from navigation_stage2_common import ROOT, read, write, digest
from check_navigation_awq_dart_request import prepare_input
from run_navigation_repair_checked import audit_navigation, inspect_timing
from run_navigation_latency_study_20261120 import PYTHON, OPENER, FROZEN, gpu, stop_owned, request
from summarize_navigation_latency_20261120 import (audit_session, canonical, sha_object, execution_contract,
    wide_execution_audit, wide_output_gate)

POLICY_SHA = 'e73eb87dfa0946a42fc9b8efb4e1911ac41e382478a0ace24e5e5e19cd8132bc'
EVALUATOR_SHA = '5f8be04ae9b9b78461a8b7c3762b77ae637203d527e1e99c88d6f62a36a27caf'
AUTHOR_SHA = '72efa80174a832cec519eb7bb73ba2f635848b47af4dcaafaa03078e0d8d3a1f'


def require(condition, text):
    if not condition:
        raise ValueError(text)


def utc():
    return datetime.now(timezone.utc).isoformat()


def project_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    require(path.is_relative_to(ROOT), 'Path is outside project: '+str(path))
    return path


def verify_manifest(path):
    manifest = read(path)
    reference = read(ROOT/'data/baseline/navigation_repair_20261112/subset20.json')
    original = read(ROOT/'data/baseline/navigation_repair_20261112/original100.json')
    for key in ('temperature', 'top_p', 'max_tokens', 'seed', 'max_episode_steps', 'success_distance_m',
                'step_size_m', 'turn_deg', 'rgb_hw', 'camera_height_m', 'hfov_deg', 'model_inputs',
                'output_actions', 'action_parser', 'dataset_path', 'split', 'author_policy_sha256'):
        require(manifest[key] == reference[key], 'Frozen navigation condition differs: '+key)
    episodes, trajectories = list(map(str, manifest['episodes'])), list(map(str, manifest['trajectory_ids']))
    require(1 <= len(episodes) <= 20 and len(set(episodes)) == len(episodes) == len(trajectories),
            'Only an explicit unique roster of 1..20 cases is allowed')
    original_ids = list(map(str, original['episodes']))
    identity = dict(zip(original_ids, map(str, original['trajectory_ids'])))
    require(all(identity.get(eid) == trajectory for eid, trajectory in zip(episodes, trajectories)),
            'Episode/trajectory is not in original frozen100')
    require([eid for eid in original_ids if eid in episodes] == episodes, 'Roster order differs from frozen100')
    require(bool(manifest.get('selection')), 'External roster must describe its outcome-independent selection')
    for name, sha in [('checks/simplevln_policy.py', POLICY_SHA),
                      ('checks/run_navigation_precision_diagnostic.py', EVALUATOR_SHA),
                      ('tmp/habitat_setup/navigation_source_review/eval/vln/eval.py', AUTHOR_SHA)]:
        require(digest(ROOT/name) == sha, 'Frozen author policy/evaluator changed: '+name)
    return manifest


def normalize_sessions(config, default_entry):
    sessions = read(config)
    require(isinstance(sessions, list) and bool(sessions), 'Session configuration must be a nonempty list')
    labels = set()
    for session in sessions:
        require(session['group'] in ('fp16', 'awq_w4a16', 'gptq_w4a16'), 'Unsupported group')
        require(bool(re.fullmatch(r'[A-Za-z0-9_-]+', session['label'])) and session['label'] not in labels,
                'Session labels must be unique and filename-safe')
        labels.add(session['label'])
        session.setdefault('service_entry', default_entry)
        session.setdefault('model_name', 'navigation-repair-'+session['group'])
        require(project_path(session['service_entry']).is_file(), 'Service entry absent')
        require(all(isinstance(k, str) and isinstance(v, str) for k,v in session.get('env', {}).items()),
                'Session environment must contain strings')
        if session.get('frozen_table'):
            require(project_path(session['frozen_table']).is_file(), 'Frozen launch table absent')
    return sessions


def session_key(session):
    return dict(group=session['group'], service_entry=str(project_path(session['service_entry'])),
        model_name=session.get('model_name', 'navigation-repair-'+session['group']),
        env=session.get('env', {}), extra_args=session.get('extra_args', []),
        frozen_table_sha256=digest(project_path(session['frozen_table'])) if session.get('frozen_table') else None)


def fixed_evidence(paths, session):
    matches = []
    selected_audits = []
    for run in paths:
        run = project_path(run)
        require(read(run/'status.json')['status'] == 'complete', 'Fixed study is not complete: '+str(run))
        protocol = read(run/'protocol.json')
        if set(protocol['cells']) != {'0_18','0_64','1_18','1_64'} or protocol['measured_repeats'] < 10:
            continue
        if protocol['http_mode'] != 'nonstream':
            continue
        for number, tested in enumerate(protocol['sessions']):
            if session_key(tested) != session_key(session):
                continue
            audited = audit_session(run, protocol, number, tested)
            selected_audits.append(audited)
            require(all(c['output_sequences_identical'] for c in audited['cells'].values()),
                    'Matched fixed study has unstable repeated outputs')
            # The new cohort must run the source bytes that produced the fixed evidence.
            for name, item in audited['source_verification'].items():
                require(digest(ROOT/'checks'/name) == item['sha256'], 'Runtime changed since fixed study: '+name)
            matches.append(dict(study=str(run.relative_to(ROOT)), fixed_session=audited['id'],
                fixed_result_sha256=digest(ROOT/audited['id']/'result.json'),
                engine=audited['engine'], model_weights_sha256=audited['model_weights_sha256'],
                requested_engine=audited['requested_engine'], execution_identity=audited['execution_identity'],
                frozen_launch_configuration_sha256=audited['frozen_launch_configuration_sha256'],
                wide_evidence=audited['wide_evidence'],
                vision_qmax_evidence=audited['vision_qmax_evidence'],
                actual_input_tensors=audited['actual_input_tensors'],
                source_sha256={name:item['sha256'] for name,item in audited['source_verification'].items()},
                scope='Same-source/effective-configuration four-cell nonstream evidence; not a repair100 acceptance receipt'))
    require(bool(matches), 'No complete matching nonstream four-cell fixed study for '+session['label'])
    if selected_audits[-1]['wide_evidence']:
        native = []
        for run in paths:
            run = project_path(run); protocol = read(run/'protocol.json')
            for number,tested in enumerate(protocol['sessions']):
                if tested['group'] == session['group'] and not tested.get('env',{}).get('PROJECT2_LATENCY_W4_TABLE'):
                    native.append(audit_session(run,protocol,number,tested))
        gate = wide_output_gate(selected_audits[-1],native+[selected_audits[-1]])
        require(gate['status'] == 'verified', 'Wide fixed evidence lacks same-engine native output equivalence: '+canonical(gate))
        matches[-1]['wide_native_output_gate'] = gate
    return matches[-1]


def trajectory_evidence(folder, episodes):
    result = {}
    for eid in map(str, episodes):
        rows = [json.loads(line) for line in (folder/('episode_'+eid)/'steps.jsonl').read_text().splitlines()]
        require(rows and rows[0]['model_called'], 'Episode initial model request missing: '+eid)
        first = rows[0]
        initial = dict(request_text=first['request_text'], request_image_sha256=first['request_image_sha256'],
            image_count=first['image_count'], prompt_tokens=first['usage']['prompt_tokens'])
        semantic = [{key:r.get(key) for key in ('step','action','raw','error','model_called','finish_reason',
            'usage','request_text','request_image_sha256','image_count','before','after','metrics')} for r in rows]
        result[eid] = dict(initial_input=initial, initial_input_sha256=sha_object(initial),
            action_sequence_sha256=sha_object([r['action'] for r in rows]),
            model_outputs_sha256=sha_object([r['raw'] for r in rows if r['model_called']]),
            complete_semantic_trajectory_sha256=sha_object(semantic),
            primitive_actions=len(rows), model_requests=sum(r['model_called'] for r in rows))
    return result


def compare_trajectories(reference, candidate):
    require(set(reference) == set(candidate), 'Compared rosters differ')
    rows = {}
    for eid in reference:
        a, b = reference[eid], candidate[eid]
        rows[eid] = dict(initial_inputs_equal=a['initial_input_sha256'] == b['initial_input_sha256'],
            action_sequences_equal=a['action_sequence_sha256'] == b['action_sequence_sha256'],
            model_outputs_equal=a['model_outputs_sha256'] == b['model_outputs_sha256'],
            complete_semantic_trajectory_equal=a['complete_semantic_trajectory_sha256'] == b['complete_semantic_trajectory_sha256'])
    return dict(episodes=rows, all_initial_inputs_equal=all(r['initial_inputs_equal'] for r in rows.values()),
        all_actions_equal=all(r['action_sequences_equal'] for r in rows.values()),
        all_model_outputs_equal=all(r['model_outputs_equal'] for r in rows.values()),
        all_semantic_trajectories_equal=all(r['complete_semantic_trajectory_equal'] for r in rows.values()))


def run_session(run, number, session, manifest, evidence, port, timeout):
    directory = run/f"{number:02d}_{session['label']}"
    service = directory/'service'; service.mkdir(parents=True)
    # The unchanged runtime uses labels only while index < len(plan), then None.
    # This prefix enables optional QMAX numerical gates only in excluded smokes;
    # evaluator warmups and actual navigation keep their natural request labels.
    write(service/'request_plan.json',[dict(phase='input_warmup',sample=i,tokens=0,repeat=0) for i in (0,1)])
    env = {k:v for k,v in os.environ.items() if not k.startswith('PROJECT2_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LIBGL_ALWAYS_SOFTWARE='1')
    env['PATH'] = '/usr/lib/wsl/lib:'+env.get('PATH', '')
    env.pop('DISPLAY', None); env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(session.get('env', {}))
    if session.get('frozen_table'):
        env['PROJECT2_W4A16_REPAIR_TABLE'] = str(project_path(session['frozen_table']))
    command = [PYTHON, '-B', '-u', str(project_path(session['service_entry'])), '--group', session['group'],
               '--run-dir', str(service), '--port', str(port)]+session.get('extra_args', [])
    write(directory/'fixed_evidence_binding.json', evidence)
    write(directory/'command.json', dict(argv=command, configuration=session_key(session)))
    with socket.socket() as probe: probe.bind(('127.0.0.1', port))
    child = evaluator = None
    try:
        with (directory/'service_console.log').open('x') as console:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write(directory/'ownership.json', dict(service_pid=child.pid, service_process_group=child.pid,
                                               controller_pid=os.getpid(), port=port, created_utc=utc()))
        deadline = time.monotonic()+timeout
        while True:
            require(child.poll() is None, 'Service exited during startup')
            try:
                with OPENER.open(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    require(json.load(response)['data'][0]['id'] == session['model_name'], 'Served model differs')
                break
            except OSError:
                require(time.monotonic() < deadline, 'Service startup timeout'); time.sleep(2)
        launch = read(service/'launch.json')
        require(launch['engine'] == evidence['requested_engine'], 'Navigation requested engine differs from fixed study')
        require(launch['model_weights_sha256'] == evidence['model_weights_sha256'], 'Navigation weights differ')
        require(launch['sources_sha256'] == evidence['source_sha256'], 'Navigation sources differ from fixed study')
        require(read(service/'execution_audit.json')['first_native_forward_verified'], 'Native forward missing')
        require(read(service/'startup_graph_capture.json')['actual_capture_observed'], 'Decode graph capture missing')
        payloads = {s:json.loads(prepare_input(1., s, 'navigation')[-1]) for s in (0, 1)}
        smoke = []
        from navigation_repair_output_gate import inspect_answer
        for sample in (0, 1):
            spec = dict(phase='input_warmup', sample=sample, tokens=0, repeat=0)
            row = request(port, session['model_name'], spec, payloads, 'nonstream', 1.)
            gate = inspect_answer(row['response']); require(gate['passed'], 'Natural smoke failed')
            smoke.append(row); write(directory/'input_warmups.json', smoke, exclusive=False)
            write(directory/f'natural_output_gate_{sample}.json', gate)
        for length in (701, 2815):
            value = read(service/f'input_{length}.json')
            actual = {k:value[k] for k in ('input_ids','pixel_values')}
            require(actual == evidence['actual_input_tensors'][str(length)], 'Fixed input tensors differ from speed study')
        # Transport execution receipts exist only after real image requests.
        contract = execution_contract(service, launch,
            {name:dict(sha256=sha) for name,sha in launch['sources_sha256'].items()})
        require(contract['effective_engine'] == evidence['engine'], 'Navigation effective engine differs from fixed study')
        require(contract['execution_identity'] == evidence['execution_identity'], 'Navigation execution identity differs from fixed study')
        if contract['wide_evidence']:
            table = read(service/'repair_execution_audit.json')['frozen_launch_table']
            actual = sha_object(dict(shapes=table['shapes'],representatives=table['representatives'],max_rows=table['max_rows']))
            require(actual == evidence['frozen_launch_configuration_sha256'], 'Navigation native fallback launch configurations differ')
            require(table['frozen_table_source']['sha256'] == contract['wide_evidence']['numerical_probe']['baseline_table_sha256'],
                    'Navigation native fallback differs from numerically checked baseline')
        actual_manifest = dict(manifest)
        # Descriptive metadata only; evaluator policy, roster and budget remain frozen.
        actual_manifest.update(model=session['model_name'], latency_study_group=session['group'],
            dtype='FP16' if session['group'] == 'fp16' else 'W4A16',
            latency_study_configuration=session_key(session),
            scope='Independent latency candidate navigation validation; never merged into prior repair100')
        write(directory/'case_manifest.json', actual_manifest)
        folder = directory/'navigation'
        evaluator_command = ['bash','scripts/run_habitat_task_python.sh','-u',
            'checks/run_navigation_precision_diagnostic.py','--policy','memory',
            '--manifest',str(directory/'case_manifest.json'),'--output',str(folder),
            '--endpoint',f'http://127.0.0.1:{port}/v1']
        with (directory/'navigation_console.log').open('x') as console:
            evaluator = subprocess.Popen(evaluator_command, cwd=ROOT, env=env, stdout=console,
                                         stderr=subprocess.STDOUT, start_new_session=True)
        write(directory/'evaluator_ownership.json', dict(pid=evaluator.pid, process_group=evaluator.pid,
                                                        argv=evaluator_command, created_utc=utc()))
        while evaluator.poll() is None:
            require(child.poll() is None, 'Service exited during navigation'); time.sleep(3)
        require(evaluator.returncode == 0, 'Evaluator failed; retained console and partial trajectories')
        tail = request(port, session['model_name'], dict(phase='flush',sample=0,tokens=18,repeat=0),
                       payloads, 'nonstream', 1.)
        write(directory/'tail_request.json', tail)
        # Two explicit smoke requests precede the evaluator's own two warmups.
        result = audit_navigation(folder, service, first_index=2)
        for index, row in enumerate(smoke): inspect_timing(service, index, row['usage'])
        if contract['vision_qmax_evidence']:
            # Reaudit any additional first-grid receipts created by navigation;
            # there must still be exactly the two excluded numerical warmups.
            contract = execution_contract(service,launch,
                {name:dict(sha256=sha) for name,sha in launch['sources_sha256'].items()})
            require(contract['execution_identity'] == evidence['execution_identity'], 'Navigation QMAX execution identity changed')
            for grid in contract['vision_qmax_evidence']['first_grid_coverage']:
                index = grid['request_index']
                require(index < result['timing_end_exclusive'], 'QMAX grid receipt is outside completed navigation requests')
                timing = read(service/f'timing_{index:03d}.json')
                prefills = [step for step in timing['steps'] if step['phase'] == 'prefill']
                require(timing['index'] == index and len(prefills) == 1 and prefills[0]['rows'] == grid['input_tokens'],
                        'Navigation QMAX first-grid receipt differs from actual prefill timing')
        wide_coverage = []
        if contract['wide_evidence']:
            native = read(service/'execution_audit.json')['native_awq_receipt']
            for index in range(result['timing_end_exclusive']):
                timing = read(service/f'timing_{index:03d}.json')
                wide_coverage.append(wide_execution_audit(service,index,timing,contract,native))
            require({701,2815}.issubset({v['prefill_m'] for v in wide_coverage}), 'Navigation wide mandatory smoke coverage missing')
            write(directory/'wide_request_coverage.json',wide_coverage)
        result.update(scope='New latency-candidate cohort, not prior repair acceptance', session=session,
            fixed_evidence_binding=evidence, passed_stability=result['error_count'] == 0 and not result['warmup_errors'],
            trajectory_evidence=trajectory_evidence(folder, manifest['episodes']),
            source_sha256=launch['sources_sha256'], timing_navigation_first_warmup=2,
            execution_contract=contract,
            wide_request_coverage=wide_coverage,
            wide_fallback_scope='Only actual prefill lengths absent from frozen exact-M table; all decode graph captures retain selected phase route',
            timing_navigation_first_actual_request=4, quality_equivalence_claim=False)
        if session.get('historical_navigation'):
            old_folder = project_path(session['historical_navigation'])
            old_manifest = read(old_folder/'manifest.json')
            require(old_manifest.get('repair_group') == session['group'], 'Historical reference group differs')
            old = trajectory_evidence(old_folder, manifest['episodes'])
            result['historical_trajectory_comparison'] = dict(reference=str(old_folder.relative_to(ROOT)),
                reference_manifest_sha256=digest(old_folder/'manifest.json'),
                comparison=compare_trajectories(old, result['trajectory_evidence']),
                scope='Historical cross-round comparison; not a same-round speed baseline or statistical quality-equivalence proof')
        for name, sha in launch['sources_sha256'].items():
            require(digest(ROOT/'checks'/name) == sha, 'Runtime source changed during navigation: '+name)
        require(digest(ROOT/'checks/simplevln_policy.py') == POLICY_SHA and
                digest(ROOT/'checks/run_navigation_precision_diagnostic.py') == EVALUATOR_SHA, 'Policy/evaluator changed')
        write(directory/'result.json', result)
        return result
    finally:
        stop_owned(evaluator)
        stop_owned(child)
        write(directory/'cleanup.json', dict(completed_utc=utc(), service_pid=child.pid if child else None,
                                             evaluator_pid=evaluator.pid if evaluator else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--session-config', required=True, type=Path)
    parser.add_argument('--fixed-study', nargs='+', type=Path, default=[])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--service-entry', default='checks/serve_navigation_latency_20261120.py')
    parser.add_argument('--base-port', type=int, default=24410)
    parser.add_argument('--startup-timeout', type=int, default=1200)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = verify_manifest(args.manifest)
    sessions = normalize_sessions(args.session_config, args.service_entry)
    evidence = [fixed_evidence(args.fixed_study, session) for session in sessions] if args.fixed_study else []
    if args.check_only:
        print(canonical(dict(status='cpu_configuration_checked', case_count=len(manifest['episodes']),
            sessions=[s['label'] for s in sessions], fixed_evidence_bound=len(evidence),
            policy_sha256=POLICY_SHA, evaluator_sha256=EVALUATOR_SHA, gpu_started=False)))
        return
    require(args.output is not None and len(evidence) == len(sessions), 'A fresh output and matching fixed-study evidence are required')
    require(1024 <= args.base_port < 65536-len(sessions), 'Invalid port range')
    run = args.output.resolve()
    require(run.is_relative_to(ROOT/'logs') and not run.exists(), 'Use a fresh logs directory')
    import fcntl
    locks = []
    for path in (ROOT/'logs/navigation_latency_study_20261120.lock', ROOT/'tmp/navigation_repair_gpu.lock'):
        lock = path.open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
    run.mkdir(parents=True)
    completed = []
    try:
        names = ['checks/run_navigation_latency_navigation_20261120.py', 'checks/run_navigation_latency_study_20261120.py',
            'checks/summarize_navigation_latency_20261120.py', 'checks/run_navigation_repair_checked.py',
            'checks/run_navigation_precision_diagnostic.py', 'checks/simplevln_policy.py', 'checks/simplevln_env.py',
            'checks/navigation_precision_quality.py', 'checks/run_closed_loop_baseline.py',
            'scripts/run_habitat_task_python.sh', 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py']
        names += ['checks/'+name for e in evidence for name in e['source_sha256']]
        snapshot = run/'source_snapshot'; snapshot.mkdir()
        sources = {}
        for name in sorted(set(names)):
            target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name,target); sources[name] = digest(target)
        write(run/'original_manifest.json', manifest)
        write(run/'protocol.json', dict(created_utc=utc(), scope='Independent latency-candidate navigation validation',
            sessions=sessions, original_manifest_sha256=digest(args.manifest), case_count=len(manifest['episodes']),
            source_sha256=sources, author_policy_sha256=AUTHOR_SHA, fixed_study_bindings=evidence,
            gpu_before=gpu(), navigation='Original author policy, temperature0.3/top_p0.95/max512/seed7/400steps; no added idle',
            timing='Two excluded explicit shape smokes, two excluded evaluator warmups; real HTTP only; final tail excluded',
            does_not_extend_prior_repair100=True, selection='Only externally supplied frozen manifest; no outcome-based selection in this controller'))
        for number, session in enumerate(sessions):
            write(run/'status.json', dict(status='running', updated_utc=utc(), current_session=session,
                  completed_sessions=len(completed)), exclusive=False)
            result = run_session(run,number,session,manifest,evidence[number],args.base_port+number,args.startup_timeout)
            for previous in completed:
                comparison = compare_trajectories(previous['trajectory_evidence'],result['trajectory_evidence'])
                require(comparison['all_initial_inputs_equal'], 'Cross-session initial navigation inputs differ')
                result.setdefault('same_run_trajectory_comparisons',[]).append(dict(reference=previous['session']['label'],
                    same_weights=previous['session']['group'] == session['group'], comparison=comparison))
            completed.append(result)
            write(run/'completed.json',completed,exclusive=False)
            for name, sha in sources.items(): require(digest(ROOT/name)==sha,'Source changed during cohort: '+name)
        write(run/'result.json',dict(status='complete',completed_utc=utc(),sessions=completed,
            scope='Independent small-cohort validation; historical and fixed-input results remain separate'))
        write(run/'status.json',dict(status='complete',updated_utc=utc(),completed_sessions=len(completed)),exclusive=False)
    except BaseException as error:
        write(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),at_utc=utc()))
        write(run/'status.json',dict(status='failed',updated_utc=utc(),completed_sessions=len(completed)),exclusive=False)
        raise
    finally:
        for lock in locks: fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__ == '__main__':
    main()
