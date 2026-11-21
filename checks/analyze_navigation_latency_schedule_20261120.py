"""Stdlib-only analysis of the independent schedule timestamp diagnostic.

Merge all chunks per PID before matching requests: a trace file is not a
request boundary. HTTP times from this instrumented run are never exported as
speed results. Host model-block end is not CUDA completion.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


EXPECTED_TRACE_SOURCE = {
    'engine_loop.py': 'ed2a1c26751785c9f94a8c12202d82bdb30b96e741f73506e5f1058543d50f41',
    'model_agent.py': '7844f9e2a6f05dc7b1356860e6f669ca3b57268632066837b37f9f85df404a27',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def union(intervals):
    merged = []
    for start, end in sorted((s, e) for s, e in intervals if e > s):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def intersect(left, right):
    return union((max(a, c), min(b, d)) for a, b in left for c, d in right)


def ms(intervals):
    return sum(e-s for s, e in union(intervals))/1e6


def event_ref(event):
    return dict(source=event['_source'], source_event_index=event['_ordinal'],
                pid=event['_pid'], time_ns=event['time_ns'])


def load_events(service):
    files = sorted(service.glob('schedule_trace_*.json'))
    need(files, f'No schedule trace in {service}')
    by_pid = defaultdict(list)
    source_contract = None
    hashes = {}
    for path in files:
        data = read(path)
        need(data.get('dropped_events') == 0, f'Dropped/missing event count in {path}')
        need(isinstance(data.get('pid'), int), f'No PID in {path}')
        contract = data.get('source_sha256')
        need(contract == EXPECTED_TRACE_SOURCE, f'Unexpected source contract in {path}')
        if source_contract is None:
            source_contract = contract
        need(contract == source_contract, 'Trace source hashes changed')
        prior = -1
        for ordinal, raw in enumerate(data['events']):
            event = dict(raw, _pid=data['pid'], _source=path.name, _ordinal=ordinal)
            need(isinstance(event.get('time_ns'), int) and event['time_ns'] >= prior,
                 f'Invalid/nonmonotonic event time in {path}')
            prior = event['time_ns']
            by_pid[data['pid']].append(event)
        hashes[path.name] = digest(path)
    for events in by_pid.values():
        events.sort(key=lambda e: (e['time_ns'], e['_source'], e['_ordinal']))
    return by_pid, source_contract, hashes


def timelines(events):
    waits, blocks, unmatched, closed = [], [], [], defaultdict(list)
    pending_wait, pending_block, closed_since = {}, {}, {}
    last_time = events[-1]['time_ns']
    for e in events:
        name, t = e['event'], e['time_ns']
        eid = e.get('event_id')
        if name == 'engine_loop_created':
            # Native CounterEvent inherits asyncio.Event's initially-unset state.
            closed_since[eid] = t
        elif name in ('counter_set_after', 'counter_clear_after'):
            if e['is_set']:
                if eid in closed_since:
                    closed[eid].append([closed_since.pop(eid), t])
            else:
                closed_since.setdefault(eid, t)
        elif name == 'counter_wait_begin':
            key = (eid, e.get('task'))
            need(key not in pending_wait, f'Overlapping wait on same event/task: {key}')
            pending_wait[key] = e
        elif name == 'counter_wait_end':
            key = (eid, e.get('task'))
            begin = pending_wait.pop(key, None)
            need(begin is not None, f'Wait end without begin: {event_ref(e)}')
            waits.append(dict(event_id=eid, task=e.get('task'), begin_ns=begin['time_ns'], end_ns=t,
                              counter_at_entry=begin['counter'], event_set_at_entry=begin['is_set'],
                              begin=event_ref(begin), end=event_ref(e)))
        elif name == 'model_block_begin':
            key = e['agent']
            need(key not in pending_block, 'Overlapping blocks for one model agent')
            pending_block[key] = e
        elif name in ('model_block_end', 'model_block_error'):
            begin = pending_block.pop(e['agent'], None)
            need(begin is not None, f'Block end without begin: {event_ref(e)}')
            blocks.append(dict(agent=e['agent'], begin_ns=begin['time_ns'], end_ns=t,
                               is_decoding=begin['is_decoding'], loop_count=begin['loop_count'],
                               status='error' if name.endswith('error') else 'host_coroutine_returned',
                               begin=event_ref(begin), end=event_ref(e)))
    for key, begin in pending_wait.items():
        unmatched.append(dict(kind='unfinished_wait', event_id=key[0], task=key[1], begin=event_ref(begin)))
    for key, begin in pending_block.items():
        blocks.append(dict(agent=key, begin_ns=begin['time_ns'], end_ns=None,
                           is_decoding=begin['is_decoding'], loop_count=begin['loop_count'],
                           status='end_not_in_saved_trace', begin=event_ref(begin)))
    for eid, start in closed_since.items():
        closed[eid].append([start, last_time])
    blocks.sort(key=lambda b: b['begin_ns'])
    return waits, blocks, unmatched, closed


def analyze_session(directory):
    service = directory/'service'
    rows = read(directory/'requests.json')
    plan_path = service/'request_plan.json'
    need(rows and [r['index'] for r in rows] == list(range(len(rows))), 'Request indexes must be contiguous')
    need(plan_path.is_file() and [r['spec'] for r in rows] == read(plan_path), 'Requests differ from complete saved plan')
    by_pid, contract, trace_hashes = load_events(service)
    candidates = [pid for pid, evs in by_pid.items()
                  if any(e['event'] == 'engine_response_dispatch' and e.get('finish') for e in evs)]
    need(len(candidates) == 1, f'Expected one scheduler PID, found {candidates}')
    pid = candidates[0]
    events = by_pid[pid]
    waits, blocks, incomplete, closed = timelines(events)
    ready_by_sid, dispatch_by_sid, first_output = defaultdict(list), defaultdict(list), {}
    for e in events:
        if e['event'] == 'cpu_output_ready':
            sid = str(e['session_id'])
            first_output.setdefault(sid, e)
            if e['finish']:
                ready_by_sid[sid].append(e)
        elif e['event'] == 'engine_response_dispatch' and e['finish']:
            dispatch_by_sid[str(e['session_id'])].append(e)
    ordered_sids = sorted(dispatch_by_sid, key=lambda sid: dispatch_by_sid[sid][0]['time_ns'])
    need(len(ordered_sids) == len(rows), 'Unique FINISH session count differs from request count')
    response_ids = [r.get('response', {}).get('id') for r in rows]
    if all(i is not None for i in response_ids):
        expected_sids = [str(i) for i in response_ids]
        need(len(set(expected_sids)) == len(expected_sids), 'Reused response session IDs are ambiguous')
        need(expected_sids == ordered_sids, 'HTTP response IDs/order differ from FINISH trace order')
        identity = 'exact HTTP response.id equals trace session_id, plus chronological request order'
    else:
        need(all(i is None for i in response_ids), 'Mixed missing/present HTTP response IDs')
        expected_sids = ordered_sids
        identity = 'ordered serial requests and unique chronological FINISH sessions; response.id absent'
    request_details = []
    prefill_starts = []
    for row, sid in zip(rows, expected_sids):
        dispatch = dispatch_by_sid[sid][0]
        tokens = dispatch['output_tokens']
        need(all(e['output_tokens'] == tokens for e in dispatch_by_sid[sid]), 'Conflicting duplicate FINISH dispatch lengths')
        ready = [e for e in ready_by_sid[sid]
                 if e['time_ns'] <= dispatch['time_ns'] and e['output_tokens'] == tokens
                 and e['event_id'] == dispatch['event_id']]
        need(ready, f'No matching CPU FINISH ready before dispatch for session {sid}')
        first_ready = ready[0]
        need(all(e['output_tokens'] == tokens for e in ready_by_sid[sid]), 'Conflicting duplicate FINISH ready lengths')
        http_tokens = row['usage']['completion_tokens']
        need(http_tokens == len(row['output_tokens']), 'HTTP usage/token count inconsistent')
        reason = row['response']['choices'][0]['finish_reason']
        delta = tokens-http_tokens
        if row['spec']['tokens']:
            need(delta == 0 and http_tokens == row['spec']['tokens'] and reason == 'length',
                 'Fixed-length trace/HTTP token budget mismatch')
        else:
            need(delta == 0 or (delta == 1 and reason == 'stop'), 'Natural output token mismatch exceeds optional removed stop token')
        # Native AsyncEngine removes a trailing stop token before counting HTTP
        # completion tokens (async_engine.py:914-922); record rather than hide it.
        window = [[first_ready['time_ns'], dispatch['time_ns']]]
        related_waits = [w for w in waits if w['event_id'] == dispatch['event_id']
                         and w['task'] == 'MainLoopSendResponse'
                         and w['end_ns'] > window[0][0] and w['begin_ns'] < window[0][1]]
        awaiting = intersect(window, [[w['begin_ns'], w['end_ns']] for w in related_waits])
        gate_closed = intersect(awaiting, closed[dispatch['event_id']])
        prefills = [b for b in blocks if not b['is_decoding'] and b['begin_ns'] <= first_output[sid]['time_ns']]
        need(prefills, f'No host prefill block before first CPU output for {sid}')
        prefill_start = prefills[-1]['begin_ns']
        prefill_starts.append(prefill_start)
        request_details.append(dict(index=row['index'], spec=row['spec'], session_id=sid, pid=pid,
            identity_basis=identity, returned_tokens=http_tokens, engine_finish_tokens=tokens,
            omitted_stop_token_count=delta, finish_reason=reason, event_id=dispatch['event_id'],
            first_cpu_finish_ready=event_ref(first_ready), first_finish_dispatch=event_ref(dispatch),
            duplicate_finish_ready_events=len(ready_by_sid[sid])-1,
            duplicate_finish_dispatch_attempts=len(dispatch_by_sid[sid])-1,
            finish_ready_to_dispatch_ms=(dispatch['time_ns']-first_ready['time_ns'])/1e6,
            response_event_await_ms_inside_finish_delay=ms(awaiting),
            event_closed_ms_inside_response_await=ms(gate_closed),
            await_resume_or_open_event_ms=ms(awaiting)-ms(gate_closed),
            response_waits=related_waits, closed_gate_overlap_intervals_ns=gate_closed,
            first_cpu_output=event_ref(first_output[sid]), host_prefill_begin_ns=prefill_start))
    need(prefill_starts == sorted(set(prefill_starts)), 'Request prefill attribution is not unique and chronological')
    for i, detail in enumerate(request_details):
        next_start = prefill_starts[i+1] if i+1 < len(prefill_starts) else events[-1]['time_ns']+1
        detail['host_model_blocks_until_next_prefill'] = [b for b in blocks
                                                        if prefill_starts[i] <= b['begin_ns'] < next_start]
    stats = {}
    fields = ('finish_ready_to_dispatch_ms', 'response_event_await_ms_inside_finish_delay',
              'event_closed_ms_inside_response_await', 'await_resume_or_open_event_ms')
    for detail in request_details:
        if detail['spec']['phase'] != 'measured':
            continue
        key = f"{detail['spec']['sample']}_{detail['spec']['tokens']}"
        stats.setdefault(key, []).append(detail)
    summaries = {}
    for key, chosen in stats.items():
        summaries[key] = dict(n=len(chosen), metrics={field:dict(
            median=statistics.median(d[field] for d in chosen),
            min=min(d[field] for d in chosen), max=max(d[field] for d in chosen)) for field in fields})
    metadata = read(directory/'result.json').get('session') if (directory/'result.json').exists() else None
    return dict(directory=str(directory), session=metadata, scheduler_pid=pid, request_count=len(rows),
                identity_basis=identity, requests=request_details, measured_cells=summaries,
                host_model_block_timeline=blocks, incomplete_terminal_events=incomplete,
                other_trace_pids={str(p):len(e) for p,e in by_pid.items() if p != pid},
                trace_source_sha256=contract, trace_files_sha256=trace_hashes,
                requests_sha256=digest(directory/'requests.json'), plan_sha256=digest(plan_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    need(not args.output.exists(), f'Output already exists: {args.output}')
    run = args.run.resolve()
    directories = [run] if (run/'requests.json').is_file() else sorted(
        p for p in run.iterdir() if p.is_dir() and (p/'requests.json').is_file())
    need(directories, 'No session request files')
    report = dict(status='passed', scope='Independent CPU scheduling diagnostic; excluded from primary HTTP speed tables',
        run=str(run), analyzer_sha256=digest(Path(__file__)), sessions=[], failures=[],
        definitions=dict(finish_ready_to_dispatch_ms='First matching CPU InferOutput(finish=True) to first FINISH dispatch attempt',
            response_event_await_ms_inside_finish_delay='Elapsed await intervals of MainLoopSendResponse overlapping that delay; includes wakeup scheduling',
            event_closed_ms_inside_response_await='Intersection of response await with observed CounterEvent unset intervals',
            host_model_blocks='Host async coroutine begin/end only; end is not CUDA completion'),
        limitations=['Trace files are merged per PID and never assumed to equal one request.',
                     'Duplicate FINISH observations are retained and must agree in token length.',
                     'Natural EOS may contribute one engine stop token omitted from HTTP output.',
                     'Unfinished terminal waits/blocks may reflect shutdown or events after the last flush.',
                     'No GPU kernel-time attribution; no instrumented HTTP latency enters speed summaries.'])
    for directory in directories:
        try:
            report['sessions'].append(analyze_session(directory))
        except Exception as error:
            report['status'] = 'failed'
            report['failures'].append(dict(directory=str(directory), type=type(error).__name__, error=str(error)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], sessions=len(report['sessions']), failures=report['failures'],
                         output=str(args.output)), ensure_ascii=False))
    if report['status'] != 'passed':
        sys.exit(1)


if __name__ == '__main__':
    main()
