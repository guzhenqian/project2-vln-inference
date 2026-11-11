"""Freeze the authorized nested 50/100 roster without consulting any outcomes."""
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'data/baseline/navigation_stage2_20261109'
RUN = ROOT / 'logs/navigation_stage2_sequence_20261109'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    assert not PLAN.exists() and not RUN.exists(), 'Preserve existing phase2 work'
    old = ROOT / 'data/baseline/overnight200_fp16_20261031'
    source = read(old / 'manifest.json')
    batches = [read(old / ('batch_%02d.json' % i)) for i in (1, 2, 3)]
    details = {str(row['episode_id']): row for row in source['sample_details']}
    pool = [str(eid) for batch in batches for eid in batch['episodes']]
    assert len(set(pool)) == 150
    assert len({details[eid]['trajectory_id'] for eid in pool}) == 150
    PLAN.mkdir(parents=True)
    RUN.mkdir(parents=True)
    metadata = []
    for i, batch in enumerate(batches, 1):
        ids = list(map(str, batch['episodes']))
        counts = Counter(details[eid]['scene'] for eid in ids)
        assert len(ids) == 50 and len(counts) == 11
        metadata.append(dict(batch=i, episode_ids=ids, scene_counts=dict(counts),
            length_bin_counts=dict(Counter(str(details[eid]['length_bin']) for eid in ids)),
            source=f'data/baseline/overnight200_fp16_20261031/batch_{i:02d}.json',
            source_sha256=digest(old / f'batch_{i:02d}.json')))
        write(PLAN / f'source_batch_{i:02d}.json', batch)
    contract = copy.deepcopy(read(ROOT / 'setup/navigation-stage2-common-20261109.json'))
    contract.update(version='navigation_stage2_execution_20261109',
        status='authorized_validation_in_progress', execution_enabled=True,
        scope='Nine single-method configurations; seven new runs plus two reused DART results')
    contract['measurement'].update(
        audit_mode='Preserve phase1 detailed runtime instrumentation and original HTTP statistics; no lightweight-mode migration',
        performance_mode='Original measurement protocol; report method-specific instrumentation and algorithm costs',
        mode_switch_gate='No mode switch authorized for this screening round',
        timing_status='Reuse original DART per-case HTTP timings; all new runs use the frozen evaluator and original sampling boundaries')
    contract['data_policy'].update(screening_roster='Original batch_01 (50)',
        confirmation_roster='Development re-screen: original batch_01+batch_02 (100 cumulative); not independent final confirmation',
        existing150_role='Shared development pool; 50/100 nested subsets reused exactly across algorithms')
    contract['remaining_gates'] = [item for item in contract['remaining_gates'] if 'lightweight' not in item]
    contract['screening'] = dict(initial_count=50, cumulative_rescreen_count=100,
        reserve_count=150, nested=True, reuse_completed_cases=True,
        new_groups=['fp16', 'awq', 'gptq', 'fastv75', 'fastv50', 'sparsevlm75', 'sparsevlm50'],
        reuse_groups=['dart75', 'dart50'],
        validation_order='Checkpoint/format, native CUDA, two/nine-image requests, full single1250, then formal screening',
        launch_gate='All planned methods validated or a specific integration blocker explicitly reported; never silently substitute an algorithm',
        selection_policy='After all50: preserve stability failures, compare paired SR/SPL and P95/memory; small differences do not eliminate candidates. Re-screen viable or uncertain candidates to cumulative100.',
        execution='Single GPU sequential; no concurrent calibration and inference',
        overnight_priority='Finish initial50 for all technically accepted groups; no promised completion deadline')
    contract['rosters'] = metadata
    write(ROOT / 'setup/navigation-stage2-execution-20261109.json', contract)
    write(PLAN / 'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        pool150=pool, initial50=pool[:50], cumulative100=pool[:100], batches=metadata,
        selection='Reuse original outcome-independent stratified batches exactly; no new outcome-based sampling',
        execution_contract='setup/navigation-stage2-execution-20261109.json',
        execution_contract_sha256=digest(ROOT / 'setup/navigation-stage2-execution-20261109.json')))
    write(RUN / 'status.json', dict(status='adapting', started_utc=datetime.now(timezone.utc).isoformat(),
        formal_started=False, gptq_quantization='running',
        roster='data/baseline/navigation_stage2_20261109/manifest.json',
        note='All validations precede screening; DART75/50 reuse authorized'))
    print('STAGE2_ROSTER_FROZEN initial50=50 cumulative100=100 scenes_per_batch=11')


if __name__ == '__main__':
    main()
