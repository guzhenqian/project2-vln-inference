"""Freeze an outcome-blind 20/80/100 navigation repair roster (CPU only).

The only case-level inputs are original membership and official annotations.
No model result, success label, measured episode length, or timing is read.
Run once to create the directory; --verify independently recomputes selection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/baseline/navigation_stage2_20261109'
OUTPUT = ROOT / 'data/baseline/navigation_repair_20261112'
PROTOCOL = ROOT / 'reports/navigation_repair_protocol_20261112.md'
SEED = 20261112
CANDIDATES = 100000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def membership_sha(ids):
    return hashlib.sha256(('\n'.join(ids) + '\n').encode('ascii')).hexdigest()


def prepare():
    original = read(SOURCE / 'manifest.json')
    batches = [read(SOURCE / f'source_batch_{i:02d}.json') for i in (1, 2)]
    ids = list(map(str, original['cumulative100']))
    assert ids == list(map(str, original['pool150'][:100]))
    assert ids == sum((list(map(str, b['episodes'])) for b in batches), [])
    assert len(ids) == len(set(ids)) == 100
    template = batches[0]
    dataset = ROOT / template['dataset_path'].format(split=template['split'])
    assert sha(dataset) == template['source_annotation_sha256']
    assert batches[1]['source_annotation_sha256'] == template['source_annotation_sha256']
    contract = ROOT / original['execution_contract']
    assert sha(contract) == original['execution_contract_sha256']
    with gzip.open(dataset, 'rt', encoding='utf-8') as stream:
        annotations = {str(e['episode_id']): e for e in json.load(stream)['episodes']}
    trajectory_ids = dict(zip(ids, sum((list(map(str, b['trajectory_ids'])) for b in batches), [])))
    rows = []
    for index, eid in enumerate(ids):
        episode = annotations[eid]
        assert str(episode['trajectory_id']) == trajectory_ids[eid]
        rows.append(dict(episode_id=eid, trajectory_id=trajectory_ids[eid], original100_index=index,
            scene_id=episode['scene_id'], geodesic_distance_m=float(episode['info']['geodesic_distance']),
            instruction_word_count=len(episode['instruction']['instruction_text'].split())))
    for field, label in [('geodesic_distance_m', 'distance'), ('instruction_word_count', 'instruction')]:
        for rank, row in enumerate(sorted(rows, key=lambda r: (r[field], r['original100_index']))):
            row[label + '_rank_in_original100'] = rank
            row[label + '_quartile'] = rank // 25
    bins = [[r for r in rows if r['distance_quartile'] == q] for q in range(4)]
    scene_sizes = Counter(r['scene_id'] for r in rows)
    # Hamilton-style target with a mandatory minimum of one per scene.
    targets = {scene: max(1, count // 5) for scene, count in scene_sizes.items()}
    while sum(targets.values()) < 20:
        scene = min(scene_sizes, key=lambda s: (-(scene_sizes[s] / 5 - targets[s]), s))
        targets[scene] += 1
    rng = random.Random(SEED)
    best = None
    for _ in range(CANDIDATES):
        selected = sum((rng.sample(bucket, 5) for bucket in bins), [])
        scenes = Counter(r['scene_id'] for r in selected)
        instructions = Counter(r['instruction_quartile'] for r in selected)
        rank_spread = sum((rank - (5 * i + 2)) ** 2 for i, rank in
            enumerate(sorted(r['distance_rank_in_original100'] for r in selected)))
        score = (len(set(scene_sizes) - set(scenes)),
            sum(abs(instructions[q] - 5) for q in range(4)),
            sum(abs(scenes[s] - targets[s]) for s in scene_sizes), rank_spread,
            tuple(sorted(r['original100_index'] for r in selected)))
        if best is None or score < best[0]:
            best = (score, selected)
    score, selected = best
    assert score[0] == score[1] == 0, 'Selection failed scene/instruction coverage gates'
    selected = sorted(selected, key=lambda r: r['original100_index'])
    selected_ids = [r['episode_id'] for r in selected]
    remaining_ids = [eid for eid in ids if eid not in set(selected_ids)]
    assert len(selected_ids) == 20 and len(remaining_ids) == 80
    assert set(selected_ids).isdisjoint(remaining_ids) and set(selected_ids + remaining_ids) == set(ids)
    inputs = [SOURCE / 'manifest.json', SOURCE / 'source_batch_01.json', SOURCE / 'source_batch_02.json',
              dataset, contract, Path(__file__), PROTOCOL]
    inputs_sha = {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs}
    algorithm = dict(seed=SEED, candidate_count=CANDIDATES,
        membership_source='original cumulative100, unchanged; no result files read',
        ranking='Official geodesic distance and instruction whitespace word count, ties by original100 index; four rank quartiles of 25.',
        candidates='Python random.Random(seed); sample 5 distinct cases from each distance quartile per candidate.',
        score_lexicographic=['number of original scenes missing', 'L1 instruction-quartile count deviation from [5,5,5,5]',
            'L1 scene-count deviation from min1 Hamilton-style proportional target',
            'sum squared distance-rank deviation from 2,7,...,97', 'sorted original100 indices'],
        scene_quota_note='Scene target is a soft score; all 11 scenes and exact 5/5/5/5 distance and instruction quartiles are hard assertions.',
        best_score=list(score[:4]), scene_targets=dict(sorted(targets.items())),
        output_order='Original100 relative order within each list; cumulative execution may be selected20 then remaining80.')
    artifacts = {}
    for filename, members, label in [('subset20.json', selected_ids, 'engineering acceptance20'),
                                     ('remaining80.json', remaining_ids, 'remaining80 after acceptance'),
                                     ('original100.json', ids, 'original same100 formal membership')]:
        manifest = dict(template)
        for key in ('batch', 'source_precision'):
            manifest.pop(key, None)
        manifest.update(version='navigation_repair_20261112', scope=label,
            episodes=members, trajectory_ids=[trajectory_ids[eid] for eid in members],
            model='w4a16-fix-fp16', dtype='FP16',
            backend='LMDeploy0.11.1 native decode CUDA Graph; full-prefill3072; FP16 KV64x64; TP/DP/batch1',
            graph='on', selection='Outcome-blind repair subset; see adjacent manifest.json for frozen algorithm.',
            sampling_seed=SEED, pre_registered='2026-11-12 before new repair inference results',
            repair_manifest_role='FP16 base manifest; controller changes only model identity/precision metadata per configuration, then freezes run copy.',
            repair_roster_sha256=membership_sha(members),
            repair_protocol='reports/navigation_repair_protocol_20261112.md',
            repair_protocol_sha256=inputs_sha['reports/navigation_repair_protocol_20261112.md'],
            differences_from_author_benchmark='Original frozen development subset and policy; repaired inference execution and explicitly documented precision recipe.')
        assert manifest['temperature'] == .3 and manifest['top_p'] == .95
        assert manifest['max_tokens'] == 512 and manifest['seed'] == 7 and manifest['max_episode_steps'] == 400
        artifacts[filename] = encode(manifest)
    metadata = dict(version='navigation_repair_20261112', pre_registered_date='2026-11-12',
        outcome_data_used=False, selection_algorithm=algorithm, sources_sha256=inputs_sha,
        selected20=selected_ids, remaining80=remaining_ids, original100=ids,
        roster_sha256={key: membership_sha(value) for key, value in
                      [('selected20', selected_ids), ('remaining80', remaining_ids), ('original100', ids)]},
        coverage=dict(original_scene_count=len(scene_sizes), selected_scene_count=len({r['scene_id'] for r in selected}),
            scene_counts=dict(sorted(Counter(r['scene_id'] for r in selected).items())),
            distance_quartile_counts=[sum(r['distance_quartile'] == q for r in selected) for q in range(4)],
            instruction_quartile_counts=[sum(r['instruction_quartile'] == q for r in selected) for q in range(4)],
            selected_geodesic_range_m=[min(r['geodesic_distance_m'] for r in selected), max(r['geodesic_distance_m'] for r in selected)],
            selected_instruction_word_range=[min(r['instruction_word_count'] for r in selected), max(r['instruction_word_count'] for r in selected)]),
        case_metadata_original100=rows,
        artifact_files_sha256={name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
        caveat='20 is an engineering screen; it does not establish statistical equivalence, algorithm ranking, or generalization.')
    return metadata, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Read-only: recompute selection and validate every frozen byte/hash.')
    args = parser.parse_args()
    metadata, artifacts = prepare()
    if args.verify:
        stored = read(OUTPUT / 'manifest.json')
        created = stored.pop('created_utc')
        assert created and stored == metadata, 'Frozen selection/metadata changed'
        for name, data in artifacts.items():
            assert (OUTPUT / name).read_bytes() == data, 'Frozen manifest changed: ' + name
    else:
        OUTPUT.mkdir(exist_ok=False)
        metadata['created_utc'] = datetime.now(timezone.utc).isoformat()
        for name, data in {**artifacts, 'manifest.json': encode(metadata)}.items():
            with (OUTPUT / name).open('xb') as stream:
                stream.write(data)
    print(json.dumps(dict(status='verified' if args.verify else 'created', directory=OUTPUT.relative_to(ROOT).as_posix(),
        selected20=metadata['selected20'], roster_sha256=metadata['roster_sha256'], coverage=metadata['coverage']), ensure_ascii=False))


if __name__ == '__main__':
    main()
