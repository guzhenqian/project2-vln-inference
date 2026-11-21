"""Freeze every model request from the first six original100 FP16 episodes.

--prepare-only is an offline, image-decoding operation; run it only in the
authorized CPU window. --check-only reads metadata/source text only. Importing
this module or its manifest/payload loaders never imports PIL, imageio or torch,
and loaders never read source PNGs. Outputs do not feed back into navigation:
this is a fixed historical-input latency replay, never a navigation SR trial.
"""
import argparse
import base64
from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'navigation_latency_replay_20261120/v1'
EPISODES = ('1250', '159', '1413', '558', '1123', '384')
EXPECTED_CALLS = dict(zip(EPISODES, (128, 14, 16, 156, 17, 12)))
EXPECTED_PRIMITIVE_STEPS = dict(zip(EPISODES, (400, 53, 52, 400, 59, 46)))
REQUEST_COUNT = 343
REPORT = 'reports/navigation_repair_20261114_continuation_fp16_100.json'
REPORT_SHA256 = 'df0e038ddb68df3f86c2a1c14b8db829bd85b1093b7db521d3b7961c947a9547'
MANIFEST = 'data/baseline/navigation_repair_20261112/manifest.json'
ORIGINAL100 = 'data/baseline/navigation_repair_20261112/original100.json'
POLICY = 'checks/simplevln_policy.py'
AUTHOR = 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'
DATASET = 'data/datasets/R2R_VLNCE_v1-3_overnight200/val_unseen/val_unseen.json.gz'
SCREEN = 'logs/navigation_repair_20261112/screen_fp16_v1/fp16/navigation'
REMAINING = 'logs/navigation_repair_20261112/continuation_20261114_v1/remaining80_fp16/fp16/navigation'
PINNED_SOURCE_SHA256 = {
    MANIFEST: '10bd43d8256c2a400c7729e9f6452e77fa2506fb0c0ef7fc29fad08993da1cea',
    ORIGINAL100: 'a9750fb607b8ff5c1397653d921467b610a9e03ef4abc6d94dc4727d3d263b8c',
    POLICY: 'e73eb87dfa0946a42fc9b8efb4e1911ac41e382478a0ace24e5e5e19cd8132bc',
    AUTHOR: '72efa80174a832cec519eb7bb73ba2f635848b47af4dcaafaa03078e0d8d3a1f',
    DATASET: '152c04e9eb0b5ecee136797441838293eefe74208d4d8bd0a8313c134e0106ad',
    SCREEN+'/manifest.json': '415604fee5582845ade6d6be661f66e59dc2a14c0a8200e3d256553c83578505',
    REMAINING+'/manifest.json': 'a74348176ece81412b944549e76554f45fdbf2456c3bff9d6c779607144f1d0b',
    REMAINING+'/episode_1250/steps.jsonl': 'a332d7fc344aef13de2e3c8bdbec4d44bb19b3f3a6e202ab1153593de1ed98a2',
    REMAINING+'/episode_159/steps.jsonl': '012cb6f565f7a63a8e5271f804f3240b6b2434cdb8fa87138205cae33c20fffd',
    REMAINING+'/episode_1413/steps.jsonl': '0ee43a2fd5e5ed3986210d58d352be12cd98e0afcb5935daf94a9c90810ee7ef',
    SCREEN+'/episode_558/steps.jsonl': 'b786370c6f4a89c4c640b505f8dd47516c3b1b174e354ba64b2a218c6074677e',
    REMAINING+'/episode_1123/steps.jsonl': '9e7d98bd7fc152fe5b4d687137cb2cb7515c6dc0e75fb1a60ce3987a24fe83f5',
    REMAINING+'/episode_384/steps.jsonl': 'dd41b3e4969a3714566656214b3d62e493e671345a7576214e9992d7916d4fa7',
}
GENERATION = dict(temperature=0, top_p=1, top_k=1, seed=7, max_tokens=18,
                  min_new_tokens=18, ignore_eos=True, stream=False, return_token_ids=True)
CACHE_CONTRACT = dict(vision_cache=False, prefix_kv_cache=False,
                      future_cache_reset='every_episode',
                      preparation_decode_cache='At most 199 prior RGB frames plus the current frame; offline only')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value):
    """Stable wire encoding shared by the offline preparation and runner."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def semantic_payload_sha256(payload):
    """Hash all request semantics except the service model identity."""
    return digest_bytes(canonical_json_bytes({k:v for k,v in payload.items() if k != 'model'}))


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Expected relative path')
    root = Path(root).resolve()
    path = (root/relative).resolve()
    require(path == root or root in path.parents, f'Path leaves its declared root: {relative}')
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_steps_path(episode):
    base = SCREEN if episode == '558' else REMAINING
    return f'{base}/episode_{episode}/steps.jsonl'


def validate_source_files(expected):
    actual = {name:digest_file(safe_path(ROOT, name)) for name in expected}
    require(actual == expected, 'Frozen replay source hash mismatch: '+str([
        name for name in expected if actual[name] != expected[name]]))
    return actual


def inspect_sources():
    """Metadata-only inspection. Never opens PNGs or imports image libraries."""
    require(digest_file(ROOT/REPORT) == REPORT_SHA256, 'Accepted FP16 report changed')
    report = read_json(ROOT/REPORT)
    require(report.get('status') == 'complete' and report.get('group') == 'fp16'
            and report.get('case_count') == 100, 'Expected accepted complete FP16 original100 report')
    evidence = report['evidence_files_sha256']
    require(all(evidence.get(name) == value for name,value in PINNED_SOURCE_SHA256.items()),
            'Source pins differ from accepted report evidence_files_sha256')
    source_hashes = dict(PINNED_SOURCE_SHA256, **{REPORT:REPORT_SHA256})
    validate_source_files(source_hashes)
    manifest, original = read_json(ROOT/MANIFEST), read_json(ROOT/ORIGINAL100)
    require(tuple(manifest['original100'][:6]) == EPISODES and tuple(original['episodes'][:6]) == EPISODES,
            'Selection is not exactly the first six original100 episodes')
    require(manifest['artifact_files_sha256']['original100.json'] == PINNED_SOURCE_SHA256[ORIGINAL100],
            'Original100 artifact is not linked by the frozen manifest')
    accepted = {str(row['episode_id']):row for row in report['episodes']}
    episodes = []
    for eid in EPISODES:
        rel = source_steps_path(eid)
        rows = [json.loads(line) for line in (ROOT/rel).read_text(encoding='utf-8').splitlines() if line.strip()]
        require(len(rows) == EXPECTED_PRIMITIVE_STEPS[eid] and [row['step'] for row in rows] == list(range(len(rows))),
                f'Incomplete or reordered primitive steps: {eid}')
        nav_root = str(Path(rel).parent.parent).replace('\\','/')
        source_manifest = read_json(ROOT/nav_root/'manifest.json')
        require(eid in source_manifest['episodes'] and source_manifest['split'] == 'val_unseen'
                and source_manifest['dataset_path'].format(split='val_unseen') == DATASET,
                f'Navigation source manifest mismatch: {eid}')
        calls = [row for row in rows if row.get('model_called') is True]
        require(len(calls) == EXPECTED_CALLS[eid] == accepted[eid]['actual_model_calls'],
                f'Model-call count differs from accepted full episode: {eid}')
        for row in rows:
            require(str(row['episode_id']) == eid and type(row['model_called']) is bool, 'Step identity/type mismatch')
            require(row['frame'] == f'episode_{eid}/frame_{row["step"]:04d}.png', 'Unexpected primitive-frame path')
            # Path validation only: no image stat, read, hash, or decode here.
            safe_path(ROOT, nav_root+'/'+row['frame'])
        for row in calls:
            hashes, texts = row['request_image_sha256'], row['request_text']
            require(row.get('error') is None and len(hashes) == row['image_count'] and 2 <= len(hashes) <= 9,
                    f'Invalid original model call: {eid}/{row["step"]}')
            require(all(isinstance(h, str) and len(h) == 64 for h in hashes) and
                    isinstance(texts, list) and all(isinstance(t, str) for t in texts), 'Missing original request evidence')
            require(type(row['usage']['prompt_tokens']) is int and row['usage']['prompt_tokens'] > 0,
                    'Missing source prompt-token count')
        episodes.append(dict(episode_id=eid, source_steps_path=rel, source_navigation_root=nav_root, rows=rows))
    require(sum(EXPECTED_CALLS.values()) == REQUEST_COUNT, 'Fixed call total changed')
    return dict(episodes=episodes, source_files_sha256=source_hashes)


def _write_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)


def _validate_payload(payload, entry):
    require(payload.get('model') == 'MODEL', 'Stored model identity must be MODEL')
    require(set(payload) == {'model', 'messages', *GENERATION}, 'Unexpected replay payload fields')
    require(all(payload.get(k) == v for k,v in GENERATION.items()), 'Fixed generation controls changed')
    require(isinstance(payload['messages'], list) and len(payload['messages']) == 2, 'Expected system and user messages')
    content = payload['messages'][1]['content']
    texts = [item['text'] for item in content if item['type'] == 'text']
    hashes = [digest_bytes(base64.b64decode(item['image_url']['url'].split(',',1)[1], validate=True))
              for item in content if item['type'] == 'image_url']
    require(texts == entry['request_text'] and hashes == entry['request_image_sha256'], 'Frozen request content differs')
    require(len(hashes) == len(entry['source_frame_paths']), 'Image/frame provenance count differs')


def load_manifest(replay_dir):
    """Verify complete frozen metadata; does not open source steps or PNGs."""
    directory = Path(replay_dir).resolve()
    raw = (directory/'manifest.json').read_bytes()
    expected = (directory/'manifest.sha256').read_text(encoding='ascii').strip()
    require(len(expected) == 64 and digest_bytes(raw) == expected, 'Replay manifest checksum mismatch')
    manifest = json.loads(raw)
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'complete', 'Replay preparation incomplete')
    require(manifest['preparation_tool_sha256'] == digest_file(__file__), 'Replay loader/preparation tool changed')
    require(manifest['generation'] == GENERATION and manifest['cache_contract'] == CACHE_CONTRACT,
            'Replay generation/cache contract changed')
    require(manifest['episode_order'] == list(EPISODES) and manifest['request_count'] == REQUEST_COUNT,
            'Replay is not the fixed six-episode complete cohort')
    require(manifest['source_report_sha256'] == REPORT_SHA256 and
            manifest['source_files_sha256'] == dict(PINNED_SOURCE_SHA256, **{REPORT:REPORT_SHA256}),
            'Replay source evidence contract changed')
    requests = manifest['requests']
    require(len(requests) == REQUEST_COUNT and [r['index'] for r in requests] == list(range(REQUEST_COUNT)),
            'Incomplete or reordered replay request index')
    cursor = 0
    for episode_index,eid in enumerate(EPISODES):
        count = EXPECTED_CALLS[eid]
        selected = requests[cursor:cursor+count]
        require(all(r['episode_id'] == eid and r['episode_index'] == episode_index and
                    r['episode_call_index'] == i and r['is_episode_start'] == (i == 0) and
                    r['is_episode_end'] == (i == count-1) for i,r in enumerate(selected)), 'Episode boundary/order changed')
        require([r['step'] for r in selected] == sorted(set(r['step'] for r in selected)), 'Repeated/reordered model call')
        for row in selected:
            require(row['payload_file'] == f'payloads/{row["index"]:06d}.json', 'Unexpected payload path')
            require(row['source_steps_path'] == source_steps_path(eid), 'Request provenance changed')
            require(row['source_prompt_tokens'] == row['prompt_tokens_source'] and row['source_prompt_tokens'] > 0,
                    'Prompt-token evidence mismatch')
            require(row['normalized_payload_bytes_sha256'] == row['payload_bytes_sha256'] and
                    row['semantic_payload_sha256'] == row['semantic_sha256'], 'Payload hash aliases disagree')
        cursor += count
    return manifest


def load_request(replay_dir, entry, model):
    """Return (wire_bytes, proof) before service timing; never reads source PNGs."""
    require(isinstance(model, str) and model, 'Model identity must be a nonempty string')
    require(entry['payload_file'] == f'payloads/{entry["index"]:06d}.json', 'Unexpected payload filename')
    raw = safe_path(replay_dir, entry['payload_file']).read_bytes()
    normalized_sha = digest_bytes(raw)
    require(len(raw) == entry['payload_bytes_length'] and
            normalized_sha == entry['normalized_payload_bytes_sha256'] == entry['payload_bytes_sha256'],
            'Frozen payload bytes changed')
    payload = json.loads(raw)
    require(raw == canonical_json_bytes(payload), 'Stored payload is not canonical')
    _validate_payload(payload, entry)
    semantic_sha = semantic_payload_sha256(payload)
    require(semantic_sha == entry['semantic_payload_sha256'] == entry['semantic_sha256'], 'Payload semantics changed')
    outgoing = dict(payload, model=model)
    require(semantic_payload_sha256(outgoing) == semantic_sha, 'Model substitution changed request semantics')
    wire = canonical_json_bytes(outgoing)
    return wire, dict(normalized_payload_bytes_sha256=normalized_sha,
                     semantic_payload_sha256=semantic_sha, wire_bytes_sha256=digest_bytes(wire),
                     model_only_substitution=True, model=model, wire_bytes_length=len(wire))


def prepare(output):
    require(not output.exists(), f'Replay output must be a fresh directory: {output}')
    inspected = inspect_sources()
    source_hashes = inspected['source_files_sha256']
    tool_sha = digest_file(__file__)
    # Image libraries and the author policy are loaded only inside preparation.
    import imageio
    try:
        import imageio.v2 as imageio_v2
    except ImportError:
        # The original Habitat evaluation uses Python3.6/imageio2.9/Pillow8.3.
        imageio_v2 = imageio
    import PIL
    from PIL import Image
    spec = importlib.util.spec_from_file_location('navigation_replay_original_policy', ROOT/POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    dataset = json.loads(gzip.decompress((ROOT/DATASET).read_bytes()))
    instructions = {str(row['episode_id']):row['instruction']['instruction_text'] for row in dataset['episodes']}
    require(all(eid in instructions for eid in EPISODES), 'Missing selected episode instruction')
    output.mkdir(parents=True, exist_ok=False)
    (output/'payloads').mkdir()
    _write_new(output/'preparation_started.json', canonical_json_bytes(dict(status='preparing', schema=SCHEMA)))
    frame_hashes, requests, summaries = {}, [], []
    try:
        for episode_index,episode in enumerate(inspected['episodes']):
            eid = episode['episode_id']
            history = deque()
            calls = 0
            try:
                for row in episode['rows']:
                    relative = episode['source_navigation_root']+'/'+row['frame']
                    frame_path = safe_path(ROOT, relative)
                    frame_bytes = frame_path.read_bytes()
                    frame_hashes[relative] = digest_bytes(frame_bytes)
                    rgb = imageio_v2.imread(BytesIO(frame_bytes), format='PNG')
                    current_image = None
                    if row['model_called']:
                        agent = policy.AuthorAgent('http://unused', 'MODEL', str(output/'author_agent'),
                                                   25, 15, 200, 1.0, False)
                        agent.rgb_list = [item['image'] for item in history]
                        agent.predict_inference = lambda:'stop'
                        try:
                            agent.act({'rgb':rgb, 'instruction':{'text':instructions[eid]}}, {}, eid)
                            current_image = agent.rgb_list[-1]
                            prior_paths = [item['path'] for item in history]
                            selected_paths = (agent.uniform_sample_with_ends(prior_paths,8) if prior_paths else [relative])+[relative]
                            payload = dict(model='MODEL', messages=agent.conversations, **GENERATION)
                            raw = canonical_json_bytes(payload)
                            hashes = row['request_image_sha256']
                            entry = dict(index=len(requests), episode_id=eid, episode_index=episode_index,
                                episode_call_index=calls, step=row['step'], is_episode_start=calls == 0,
                                is_episode_end=calls == EXPECTED_CALLS[eid]-1,
                                payload_file=f'payloads/{len(requests):06d}.json', payload_bytes_length=len(raw),
                                payload_bytes_sha256=digest_bytes(raw), normalized_payload_bytes_sha256=digest_bytes(raw),
                                semantic_sha256=semantic_payload_sha256(payload), semantic_payload_sha256=semantic_payload_sha256(payload),
                                source_steps_path=episode['source_steps_path'], source_row_sha256=digest_bytes(canonical_json_bytes(row)),
                                source_frame_paths=selected_paths, history_source_frame_paths=prior_paths+[relative],
                                image_count=len(hashes), request_image_sha256=hashes, request_text=row['request_text'],
                                source_prompt_tokens=row['usage']['prompt_tokens'], prompt_tokens_source=row['usage']['prompt_tokens'])
                            _validate_payload(payload, entry)  # Exact original JPEG hashes and complete text, in order.
                            _write_new(output/entry['payload_file'], raw)
                            requests.append(entry)
                            calls += 1
                        finally:
                            # Do not let the fresh agent retain older PIL frames after deque eviction.
                            agent.rgb_list.clear()
                            del agent
                    else:
                        current_image = Image.fromarray(rgb.astype('uint8')).convert('RGB')
                    if len(history) == 199:
                        history.popleft()['image'].close()
                    history.append(dict(path=relative, image=current_image))
                    del rgb, frame_bytes
            finally:
                while history:
                    history.popleft()['image'].close()
            require(calls == EXPECTED_CALLS[eid], f'Prepared episode model-call count changed: {eid}')
            summaries.append(dict(episode_id=eid, primitive_steps=len(episode['rows']), model_calls=calls,
                                  first_request_index=len(requests)-calls, last_request_index=len(requests)-1))
            print(json.dumps(dict(prepared_episode=eid, model_calls=calls, total_requests=len(requests))), flush=True)
        require(len(requests) == REQUEST_COUNT, 'Replay is incomplete')
        validate_source_files(source_hashes)
        require(digest_file(__file__) == tool_sha, 'Preparation source changed while running')
        # All PNG reading/hashing remains inside offline preparation, never loader/timed replay.
        require(all(digest_file(safe_path(ROOT,path)) == expected for path,expected in frame_hashes.items()),
                'Source PNG changed after its offline decode')
        manifest = dict(schema=SCHEMA, status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
            scope='All 343 original FP16 model calls from the first six original100 episodes, each exactly once in original episode/step order; fixed 18-token latency replay, not navigation SR',
            selection='First six frozen original100 IDs; no latency or outcome-based request selection or omission',
            preparation_tool_sha256=tool_sha, source_report_path=REPORT, source_report_sha256=REPORT_SHA256,
            source_manifest_path=MANIFEST, source_manifest_sha256=PINNED_SOURCE_SHA256[MANIFEST],
            source_files_sha256=source_hashes, source_png_sha256=frame_hashes,
            source_hashes_rechecked_after_preparation=True, original_jpeg_and_request_text_equal=True,
            canonical_payload_encoding='UTF-8; ensure_ascii=False; sort_keys=True; compact separators; allow_nan=False',
            canonical_model='MODEL', generation=GENERATION, cache_contract=CACHE_CONTRACT,
            outputs_fed_back=False, navigation_success_metric=False, formal_request_warmup=False,
            episode_order=list(EPISODES), episode_counts=summaries, request_count=len(requests), requests=requests,
            preparation_libraries=dict(pillow=PIL.__version__, imageio=imageio.__version__, python=sys.version))
        final_bytes = canonical_json_bytes(manifest)
        _write_new(output/'manifest.pending.json', final_bytes)
        _write_new(output/'manifest.sha256', (digest_bytes(final_bytes)+'\n').encode('ascii'))
        # Publication of manifest.json is the final action making the directory runnable.
        os.replace(output/'manifest.pending.json', output/'manifest.json')
    except BaseException as error:
        _write_new(output/'preparation_failure.json', canonical_json_bytes(dict(status='failed',
                   error_type=type(error).__name__, error=str(error), completed_requests=len(requests))))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.check_only:
        require(args.output is None, '--check-only writes no replay output; omit --output')
        inspected = inspect_sources()
        print(json.dumps(dict(status='metadata_checked', episode_order=EPISODES,
            request_count=REQUEST_COUNT, model_calls=EXPECTED_CALLS, primitive_steps=EXPECTED_PRIMITIVE_STEPS,
            checked_sources=len(inspected['source_files_sha256']), png_reads=0, image_decodes=0)))
        return
    require(args.output is not None, '--prepare-only requires --output <fresh replay directory>')
    prepare(args.output.resolve())
    print(json.dumps(dict(status='complete', output=str(args.output.resolve()), request_count=REQUEST_COUNT)))


if __name__ == '__main__':
    main()
