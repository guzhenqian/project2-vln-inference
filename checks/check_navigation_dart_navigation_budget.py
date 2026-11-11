"""CPU-only prompt budget for the unchanged, frozen 150 navigation episodes.

Loads tokenizer and two saved CPU processor-input tensors, never model weights.
Enumerates every instruction with 2..9 images. A passing result proves token
capacity only; full-image GPU memory and actual inference remain unverified.
"""

import argparse
import ast
import gzip
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
SOURCE = ROOT / 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-prefill', type=int, default=3072)
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_prefill <= 3584:
        parser.error('max-prefill must be in 1..3584 for session4096 and max-new512')
    print('CPU TOKEN BUDGET ONLY: no model weights, GPU, HTTP, generation or navigation', flush=True)
    paths = [ROOT / f'data/baseline/overnight200_fp16_20261031/batch_{i:02d}.json'
             for i in range(1, 4)]
    batches = [read(path) for path in paths]
    episode_ids = [str(eid) for batch in batches for eid in batch['episodes']]
    baseline = read(ROOT / 'reports/benchmark_overnight200_20261031.json')
    if (len(episode_ids) != 150 or len(set(episode_ids)) != 150
            or baseline['completed_episode_count'] != 150
            or set(episode_ids) != {str(e['episode_id']) for e in baseline['episodes']}):
        raise ValueError('The three frozen batches do not match the completed 150 baseline episodes')
    dataset_path = ROOT / batches[0]['dataset_path'].format(split=batches[0]['split'])
    if any(digest(dataset_path) != batch['source_annotation_sha256'] for batch in batches):
        raise ValueError('Frozen navigation annotations changed')
    if any(digest(SOURCE) != batch['author_policy_sha256'] for batch in batches):
        raise ValueError('Pinned author policy changed')
    provenance = read(ROOT / 'logs/overnight200_fp16_20261031/batch_02/provenance.json')
    for name in ('simplevln_policy.py', 'simplevln_env.py'):
        if digest(ROOT / 'checks' / name) != provenance['scripts'][name]:
            raise ValueError('Frozen navigation policy/environment changed: ' + name)
    source_tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    templates = [ast.literal_eval(node.value) for node in ast.walk(source_tree)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Attribute) and t.attr == 'promt_template'
                         for t in node.targets)]
    if len(templates) != 1:
        raise ValueError('Expected the single original author prompt template')
    template = templates[0]
    env_tree = ast.parse((ROOT / 'checks/simplevln_env.py').read_text(encoding='utf-8'))
    dimensions = {}
    for node in ast.walk(env_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = ast.unparse(target)
                if name in ('config.SIMULATOR.RGB_SENSOR.WIDTH', 'config.SIMULATOR.RGB_SENSOR.HEIGHT'):
                    dimensions[name.rsplit('.', 1)[-1]] = ast.literal_eval(node.value)
    if dimensions != {'WIDTH': 640, 'HEIGHT': 480}:
        raise ValueError('Frozen RGB sensor dimensions changed')

    config = read(MODEL / 'config.json')
    vision, text = config['vision_config'], config['text_config']
    processor = read(MODEL / 'preprocessor_config.json')
    merge, patch = vision['spatial_merge_size'], vision['patch_size']
    pixels = dimensions['WIDTH'] * dimensions['HEIGHT']
    if (not processor['size']['shortest_edge'] <= pixels <= processor['size']['longest_edge']
            or any(value % (merge * patch) for value in dimensions.values())):
        raise ValueError('Cannot infer unchanged image grid without processor resizing')
    per_image = pixels // (merge * patch) ** 2
    if per_image != 300:
        raise ValueError('Expected the frozen 640x480 images to produce 300 merged visual tokens')

    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    image_token = tokenizer.convert_ids_to_tokens(config['image_token_id'])

    def encode(messages):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace(image_token, image_token * per_image)
        return tokenizer.encode(prompt, add_special_tokens=False)

    samples = read(ROOT / 'data/calibration/navigation_awq_v1/samples.json')
    processed = read(ROOT / 'data/calibration/navigation_awq_learning_20261104/processed/manifest.json')
    sample_checks = []
    for sample, entry in zip(samples[:2], processed[:2]):
        path = ROOT / entry['path']
        if sample['id'] != entry['id'] or digest(path) != entry['sha256']:
            raise ValueError('Saved processor sample identity/hash mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        encoded = encode(sample['messages'])
        if encoded != saved['input_ids'].flatten().tolist():
            raise ValueError('Tokenizer budget input IDs differ from saved real processor input')
        if not torch.equal(saved['image_grid_thw'], torch.tensor([[1, 30, 40]] * entry['images'])):
            raise ValueError('Saved image grid differs from inferred 640x480 grid')
        sample_checks.append({'id': sample['id'], 'input_tokens': len(encoded),
                              'images': entry['images'], 'exact_input_ids_match': True})

    dataset = json.loads(gzip.decompress(dataset_path.read_bytes()))
    by_id = {str(e['episode_id']): e for e in dataset['episodes']}
    rows = []
    for episode_id in episode_ids:
        instruction = by_id[episode_id]['instruction']['instruction_text']
        suffix = template.format(instruction).split('current observation.')[1]
        for image_count in range(2, 10):
            content = [{'type': 'text', 'text': 'Imagine you are a robot programmed for navigation tasks. '
                        'You have been given a video of historical observations'}]
            content += [{'type': 'image', 'image': 'not_loaded'} for _ in range(image_count - 1)]
            content += [{'type': 'text', 'text': 'and an image of the current observation'},
                        {'type': 'image', 'image': 'not_loaded'}, {'type': 'text', 'text': suffix}]
            messages = [{'role': 'system', 'content': [{'type': 'text',
                         'text': 'You are a visual language navigation model.'}]},
                        {'role': 'user', 'content': content}]
            ids = encode(messages)
            visual = ids.count(config['image_token_id'])
            if visual != image_count * per_image:
                raise ValueError('Unexpected visual placeholder token count')
            rows.append({'episode_id': episode_id, 'images': image_count,
                         'input_tokens': len(ids), 'image_tokens': visual,
                         'other_tokens': len(ids) - visual})
    maximum = max(row['input_tokens'] for row in rows)
    required = maximum + 512
    if maximum > args.max_prefill or required > 4096:
        raise ValueError(f'Input budget does not fit: max={maximum}, with output512={required}')
    by_images = [{'images': n, 'min_input_tokens': min(r['input_tokens'] for r in rows if r['images'] == n),
                  'max_input_tokens': max(r['input_tokens'] for r in rows if r['images'] == n)}
                 for n in range(2, 10)]
    block_bytes = 64 * text['num_hidden_layers'] * text['num_key_value_heads'] * text['head_dim'] * 2 * 2
    report = {
        'status': 'input_budget_verified_only', 'episode_count': 150, 'evaluated_shapes': len(rows),
        'max_prefill': args.max_prefill, 'session_len': 4096, 'max_new_tokens': 512,
        'maximum_input_tokens': maximum, 'maximum_input_plus_output': required,
        'maximum_cases': [row for row in rows if row['input_tokens'] == maximum],
        'rgb_sensor': dimensions, 'image_grid_thw': [1, 30, 40], 'tokens_per_image': per_image,
        'sample_checks': sample_checks, 'by_image_count': by_images,
        'cache_arithmetic': {'block_tokens': 64, 'block_bytes': block_bytes,
                             'full_sequence_blocks_required': math.ceil(required / 64),
                             'proposed_gpu_blocks': 64, 'reserved_gpu_blocks': 1,
                             'usable_token_slots': 63 * 64, 'fixed_cache_mib': 64 * block_bytes / 2 ** 20,
                             'lmdeploy_runtime_reserve_mib': (args.max_prefill + 2) * text['vocab_size'] * 2 / 2 ** 20},
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [*paths, dataset_path, SOURCE, ROOT / 'checks/simplevln_policy.py',
                           ROOT / 'checks/simplevln_env.py', MODEL / 'config.json',
                           MODEL / 'chat_template.jinja', MODEL / 'tokenizer.json',
                           MODEL / 'tokenizer_config.json', MODEL / 'preprocessor_config.json']},
        'limitations': ['No model weights, GPU calls, real generation, HTTP or simulator execution',
                        'Token capacity is not a GPU-memory feasibility guarantee',
                        'LMDeploy automatic cache sizing shrinks when max_prefill is raised',
                        'A fixed64-block cache must still pass actual nine-image inference',
                        'The current diagnostic selector/logging may add runtime and memory overhead'],
        'rows': rows,
    }
    print('Frozen episodes: 150; image counts: 2..9; enumerated inputs:', len(rows))
    print('Nine-image range:', by_images[-1]['min_input_tokens'], '..', by_images[-1]['max_input_tokens'])
    print(f'Maximum: {maximum} input + 512 output = {required} <= 4096')
    print('FP16 KV cache:', report['cache_arithmetic'])
    if not args.no_write:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
        output = ROOT / 'reports' / ('navigation_dart_navigation_budget_' + run_id + '.json')
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print('Saved:', output)
    print('DART_NAVIGATION_TOKEN_BUDGET_OK')
    print('Token counts only; GPU full-prefill and navigation validation remain pending.')


if __name__ == '__main__':
    main()
