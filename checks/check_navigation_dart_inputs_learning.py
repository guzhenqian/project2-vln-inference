"""Inspect two saved CPU inputs and a candidate visual-token budget; no pruning."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch

    config_path = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy/config.json'
    manifest_path = ROOT / 'data/calibration/navigation_awq_learning_20261104/processed/manifest.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    merge = config['vision_config']['spatial_merge_size']
    print('MODE: read-only CPU inspection; no model loading, pruning, or inference')
    print('DeepStack vision block indices (zero-based):',
          config['vision_config']['deepstack_visual_indexes'])
    print('Candidate budget: keep 75% of image tokens, rounded up per image')

    if len(manifest) < 2:
        raise ValueError('Expected the existing two-image and nine-image teaching samples')
    for entry in manifest[:2]:
        path = (ROOT / entry['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Sample path must stay inside the project')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Saved input hash mismatch: ' + entry['id'])
        inputs = torch.load(path, map_location='cpu', weights_only=True)
        grid = inputs['image_grid_thw']
        if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any():
            raise ValueError('Invalid image grid: ' + entry['id'])
        if (grid[:, 1:] % merge).any():
            raise ValueError('Image grid does not align with spatial merge size')
        per_image = (grid.prod(-1) // (merge * merge)).tolist()
        total = inputs['input_ids'].numel()
        visual = int((inputs['input_ids'] == config['image_token_id']).sum())
        if (visual != sum(per_image) or total != entry['sequence_length']
                or len(per_image) != entry['images']):
            raise ValueError('Input token counts disagree: ' + entry['id'])
        kept_per_image = [(int(count) * 3 + 3) // 4 for count in per_image]
        kept = sum(kept_per_image)
        other = total - visual
        print('\nSample:', entry['id'])
        print('  Images:', len(per_image))
        print('  Image tokens per image:', per_image)
        print(f'  Original: total={total}, image={visual}, text_and_special={other}')
        print('  Candidate image tokens per image:', kept_per_image)
        print(f'  Candidate: image={kept}, total={other + kept}, actual_keep={kept / visual:.2%}')

    print('\nDART_INPUT_BUDGET_CHECK_OK')
    print('Budget only; DART integration, feature/position/KV alignment remain unverified.')


if __name__ == '__main__':
    main()
