"""User-run CPU vision reference for DART preparation; does not select tokens.

Reads only vision weights from the unquantized source checkpoint, rounds them
to the FP16 baseline precision, and computes with FP32 on CPU. This is a feature-layout check, not an LMDeploy,
FP16 numerical-equivalence, DART-algorithm, or performance validation.
"""

import argparse
import hashlib
import json
import math
import struct
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
MANIFEST = ROOT / 'data/calibration/navigation_awq_learning_20261104/processed/manifest.json'
PREFIX = 'model.visual.'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-index', type=int, choices=(0, 1), default=0)
    parser.add_argument('--check-only', action='store_true',
                        help='Check CPU input and weight metadata without loading weights or running the vision tower')
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from transformers.modeling_utils import no_init_weights
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.set_num_threads(4)
    config_path = MODEL / 'config.json'
    config = read_json(config_path)
    for section in (config, config['text_config'], config['vision_config']):
        if section.get('quantization_config') is not None:
            raise ValueError('DART-only preparation requires an unquantized checkpoint')
    records = read_json(MANIFEST)
    record = records[args.sample_index]
    sample_path = (ROOT / record['path']).resolve()
    if not sample_path.is_relative_to(ROOT) or digest(sample_path) != record['sha256']:
        raise ValueError('Saved sample path or SHA256 mismatch')
    inputs = torch.load(sample_path, map_location='cpu', weights_only=True)
    grid = inputs['image_grid_thw']
    merge = config['vision_config']['spatial_merge_size']
    if grid.ndim != 2 or grid.shape[1] != 3 or (grid <= 0).any() or (grid[:, 1:] % merge).any():
        raise ValueError('Invalid image grid')
    counts = (grid.prod(-1) // (merge * merge)).tolist()
    ids = inputs['input_ids']
    if ids.shape != (1, record['sequence_length']) or len(counts) != record['images']:
        raise ValueError('Input dimensions disagree with the saved manifest')
    image_sequence_indices = (ids[0] == config['image_token_id']).nonzero().flatten()
    if image_sequence_indices.numel() != sum(counts):
        raise ValueError('Image token count does not match the image grid')
    if inputs['pixel_values'].shape[0] != int(grid.prod(-1).sum()):
        raise ValueError('Pixel patch count does not match the image grid')

    weights_path = MODEL / 'model.safetensors'
    with weights_path.open('rb') as stream:
        length_bytes = stream.read(8)
        header_length = struct.unpack('<Q', length_bytes)[0]
        if header_length > 32 * 1024 * 1024:
            raise ValueError('Unexpectedly large safetensors header')
        header_bytes = stream.read(header_length)
    header = json.loads(header_bytes)
    if any(key.endswith(('.qweight', '.qzeros', '.weight_packed')) for key in header):
        raise ValueError('Quantized tensors are not allowed in the DART-only source checkpoint')
    vision_header = {key[len(PREFIX):]: value for key, value in header.items() if key.startswith(PREFIX)}
    if not vision_header or any(value['dtype'] not in ('F32', 'F16') for value in vision_header.values()):
        raise ValueError('Expected unquantized F32 or F16 source vision weights')
    vision_config = Qwen3VLVisionConfig(**config['vision_config'])
    vision_config._attn_implementation = 'sdpa'
    with torch.device('meta'), no_init_weights():
        layout_model = Qwen3VLVisionModel(vision_config)
    expected = {key: list(value.shape) for key, value in layout_model.state_dict().items()}
    if expected != {key: value['shape'] for key, value in vision_header.items()}:
        raise ValueError('Installed vision model state layout differs from saved vision weights')
    del layout_model

    print('DART_FEATURE_PREFLIGHT_OK', flush=True)
    print('Unquantized source:', MODEL.relative_to(ROOT).as_posix(), flush=True)
    print(f'Sample={record["id"]}; images={len(counts)}; visual_tokens={sum(counts)}', flush=True)
    print(f'Unquantized vision tensors={len(vision_header)}; DeepStack branches='
          f'{len(config["vision_config"]["deepstack_visual_indexes"])}', flush=True)
    print('Execution: source weights rounded to FP16, then CPU FP32 reference; no decoder, GPU, or token pruning', flush=True)
    if args.check_only:
        print('CHECK_ONLY: metadata and input checks only; no model weights loaded or outputs written', flush=True)
        return

    # A conservative CPU working-memory guard; the source checkpoint is never fully loaded.
    float32_weight_bytes = sum(math.prod(shape) * 4 for shape in expected.values())
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < float32_weight_bytes * 2 + 1024 ** 3:
            raise RuntimeError('Insufficient available CPU memory for this isolated vision check')

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'logs' / ('navigation_dart_features_learning_' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        'scope': 'CPU FP32 vision-reference layout check only; not DART or serving compatibility',
        'sample_index': args.sample_index, 'sample_id': record['id'],
        'sample_path': record['path'], 'sample_sha256': record['sha256'],
        'manifest_sha256': digest(MANIFEST), 'model_config_sha256': digest(config_path),
        'weight_source': weights_path.relative_to(ROOT).as_posix(),
        'weight_header_sha256': hashlib.sha256(length_bytes + header_bytes).hexdigest(),
        'weight_header_hash_note': 'Header only, not a full checkpoint content hash',
        'vision_tensor_count': len(vision_header),
        'source_vision_dtypes': sorted({value['dtype'] for value in vision_header.values()}),
        'checkpoint_quantized': False, 'weight_rounding_dtype': 'float16',
        'compute_dtype': 'float32', 'device': 'cpu', 'attention': 'sdpa',
        'deepstack_vision_indices_zero_based': config['vision_config']['deepstack_visual_indexes'],
        'versions': {name: version(name) for name in ('torch', 'transformers', 'safetensors')},
        'token_pruning_performed': False, 'decoder_loaded': False,
    }
    save_json(output / 'input.json', provenance)
    started = time.perf_counter()
    try:
        print('Loading only the vision tower and mergers onto CPU...', flush=True)
        with no_init_weights():
            model = Qwen3VLVisionModel(vision_config).float().eval()
        vision_hash = hashlib.sha256()
        with safe_open(str(weights_path), framework='pt', device='cpu') as source:
            state = {}
            for name in sorted(expected):
                tensor = source.get_tensor(PREFIX + name).to(dtype=torch.float16)
                if not torch.isfinite(tensor).all():
                    raise ValueError('Non-finite FP16-rounded vision weight: ' + name)
                vision_hash.update(name.encode('utf-8') + b'\0')
                vision_hash.update(tensor.numpy().tobytes())
                state[name] = tensor
            model.load_state_dict(state, strict=True)
            del state, tensor
        if any(value.device.type != 'cpu' or value.dtype != torch.float32 for value in model.parameters()):
            raise ValueError('Expected all vision parameters to be CPU FP32')
        print('Computing real image features on CPU; this may take a little while...', flush=True)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            features, deepstack = model(inputs['pixel_values'].float(), grid_thw=grid)
        reference_forward_seconds = time.perf_counter() - inference_started
        expected_shape = (sum(counts), config['vision_config']['out_hidden_size'])
        if len(deepstack) != len(config['vision_config']['deepstack_visual_indexes']):
            raise ValueError('Unexpected DeepStack branch count')
        for label, tensor in [('main', features)] + [(f'deepstack_{i}', x) for i, x in enumerate(deepstack)]:
            if tuple(tensor.shape) != expected_shape or not torch.isfinite(tensor).all():
                raise ValueError('Invalid feature dimensions or non-finite values: ' + label)

        # These are original per-image grid coordinates, not absolute language-model mRoPE IDs.
        coordinates, image_ids = [], []
        for image_index, (frames, height, width) in enumerate(grid.tolist()):
            coordinates.append(torch.cartesian_prod(torch.arange(frames), torch.arange(height // merge),
                                                    torch.arange(width // merge)))
            image_ids.append(torch.full((counts[image_index],), image_index, dtype=torch.long))
        coordinates = torch.cat(coordinates)
        image_ids = torch.cat(image_ids)
        if coordinates.shape != (sum(counts), 3) or image_ids.numel() != sum(counts):
            raise ValueError('Original image coordinate layout mismatch')

        feature_path = output / 'features.pt'
        with feature_path.open('xb') as stream:
            torch.save({'main': features, 'deepstack': deepstack,
                        'image_grid_thw': grid, 'input_ids': ids,
                        'image_sequence_indices': image_sequence_indices,
                        'image_index_per_feature': image_ids,
                        'original_image_grid_coordinates_thw': coordinates,
                        'per_image_token_counts': counts}, stream)
        report = dict(provenance, status='complete',
                      loaded_vision_tensor_sha256=vision_hash.hexdigest(),
                      main_shape=list(features.shape), deepstack_shapes=[list(x.shape) for x in deepstack],
                      feature_sha256=digest(feature_path), per_image_token_counts=counts,
                      reference_forward_seconds=reference_forward_seconds,
                      total_check_seconds=time.perf_counter() - started,
                      limitations=['CPU FP32 reference is not LMDeploy FP16 numerical validation',
                                   'Shapes checked; image coordinates constructed; semantic alignment not independently verified',
                                   'Absolute mRoPE, KV cache, and decoder integration are not tested'])
        save_json(output / 'report.json', report)
        print('Main features:', list(features.shape), flush=True)
        for index, feature in enumerate(deepstack):
            print(f'DeepStack {index}: {list(feature.shape)}', flush=True)
        print('Original image coordinates:', list(coordinates.shape), flush=True)
        print('Saved:', output.relative_to(ROOT).as_posix(), flush=True)
        print('DART_VISUAL_FEATURES_READY', flush=True)
        print('No token pruning performed; selection point and decoder integration remain pending.', flush=True)
    except Exception as error:
        save_json(output / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                          'error': str(error), 'elapsed_seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
