"""User-run DART selection at the layer-2 boundary; CPU FP32 reference only.

Loads selected embedding rows, language layers 0/1 and final norm from the
unquantized source, rounded to FP16. Reuses audited real vision features.
Does not load the full decoder, generate actions, call a service, or use GPU.
"""

import argparse
import hashlib
import json
import struct
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from check_navigation_dart_features_learning import ROOT, digest, read_json, save_json
from navigation_dart_selection import select_dart_tokens


def inside(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Path must stay inside project: ' + str(value))
    return path


def checked_json(value, expected_hash):
    path = inside(value)
    if digest(path) != expected_hash:
        raise ValueError('Frozen input SHA256 mismatch: ' + value)
    return read_json(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='setup/navigation-dart-selection-learning-20261108.json')
    parser.add_argument('--check-only', action='store_true', help='Read metadata and existing features; no new language forward')
    args = parser.parse_args()

    import torch
    from safetensors import safe_open
    from transformers.modeling_utils import no_init_weights
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel, Qwen3VLTextDecoderLayer, Qwen3VLTextRMSNorm, Qwen3VLTextRotaryEmbedding,
    )

    torch.set_num_threads(4)
    recipe_path = inside(args.config)
    recipe = read_json(recipe_path)
    if recipe['prune_before_language_layer_zero_based'] != 2:
        raise ValueError('This reference lesson implements only the layer-2 boundary')
    model_dir = inside(recipe['model'])
    model_config = checked_json(recipe['model'] + '/config.json', recipe['model_config_sha256'])
    for section in (model_config, model_config['text_config'], model_config['vision_config']):
        if section.get('quantization_config') is not None:
            raise ValueError('DART-only reference requires an unquantized checkpoint')
    feature_report = checked_json(recipe['feature_report'], recipe['feature_report_sha256'])
    audit = checked_json(recipe['source_audit'], recipe['source_audit_sha256'])
    if (audit['status'] != 'PASS' or audit['source'] != recipe['model'] + '/model.safetensors'
            or audit['rounded_source_vision_sha256'] != feature_report['loaded_vision_tensor_sha256']
            or audit['previous_report'] != recipe['feature_report']):
        raise ValueError('Previously computed vision features are not bound to the unquantized FP16 source audit')
    if (feature_report['status'] != 'complete' or feature_report['compute_dtype'] != 'float32'
            or feature_report['decoder_loaded'] or feature_report['token_pruning_performed']):
        raise ValueError('Expected the completed, unpruned CPU vision reference')
    feature_path = inside(recipe['feature_report']).parent / 'features.pt'
    if digest(feature_path) != feature_report['feature_sha256']:
        raise ValueError('Saved feature SHA256 mismatch')
    features = torch.load(feature_path, map_location='cpu', weights_only=True)
    sample_path = inside(feature_report['sample_path'])
    if digest(sample_path) != feature_report['sample_sha256']:
        raise ValueError('Saved processor input SHA256 mismatch')
    inputs = torch.load(sample_path, map_location='cpu', weights_only=True)
    ids, grid = features['input_ids'], features['image_grid_thw']
    if not torch.equal(ids, inputs['input_ids']) or not torch.equal(grid, inputs['image_grid_thw']):
        raise ValueError('Feature bundle and processor input differ')
    if ids.ndim != 2 or ids.shape[0] != 1 or not (inputs['attention_mask'] == 1).all():
        raise ValueError('This reference requires one unpadded sample')
    length = ids.shape[1]
    mask = ids[0] == model_config['image_token_id']
    image_positions = mask.nonzero().flatten()
    counts = (grid.prod(-1) // model_config['vision_config']['spatial_merge_size'] ** 2).tolist()
    if counts != features['per_image_token_counts'] or not torch.equal(image_positions, features['image_sequence_indices']):
        raise ValueError('Visual feature-to-sequence mapping mismatch')
    spans = []
    offset = 0
    for count in counts:
        positions = image_positions[offset:offset + count]
        if len(positions) != count or not torch.equal(positions, torch.arange(int(positions[0]), int(positions[0]) + count)):
            raise ValueError('Expected a contiguous image-token span for each image')
        spans.append((int(positions[0]), int(positions[-1]) + 1))
        offset += count
    width = model_config['text_config']['hidden_size']
    if len(features['deepstack']) != 3:
        raise ValueError('Expected three DeepStack branches')
    for tensor in [features['main']] + features['deepstack']:
        if tuple(tensor.shape) != (sum(counts), width) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid saved visual feature shape or values')

    config = Qwen3VLConfig(**model_config)
    config.text_config._attn_implementation = 'sdpa'
    position_ids, original_rope_delta = Qwen3VLModel.get_rope_index(
        SimpleNamespace(config=config), input_ids=ids, image_grid_thw=grid,
        attention_mask=torch.ones_like(ids))
    if position_ids.shape != (3, 1, length):
        raise ValueError('Unexpected full-sequence mRoPE shape')
    weights_path = model_dir / 'model.safetensors'
    with weights_path.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        if header_length > 32 * 1024 * 1024:
            raise ValueError('Unexpected weight header size')
        header = json.loads(stream.read(header_length))
    if any(name.endswith(('.qweight', '.qzeros', '.weight_packed')) for name in header):
        raise ValueError('Quantized weights are forbidden for this DART-only lesson')
    expected = {}
    with torch.device('meta'), no_init_weights():
        for index in range(2):
            layer = Qwen3VLTextDecoderLayer(config.text_config, layer_idx=index)
            expected.update({f'model.language_model.layers.{index}.{name}': list(value.shape)
                             for name, value in layer.state_dict().items()})
        del layer
    expected['model.language_model.norm.weight'] = [width]
    expected['model.language_model.embed_tokens.weight'] = [config.text_config.vocab_size, width]
    for name, shape in expected.items():
        if name not in header or header[name]['shape'] != shape or header[name]['dtype'] not in ('F16', 'F32'):
            raise ValueError('Unquantized language weight layout mismatch: ' + name)
    if length - spans[-1][1] < recipe['text_pivots']:
        raise ValueError('Too few suffix positions for the specified text pivots')
    print('DART_SELECTION_PREFLIGHT_OK', flush=True)
    print('Unquantized source:', recipe['model'], flush=True)
    print(f'Input={length} tokens; images={len(counts)}; image tokens={counts}', flush=True)
    print('Reference: layers 0/1 + DeepStack 0/1, selection before layer 2; CPU FP32 from FP16 weights', flush=True)
    if args.check_only:
        print('CHECK_ONLY: no new language weights loaded, forward, selection, or output writes', flush=True)
        return

    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < 3 * 1024 ** 3:
        raise RuntimeError('At least 3 GiB available CPU memory is required for this reference check')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'logs' / ('navigation_dart_selection_learning_' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(recipe=recipe, recipe_sha256=digest(recipe_path),
                      script_sha256=digest(Path(__file__)), selector_sha256=digest(Path(__file__).with_name('navigation_dart_selection.py')),
                      source_feature_sha256=feature_report['feature_sha256'],
                      language_weights_source=weights_path.relative_to(ROOT).as_posix(),
                      checkpoint_quantized=False, weight_rounding_dtype='float16', compute_dtype='float32', device='cpu',
                      versions={name: version(name) for name in ('torch', 'transformers', 'safetensors')})
    save_json(output / 'input.json', provenance)
    started = time.perf_counter()

    class KeyCapture:
        key = None

        def update(self, key, value, layer_idx, cache_kwargs):
            if layer_idx == 1:
                self.key = key.detach().clone()
            return key, value

    capture = KeyCapture()
    language_hash = hashlib.sha256()

    def fp16_weight(source, name):
        weight = source.get_tensor(name).to(dtype=torch.float16)
        if not torch.isfinite(weight).all():
            raise ValueError('Non-finite FP16 source weight: ' + name)
        language_hash.update(name.encode() + b'\0' + weight.numpy().tobytes())
        return weight

    try:
        with torch.inference_mode(), safe_open(str(weights_path), framework='pt', device='cpu') as source:
            # Gather only requested rows; no full vocabulary embedding copy or output head is loaded.
            embed_name = 'model.language_model.embed_tokens.weight'
            hidden = source.get_tensor(embed_name)[ids].to(torch.float16).float()
            if not torch.isfinite(hidden).all():
                raise ValueError('Non-finite embedding rows')
            language_hash.update(b'input_ids\0' + ids.numpy().tobytes())
            language_hash.update(b'embedding_rows_fp16\0' + hidden.to(torch.float16).numpy().tobytes())
            hidden[:, mask, :] = features['main']
            rotary = Qwen3VLTextRotaryEmbedding(config.text_config)
            position_embeddings = rotary(hidden, position_ids)
            causal_mask = torch.full((length, length), torch.finfo(torch.float32).min).triu(diagonal=1)[None, None]
            cache_position = torch.arange(length)
            for index in range(2):
                print(f'Computing language layer {index} and injecting DeepStack {index}...', flush=True)
                with no_init_weights():
                    layer = Qwen3VLTextDecoderLayer(config.text_config, layer_idx=index).float().eval()
                prefix = f'model.language_model.layers.{index}.'
                state = {name[len(prefix):]: fp16_weight(source, name) for name in sorted(expected) if name.startswith(prefix)}
                layer.load_state_dict(state, strict=True)
                del state
                hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=causal_mask,
                               position_ids=position_ids[0], past_key_values=capture,
                               cache_position=cache_position, use_cache=False)
                hidden[:, mask, :] += features['deepstack'][index]
                del layer
            norm = Qwen3VLTextRMSNorm(width, eps=config.text_config.rms_norm_eps)
            norm.load_state_dict({'weight': fp16_weight(source, 'model.language_model.norm.weight')}, strict=True)
            normalized_hidden = norm(hidden)[0]
            if capture.key is None or capture.key.shape != (1, config.text_config.num_key_value_heads, length, config.text_config.head_dim):
                raise ValueError('Did not capture the expected post-mRoPE key states')
            keys = capture.key[0].transpose(0, 1).reshape(length, -1)
            selection = select_dart_tokens(keys, normalized_hidden, mask, spans,
                                           keep_ratio=recipe['keep_ratio'], visual_pivots=recipe['visual_pivots'],
                                           text_pivots=recipe['text_pivots'])
            identity = select_dart_tokens(keys, normalized_hidden, mask, spans, keep_ratio=1.0,
                                          visual_pivots=recipe['visual_pivots'], text_pivots=recipe['text_pivots'])
            seq_keep, visual_keep = selection['sequence_keep_indices'], selection['visual_keep_indices']
            if not torch.equal(identity['sequence_keep_indices'], torch.arange(length)) or not torch.equal(identity['visual_keep_indices'], torch.arange(sum(counts))):
                raise ValueError('100% keep failed identity verification')
            compact_ids = ids[:, seq_keep]
            compact_mask = compact_ids[0] == config.image_token_id
            kept_positions = position_ids[:, :, seq_keep]
            compact_deepstack = [tensor[visual_keep] for tensor in features['deepstack']]
            if int(compact_mask.sum()) != len(visual_keep):
                raise ValueError('Compacted visual mask and feature rows disagree')
            non_visual = (~mask).nonzero().flatten()
            if not torch.equal(seq_keep[~compact_mask], non_visual):
                raise ValueError('A non-image token was lost or reordered')
            # Independent full-length scatter reference checks that the pending DS2 addition
            # reaches the same retained positions after sequence compaction.
            full_after_ds2 = hidden.clone()
            full_after_ds2[:, mask, :] += features['deepstack'][2]
            compact_after_ds2 = hidden[:, seq_keep, :].clone()
            compact_after_ds2[:, compact_mask, :] += compact_deepstack[2]
            if not torch.equal(full_after_ds2[:, seq_keep, :], compact_after_ds2):
                raise ValueError('Pending DeepStack scatter does not commute with token selection')
            if not torch.equal(kept_positions[:, :, ~compact_mask], position_ids[:, :, non_visual]):
                raise ValueError('Text mRoPE coordinates changed during selection')
            bundle = {'sequence_keep_indices': seq_keep, 'visual_keep_indices': visual_keep,
                      'input_ids': compact_ids, 'image_mask': compact_mask,
                      'boundary_hidden_unnormalized': hidden[:, seq_keep, :],
                      'mrope_position_ids': kept_positions,
                      'main': features['main'][visual_keep], 'deepstack': compact_deepstack,
                      'pending_deepstack_indices': [2],
                      'original_image_grid_coordinates_thw': features['original_image_grid_coordinates_thw'][visual_keep],
                      'image_index_per_feature': features['image_index_per_feature'][visual_keep],
                      'original_rope_delta': original_rope_delta}
            with (output / 'selection.pt').open('xb') as stream:
                torch.save(bundle, stream)
        report = dict(provenance, status='complete', selection=selection['diagnostics'],
                      before_sequence_tokens=length, after_sequence_tokens=len(seq_keep),
                      before_visual_tokens=sum(counts), after_visual_tokens=len(visual_keep),
                      language_reference_loaded_weight_sha256=language_hash.hexdigest(),
                      selection_sha256=digest(output / 'selection.pt'),
                      checks={'keep_100_percent_identity': True, 'all_non_image_positions_preserved': True,
                              'retained_original_mrope_coordinates': True, 'pending_deepstack_index_only_scatter_equivalent': True},
                      elapsed_seconds=time.perf_counter() - started,
                      limitations=['CPU FP32 reference; FP16 serving may select different near-tied tokens',
                                   'No LMDeploy runtime, production KV cache or generated actions tested',
                                   'DeepStack scatter checked at the boundary; layer 2 and subsequent decoder layers not run',
                                   'Independent grid semantic alignment not proven by this index check'])
        save_json(output / 'report.json', report)
        print(f'Image tokens: {sum(counts)} -> {len(visual_keep)}', flush=True)
        print(f'Total sequence: {length} -> {len(seq_keep)}; all text/special tokens retained', flush=True)
        print('PASS: 100% identity, original mRoPE coordinates, pending DeepStack index mapping', flush=True)
        print('Saved:', output.relative_to(ROOT).as_posix(), flush=True)
        print('DART_LAYER2_SELECTION_READY', flush=True)
        print('CPU reference only; LMDeploy/KV/generation/navigation validation remains pending.', flush=True)
    except Exception as error:
        save_json(output / 'failure.json', {'status': 'failed', 'type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
