"""Strict static-GPTQ conversion to native AWQ GEMM layout; the algorithm stays GPTQ."""
import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Expected a path inside project: ' + value)
    return path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_header(path):
    # Read metadata only; never touch tensor payloads during --check-only.
    with path.open('rb') as stream:
        first = stream.read(8)
        if len(first) != 8:
            raise ValueError('Invalid safetensors header: ' + str(path))
        size = struct.unpack('<Q', first)[0]
        if not 0 < size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError('Invalid safetensors header size: ' + str(path))
        header = json.loads(stream.read(size))
    return {key: value for key, value in header.items() if key != '__metadata__'}


def preflight(source, reference, output):
    if output.exists():
        raise FileExistsError('Preserving existing output; choose a new directory: ' + str(output))
    if any(output == path or output in path.parents or path in output.parents
           for path in (source, reference)):
        raise ValueError('Output must be separate from source and reference')
    provenance = read_json(source / 'project2_quantization.json')
    if provenance.get('algorithm') != 'GPTQ' or provenance.get('actorder') != 'static' or provenance.get('fallback_events') != []:
        raise ValueError('Require completed static GPTQ provenance without RTN fallback')
    source_config = read_json(source / 'config.json')
    config = read_json(reference / 'config.json')
    quant = source_config['quantization_config']
    if (quant.get('quant_method'), quant.get('format'), quant.get('quantization_status')) != (
            'compressed-tensors', 'pack-quantized', 'compressed'):
        raise ValueError('Expected a packed compressed-tensors checkpoint')
    groups = quant.get('config_groups', {})
    if not groups:
        raise ValueError('Quantization groups missing')
    for group in groups.values():
        weights = group.get('weights', {})
        if weights.get('actorder') not in ('static', 'weight'):
            raise ValueError('Only static GPTQ grouping can use this layout')
        if (weights.get('num_bits'), weights.get('group_size'), weights.get('symmetric'),
                weights.get('type'), weights.get('strategy')) != (8, 128, False, 'int', 'group'):
            raise ValueError('Expected asymmetric INT8 group128 weights')
        if group.get('input_activations') is not None or group.get('output_activations') is not None:
            raise ValueError('Expected weight-only quantization')
    if config.get('quantization_config') or config['text_config'].get('quantization_config'):
        raise ValueError('Reference must be the unquantized model')
    layers = config['text_config']['num_hidden_layers']
    if layers != 36 or source_config['text_config']['num_hidden_layers'] != layers:
        raise ValueError('Expected the same 36-layer source/reference architecture')
    for key in ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads'):
        if source_config['text_config'][key] != config['text_config'][key]:
            raise ValueError('Source/reference architecture mismatch: ' + key)
    if not (reference / 'model.safetensors').is_file():
        raise FileNotFoundError('Reference model.safetensors missing')
    files = sorted(source.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError('Compressed weight shards missing')
    headers = {}
    for file in files:
        for key, value in read_header(file).items():
            if key in headers:
                raise ValueError('Duplicate tensor across shards: ' + key)
            headers[key] = value
    prefixes = sorted(key[:-len('.weight_packed')] for key in headers if key.endswith('.weight_packed'))
    projections = ['self_attn.' + name for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    projections += ['mlp.' + name for name in ('gate_proj', 'up_proj', 'down_proj')]
    expected = {'model.language_model.layers.%d.%s' % (layer, name)
                for layer in range(layers) for name in projections}
    if set(prefixes) != expected:
        raise ValueError('Expected exactly 252 language projection tensors')
    for prefix in prefixes:
        for suffix in ('weight_packed', 'weight_shape', 'weight_scale', 'weight_zero_point'):
            if prefix + '.' + suffix not in headers:
                raise ValueError('Missing packed tensor component: ' + prefix + '.' + suffix)
        if headers[prefix + '.weight_packed']['dtype'] != 'I32':
            raise ValueError('Expected INT32-packed weights: ' + prefix)
        if headers[prefix + '.weight_zero_point']['dtype'] != 'I32':
            raise ValueError('Expected INT32-packed zero points: ' + prefix)
    return files, config, prefixes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', default='models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source, output, reference = map(project_path, (args.source, args.output, args.reference))
    files, config, prefixes = preflight(source, reference, output)
    input_hashes = dict(source_config_sha256=digest(source / 'config.json'),
                        reference_config_sha256=digest(reference / 'config.json'))
    print('GPTQ_CONVERSION_PREFLIGHT_OK shards=%d layers=%d INT8 group128 asymmetric' %
          (len(files), len(prefixes)), flush=True)
    if args.check_only:
        print('CHECK_ONLY: headers/configurations only; no tensors loaded or output written', flush=True)
        return

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

    def pack_awq(value):
        assert value.dtype == torch.int32 and value.ndim == 2 and value.shape[1] % 4 == 0
        assert value.min() >= 0 and value.max() <= 255
        order = [0, 1, 2, 3]
        grouped = value.reshape(value.shape[0], -1, 4)[:, :, order]
        shifts = torch.arange(4, dtype=torch.int32) * 8
        packed = (grouped << shifts).sum(-1, dtype=torch.int32)
        recovered = torch.stack([(packed >> int(shift)) & 255 for shift in shifts], dim=-1)
        inverse = [order.index(i) for i in range(4)]
        assert torch.equal(recovered[:, :, inverse].reshape(value.shape), value)
        return packed.contiguous()

    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    print('Loading compressed tensors on CPU', flush=True)
    state = {}
    for file in files:
        with safe_open(str(file), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                assert key not in state
                state[key] = reader.get_tensor(key)
    converted, checks = {}, []
    for index, prefix in enumerate(prefixes, 1):
        shape = tuple(state[prefix + '.weight_shape'].tolist())
        mapping = state.get(prefix + '.weight_g_idx')
        if mapping is not None:
            if mapping.dtype not in (torch.int32, torch.int64) or tuple(mapping.shape) != (shape[1],):
                raise ValueError('Invalid GPTQ group mapping: ' + prefix)
            expected_mapping = torch.arange(shape[1], dtype=mapping.dtype) // 128
            if not torch.equal(mapping.cpu(), expected_mapping):
                raise ValueError('Non-contiguous GPTQ group mapping: ' + prefix)
        scales = state[prefix + '.weight_scale']
        assert scales.shape == (shape[0], shape[1] // 128)
        assert torch.isfinite(scales).all() and (scales > 0).all()
        signed = unpack_from_int32(state[prefix + '.weight_packed'], 8, shape).to(torch.int32)
        zeros = unpack_from_int32(state[prefix + '.weight_zero_point'], 8, scales.shape, packed_dim=0).to(torch.int32)
        converted[prefix + '.qweight'] = pack_awq((signed + 128).t().contiguous())
        converted[prefix + '.qzeros'] = pack_awq((zeros + 128).t().contiguous())
        deployed_scales = scales.to(torch.float16)
        assert torch.isfinite(deployed_scales).all() and (deployed_scales > 0).all()
        converted[prefix + '.scales'] = deployed_scales.t().contiguous()
        checks.append(dict(layer=prefix, shape=shape, integer_roundtrip=True, scale_dtype=str(scales.dtype),
                           fp16_scale_max_relative_rounding=float(
                               ((deployed_scales.float() - scales.float()).abs() / scales.float()).max())))
        if index % 36 == 0 or index == len(prefixes):
            print('Converted and roundtrip-checked %d/%d Linear layers' % (index, len(prefixes)), flush=True)
    excluded = {'weight_packed', 'weight_shape', 'weight_scale', 'weight_zero_point', 'weight_g_idx'}
    for key, tensor in state.items():
        if key.rsplit('.', 1)[-1] in excluded:
            continue
        assert 'quantization' not in key
        converted[key] = (tensor.to(torch.float16) if tensor.is_floating_point() else tensor).contiguous()
    print('Checking preserved high-precision modules against source FP16', flush=True)
    unchanged = []
    with safe_open(str(reference / 'model.safetensors'), framework='pt', device='cpu') as original:
        original_keys = set(original.keys())
        for key, tensor in converted.items():
            if key in original_keys or key == 'lm_head.weight':
                source_key = key
                if key == 'lm_head.weight' and key not in original_keys:
                    assert config['text_config'].get('tie_word_embeddings', config.get('tie_word_embeddings', False))
                    source_key = 'model.language_model.embed_tokens.weight'
                assert source_key in original_keys, key
                assert torch.equal(tensor, original.get_tensor(source_key).to(torch.float16)), key
                unchanged.append(key)
    if 'lm_head.weight' in converted and 'lm_head.weight' not in original_keys:
        assert torch.equal(converted['lm_head.weight'], converted['model.language_model.embed_tokens.weight'])
        del converted['lm_head.weight']
    print('Saving AWQ weights and conversion audit', flush=True)
    save_file(converted, str(output / 'model.safetensors'), metadata={'format': 'pt'})
    for file in reference.iterdir():
        if (file.is_file() and file.suffix in ('.json', '.jinja')
                and file.name not in ('config.json', 'serving_adapter.json', 'model.safetensors.index.json')):
            shutil.copyfile(str(file), str(output / file.name))
    quant = dict(quant_method='awq', version='gemm', bits=8, group_size=128, zero_point=True)
    config['quantization_config'] = quant
    config['text_config']['quantization_config'] = quant
    for item in (config, config['text_config'], config['vision_config']):
        item['dtype'] = 'float16'
        item['torch_dtype'] = 'float16'
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    report = dict(
        quantization_algorithm='GPTQ', activation_order='static', deployment_format='project2_byte_packed_w8',
        source_quantization_sha256=digest(source / 'project2_quantization.json'),
        source_shards_sha256={path.name: digest(path) for path in files},
        status='all quantized integers and zero-points roundtrip exactly; FP32 scales rounded to deployment FP16 with error reported',
        conversion='integer layout only; requires isolated W8 kernel, no second integer quantization; unquantized weights/scales cast to runtime FP16',
        source=source.relative_to(ROOT).as_posix(), reference=reference.relative_to(ROOT).as_posix(),
        output=output.relative_to(ROOT).as_posix(), **input_hashes,
        layers=checks, quantized_layer_count=len(prefixes), unchanged_fp16_tensors=unchanged,
        bytes=(output / 'model.safetensors').stat().st_size, sha256=digest(output / 'model.safetensors'))
    with (output / 'conversion_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print('GPTQ_LAYOUT_VERIFIED', len(prefixes), report['bytes'], report['sha256'], flush=True)


if __name__ == '__main__':
    main()
