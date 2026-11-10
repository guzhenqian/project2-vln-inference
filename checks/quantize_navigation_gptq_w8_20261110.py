"""Isolated W8 GPTQ calibration: consume prepared inputs and preserve all previous artifacts."""
import argparse
import hashlib
import json
import re
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Expected a path inside project: ' + value)
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='setup/navigation-gptq-stage2-20261109.json')
    parser.add_argument('--check-only', action='store_true', help='Validate inputs and recipe without loading model weights')
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = read_json(config_path)
    source = project_path(config['source'])
    manifest_path = project_path(config['calibration_manifest'])
    output, log_dir = project_path(config['output']), project_path(config['log_dir'])
    for path in (output, log_dir):
        if path.exists():
            raise FileExistsError('Preserving existing path; choose a new run: ' + str(path))
    if output == log_dir or output in log_dir.parents or log_dir in output.parents:
        raise ValueError('Model and log directories must be separate')
    if config['algorithm'] != 'GPTQ' or config['batch_size'] != 1:
        raise ValueError('This lesson supports GPTQ with batch_size=1')

    import torch
    from compressed_tensors.quantization import preset_name_to_scheme
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from loguru import logger

    torch.set_num_threads(4)
    scheme = preset_name_to_scheme('W4A16_ASYM', targets=config['targets'])
    scheme.weights.num_bits = 8
    assert config['bits'] == 8
    if (scheme.weights.num_bits, scheme.weights.group_size, scheme.weights.symmetric) != (
            config['bits'], config['group_size'], config['symmetric']):
        raise ValueError('Requested parameters disagree with the installed quantization preset')
    if scheme.input_activations is not None or scheme.output_activations is not None:
        raise ValueError('Expected weight-only quantization')
    recipe = GPTQModifier(config_groups={'group_0': scheme}, ignore=config['ignore'],
                          actorder=config['actorder'], block_size=config['block_size'],
                          dampening_frac=config['dampening_frac'], offload_hessians=config['offload_hessians'])
    common = read_json(ROOT / 'setup/navigation-stage2-common-20261109.json')
    if any(version(name) != expected for name, expected in common['quantization_common']['quantization_versions'].items()):
        raise ValueError('Quantization environment changed')
    print('Verifying frozen source weights and reference assets', flush=True)
    if digest(source / 'model.safetensors') != common['model']['source_weights_sha256']:
        raise ValueError('Source weight hash mismatch')
    for relative, expected in common['reference_files_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Frozen source asset changed: ' + relative)
    records = read_json(manifest_path)
    provenance_path = manifest_path.parent / 'provenance.json'
    provenance = read_json(provenance_path)
    samples_path = project_path(provenance['source_samples'])
    if digest(samples_path) != provenance['samples_sha256'] or project_path(provenance['processor_source']) != source:
        raise ValueError('Calibration provenance changed')
    samples = read_json(samples_path)
    if len(records) != config['calibration_samples'] or [r['id'] for r in records] != [s['id'] for s in samples]:
        raise ValueError('Calibration membership/order differs from the prepared set')
    source_config = read_json(source / 'config.json')
    if not (source / 'model.safetensors').is_file():
        raise FileNotFoundError('Source model weights missing')
    dataset_rows = []
    for record in records:
        path = project_path(record['path'])
        if path.parent != manifest_path.parent or digest(path) != record['sha256']:
            raise ValueError('Prepared tensor path/hash mismatch: ' + record['id'])
        features = torch.load(path, weights_only=True, map_location='cpu')
        ids = features['input_ids']
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] != record['sequence_length']:
            raise ValueError('Input sequence shape mismatch: ' + record['id'])
        if ids.shape[1] + config['reserved_output_tokens'] > config['max_seq_length']:
            raise ValueError('Context budget exceeded: ' + record['id'])
        if features['image_grid_thw'].shape[0] != record['images']:
            raise ValueError('Image count mismatch: ' + record['id'])
        dataset_rows.append({'path': record['path'], 'input_ids': ids[0].tolist()})
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the quantization environment')
    resolved = dict(config=config, resolved_scheme=scheme.model_dump(mode='json'),
                    config_sha256=digest(config_path), manifest_sha256=digest(manifest_path),
                    provenance_sha256=digest(provenance_path),
                    source_config_sha256=digest(source / 'config.json'),
                    versions={n: version(n) for n in ['torch', 'transformers', 'llmcompressor', 'compressed-tensors']})
    print('GPTQ_LEARNING_PREFLIGHT_OK samples=%d W%dA16 group=%d symmetric=%s' %
          (len(records), config['bits'], config['group_size'], config['symmetric']), flush=True)
    if args.check_only:
        print('CHECK_ONLY: model weights not loaded; no quantization or output writes', flush=True)
        return

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
    from compressed_tensors.utils import remove_dispatch

    # Reserve fresh destinations before loading weights. Failures keep their evidence.
    output.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    save_json(log_dir / 'run_config.json', resolved)
    started = time.time()
    torch.manual_seed(config['seed'])
    print('STAGE 1/3: load source weights on CPU as FP16, then use FP32 calibration arithmetic', flush=True)
    processor = AutoProcessor.from_pretrained(str(source), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(source), local_files_only=True, torch_dtype=torch.float16,
        device_map='cpu', attn_implementation='sdpa')
    model.to(torch.float32)
    model.eval()
    model.config.use_cache = False
    target_names = [name for name, layer in model.named_modules()
                    if isinstance(layer, torch.nn.Linear) and any(
                        re.fullmatch(pattern.removeprefix('re:'), name) for pattern in config['targets'])]
    if len(target_names) != source_config['text_config']['num_hidden_layers'] * 7:
        raise ValueError('Unexpected quantization target count: %d' % len(target_names))
    save_json(log_dir / 'target_layers.json', target_names)

    def collate(batch):
        if len(batch) != 1:
            raise ValueError('Expected one complete multimodal sample per batch')
        return torch.load(project_path(batch[0]['path']), weights_only=True, map_location='cpu')

    original_dispatch = sequential_pipeline.dispatch_for_sequential

    def dispatch_with_resident_vision(value):
        original_dispatch(value)
        # Qwen3-VL reads position-embedding device outside its forward hook.
        remove_dispatch(value.model.visual)
        value.model.visual.to('cuda')
        return value

    sequential_pipeline.dispatch_for_sequential = dispatch_with_resident_vision
    print('STAGE 2/3: GPTQ calibration of %d language Linear layers; then compressed export' % len(target_names), flush=True)
    fallback_events = []
    def capture_warning(message):
        if 'Falling back to round-to-nearest' in str(message):
            fallback_events.append(str(message))
    warning_sink = logger.add(capture_warning, level='WARNING')
    try:
        oneshot(model=model, processor=processor, recipe=recipe,
                dataset=Dataset.from_list(dataset_rows), data_collator=collate,
                num_calibration_samples=config['calibration_samples'], batch_size=config['batch_size'],
                max_seq_length=config['max_seq_length'], pipeline='sequential',
                sequential_targets=['Qwen3VLTextDecoderLayer'], sequential_offload_device='cpu',
                output_dir=str(output), log_dir=str(log_dir / 'compressor'))
    finally:
        sequential_pipeline.dispatch_for_sequential = original_dispatch
        logger.remove(warning_sink)
        save_json(log_dir / 'fallback_events.json', fallback_events)
    if fallback_events:
        raise ValueError('GPTQ RTN fallback detected; checkpoint is not accepted')
    print('STAGE 3/3: check exported compressed-tensors configuration', flush=True)
    exported = read_json(output / 'config.json')['quantization_config']
    if exported['format'] != 'pack-quantized' or exported['quantization_status'] != 'compressed':
        raise ValueError('Export is not a packed compressed checkpoint')
    if not list(output.glob('*.safetensors')):
        raise ValueError('Exported weights missing')
    for group in exported['config_groups'].values():
        weights = group['weights']
        if (weights['num_bits'], weights['group_size'], weights['symmetric']) != (
                config['bits'], config['group_size'], config['symmetric']):
            raise ValueError('Exported quantization parameters differ from requested scheme')
    save_json(output / 'project2_quantization.json', dict(
        **resolved, samples=len(records), elapsed_s=time.time() - started,
        quantized_layer_count=len(target_names),
        algorithm='GPTQ', actorder='static', fallback_events=fallback_events,
        status='compressed-tensors exported; deployment conversion and validation pending',
        note='GPTQ static group128 from FP16-rounded source; no AWQ smoothing; zero RTN fallback events.'))
    print('GPTQ_LEARNING_QUANTIZED', output, flush=True)


if __name__ == '__main__':
    main()
