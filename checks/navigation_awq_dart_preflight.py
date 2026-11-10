"""Read-only, standard-library D-group checkpoint and C-source verification.

The default reads only checkpoint metadata, not tensor payloads. Header checks
establish layout, not current full-weight integrity or successful CUDA use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from navigation_awq_dart_profiles import AWQ_SPEC, MODEL_RELATIVE

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / MODEL_RELATIVE
C_PLAN = ROOT / 'data/baseline/navigation_dart150_keep75_20261108/manifest.json'
C_RUN = ROOT / 'logs/navigation_dart150_keep75_resumed_20261108T135623640378Z_99f2e6a6'
CONFIG_SHA = 'fc18ed52e9298aa32ce48b06e33ab058020de2487534eff9a45455f3e7bd2351'
HEADER_SHA = '05843a73ea4312c7b6b54f1661d7a0dd20db369dfc1bb4208139a5c5b529da54'
AUDIT_SHA = '71f9543df88a1e567b4d892a674278e18029976d3283419860ba0ff08fdde2bb'
WEIGHT_SHA = '31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_protected_sources():
    plan = read(C_PLAN)
    checked = {}
    for relative, expected in plan['source_checks'].items():
        if Path(relative).suffix in ('.py', '.sh'):
            actual = digest(ROOT / relative)
            require(actual == expected, 'C-group locked source changed: ' + relative)
            checked[relative] = actual
    require(len(checked) >= 14, 'Missing C-group source lock evidence')
    return checked


def check_checkpoint(*, full_weights=False):
    require(digest(MODEL / 'config.json') == CONFIG_SHA, 'Frozen AWQ configuration changed')
    config = read(MODEL / 'config.json')
    require(config['quantization_config'] == config['text_config']['quantization_config'] == AWQ_SPEC,
            'Both root and language config must retain the original AWQ configuration')
    require(config['vision_config'].get('quantization_config') is None, 'Vision must remain unquantized')
    require(config['text_config']['num_hidden_layers'] == 36, 'Expected 36 decoder layers')
    weight = MODEL / 'model.safetensors'
    with weight.open('rb') as stream:
        size_bytes = stream.read(8)
        require(len(size_bytes) == 8, 'Missing safetensors header length')
        size = struct.unpack('<Q', size_bytes)[0]
        require(0 < size < 32 * 1024 * 1024, 'Invalid checkpoint header length')
        header_bytes = stream.read(size)
    require(hashlib.sha256(size_bytes + header_bytes).hexdigest() == HEADER_SHA,
            'Frozen AWQ tensor layout/header changed')
    require(weight.stat().st_size == 3496772056, 'AWQ checkpoint file size changed')
    header = json.loads(header_bytes)
    require(digest(MODEL / 'conversion_audit.json') == AUDIT_SHA, 'Original conversion evidence changed')
    converted = read(MODEL / 'conversion_audit.json')
    quantized = {name.removesuffix('.qweight') for name in header if name.endswith('.qweight')}
    layers = {row['layer']: row for row in converted['layers']}
    require(len(quantized) == 252 and quantized == set(layers), 'Expected the original 36 x 7 AWQ projections')
    for name in sorted(quantized):
        row = layers[name]
        output_width, input_width = row['shape']
        require(row['integer_roundtrip'] is True, 'Original integer conversion did not pass')
        for suffix, dtype, shape in (
            ('qweight', 'I32', [input_width, output_width // 8]),
            ('qzeros', 'I32', [input_width // 128, output_width // 8]),
            ('scales', 'F16', [input_width // 128, output_width]),
        ):
            tensor = header[name + '.' + suffix]
            require(tensor['dtype'] == dtype and tensor['shape'] == shape,
                    'Incorrect AWQ layout: ' + name + '.' + suffix)
        require(name + '.weight' not in header, 'Unexpected dense duplicate of quantized projection')
    require(not any(name.endswith('.weight_packed') for name in header), 'CT source layout is not deployable AWQ')
    vision = [value for name, value in header.items() if name.startswith('model.visual.')]
    require(vision and all(value['dtype'] == 'F16' for value in vision), 'Vision checkpoint is not FP16')
    if full_weights:
        require(digest(weight) == WEIGHT_SHA, 'Full checkpoint no longer matches verified B-group weights')
    return {
        'status': 'metadata_layout_passed', 'model_source': MODEL_RELATIVE,
        'quantization': AWQ_SPEC, 'stored_quantized_projections': len(quantized),
        'vision_tensor_count': len(vision), 'vision_dtype': 'float16',
        'config_sha256': CONFIG_SHA, 'weight_header_sha256': HEADER_SHA,
        'weight_header_sha256_scope': '8-byte little-endian length prefix plus JSON header bytes',
        'conversion_audit_sha256': AUDIT_SHA, 'weight_bytes': weight.stat().st_size,
        'historical_full_weight_sha256': WEIGHT_SHA,
        'full_weight_sha256_rechecked_this_run': full_weights,
        'cuda_execution_verified_this_run': False,
    }


def require_c_finished():
    """Fail closed before GPU/resource checks. Never stop or contact C's service."""
    status = read(C_RUN / 'status.json')
    require(status.get('status') == 'complete',
            'C-group is not completed; D-group GPU execution remains disabled')
    require(not (ROOT / 'logs/navigation_dart150_keep75.active.lock').exists(),
            'C-group active lock still exists; do not take over its resources')
    report = read(C_RUN / 'report.json')
    require(report.get('status') == 'complete' and report.get('completed_episode_count') == 150
            and len(report.get('completed_batches', [])) == 3
            and report.get('all_runtime_audits_passed') is True and report.get('error_count') == 0,
            'C-group final report/audits are not complete for all 150 tasks')
    require(report.get('plan_sha256') == digest(C_PLAN), 'C final report refers to another plan')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-weights', action='store_true', help='Also hash 3.50 GB; use after C completes')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.full_weights:
        require_c_finished()
    result = {'checkpoint': check_checkpoint(full_weights=args.full_weights),
              'c_locked_sources': check_protected_sources(),
              'c_status_snapshot': read(C_RUN / 'status.json'),
              'combination_gpu_verified': False, 'formal_d150_started': False}
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
