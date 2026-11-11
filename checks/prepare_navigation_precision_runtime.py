"""Generate isolated precision guards, conversion and runtime configuration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(path, text):
    if (ROOT/path).exists():
        assert (ROOT/path).read_text(encoding='utf-8') == text, 'Existing generated file differs: '+path
        return
    with (ROOT/path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def main():
    guard = (ROOT/'checks/navigation_awq_dart_guard.py').read_text(encoding='utf-8')
    guard = guard.replace('import torch', 'import os\nimport torch\nWB = int(os.environ["PROJECT2_PRECISION_WBITS"])')
    guard = guard.replace('from navigation_awq_dart_profiles import AWQ_SPEC',
                          'from navigation_awq_dart_profiles import AWQ_SPEC as ORIGINAL_SPEC\nAWQ_SPEC = dict(ORIGINAL_SPEC, bits=WB)')
    guard = guard.replace('4, 128, in_features, out_features', 'WB, 128, in_features, out_features')
    guard = guard.replace('out_features // 8', 'out_features // (32 // WB)')
    guard = guard.replace("'AwqLinearW4A16Impl'", "'NavigationPrecisionImpl'")
    guard = guard.replace('native LMDeploy W4A16 modules', 'isolated packed precision modules')
    guard = guard.replace('FP16 activations with packed INT4 weights', 'FP16 interfaces with separately audited packed precision kernels')
    guard = guard.replace('exact AWQ GEMM W4/group128/asymmetric configuration', 'exact requested W/group128/asymmetric storage configuration')
    save('checks/navigation_precision_guard.py', guard)
    # Derive byte-packed W8 converters from the proven algorithms, with explicit provenance checks.
    for algorithm, original in [('AWQ','learning'),('GPTQ','stage2')]:
        lower = algorithm.lower()
        code = (ROOT/f'checks/convert_navigation_{lower}_{original}.py').read_text(encoding='utf-8')
        for old, new in [("(4, 128, False, 'int', 'group')", "(8, 128, False, 'int', 'group')"),
                         ('INT4', 'INT8'), ('value.shape[1] % 8', 'value.shape[1] % 4'),
                         ('value.max() <= 15', 'value.max() <= 255'),
                         ('order = [0, 2, 4, 6, 1, 3, 5, 7]', 'order = [0, 1, 2, 3]'),
                         ('value.shape[0], -1, 8', 'value.shape[0], -1, 4'),
                         ('torch.arange(8, dtype=torch.int32) * 4', 'torch.arange(4, dtype=torch.int32) * 8'),
                         ('& 15 for shift', '& 255 for shift'), ('range(8)', 'range(4)'),
                         ('weight_packed\'], 4, shape', 'weight_packed\'], 8, shape'),
                         ('weight_zero_point\'], 4, scales.shape', 'weight_zero_point\'], 8, scales.shape'),
                         ('(signed + 8)', '(signed + 128)'), ('(zeros + 8)', '(zeros + 128)'),
                         ('bits=4, group_size=128', 'bits=8, group_size=128'),
                         ("deployment_format='awq_gemm'", "deployment_format='project2_byte_packed_w8'"),
                         ('integer layout only', 'integer layout only; requires isolated W8 kernel')]:
            if algorithm == 'AWQ' and old == "deployment_format='awq_gemm'":
                continue
            assert old in code, (algorithm, old)
            code = code.replace(old, new)
        marker = "    source_config = read_json(source / 'config.json')"
        if lower == 'awq':
            code = code.replace(marker, "    provenance = read_json(source / 'project2_quantization.json')\n    assert provenance['config']['algorithm'] == 'AWQ' and provenance['config']['bits'] == 8\n"+marker)
            code = code.replace('report = dict(', "report = dict(quantization_algorithm='AWQ', deployment_format='project2_byte_packed_w8', ")
        save(f'checks/convert_navigation_{lower}_w8_20261110.py', code)
    groups = {}
    for algorithm in ('awq','gptq'):
        for precision, wb, ab in [('w4a4',4,4),('w4a8',4,8),('w8a16',8,16)]:
            model = (f'models/SimpleVLN_Qwen3VL_4B_RL_{algorithm.upper()}_W8_20261110' if wb == 8 else
                     ('models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104' if algorithm=='awq' else
                      'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_stage2_20261109'))
            groups[algorithm+'_'+precision] = dict(algorithm=algorithm, weight_bits=wb, activation_bits=ab,
                 model=model, keep_ratio=1., activation_recipe='none' if ab==16 else
                 'dynamic_per_token_symmetric_maxabs_narrow_range',
                 compute='fp16_mma' if ab==16 else 'int8_mma',
                 activation_storage='fp16' if ab==16 else 'int8' if ab==8 else 'packed_int4')
    save('setup/navigation-precision-20261110.json', json.dumps(dict(version='precision_20261110',
        scope='Experimental project-specific kernels; not stock LMDeploy support; no Token compression',
        w4_reuse='Reuse frozen AWQ/GPTQ W4 weights; add identical dynamic activation quantization at runtime',
        w8_calibration='Same32 inputs, same group128 asymmetric weights and algorithms; change only weight bits',
        retained_precision='FP16 vision, merger, embeddings, norms, attention/KV, output head and inter-projection tensors',
        w4a4_caveat='4-bit packed activation operands widened to INT8 MMA; no native INT4 MMA speed claim',
        comparison_caveat='Old W4A16 uses stock kernel; new groups use custom kernels. Results compare deployable configurations, not isolated bitwidth causality.',
        frozen_cases='datasets/navigation_20261110/selection.json', groups=groups), indent=2)+'\n')


if __name__ == '__main__':
    main()
