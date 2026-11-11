"""Prepare isolated W8 group128 calibration drivers; never edit frozen W4 drivers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for algorithm, original in [('AWQ', 'learning'), ('GPTQ', 'stage2')]:
        lower = algorithm.lower()
        source = ROOT / f'checks/quantize_navigation_{lower}_{original}.py'
        target = ROOT / f'checks/quantize_navigation_{lower}_w8_20261110.py'
        config_source = ROOT / ('setup/navigation-awq-learning-20261104.json' if lower == 'awq'
                                else 'setup/navigation-gptq-stage2-20261109.json')
        config = json.loads(config_source.read_text(encoding='utf-8'))
        config.update(version=f'navigation_{lower}_w8_20261110', bits=8,
                      output=f'models/SimpleVLN_Qwen3VL_4B_RL_{algorithm}_W8_CT_20261110',
                      log_dir=f'logs/navigation_{lower}_w8_calibration_20261110')
        config['scheme'] = 'W8A16_ASYM_GROUP128_CUSTOM'
        config['note'] = 'Same32 training inputs and group128 asymmetric weights; only weight bits change to8.'
        code = source.read_text(encoding='utf-8')
        old = "scheme = preset_name_to_scheme(config['scheme'], targets=config['targets'])"
        assert code.count(old) == 1
        code = code.replace(old, "scheme = preset_name_to_scheme('W4A16_ASYM', targets=config['targets'])\n    scheme.weights.num_bits = 8\n    assert config['bits'] == 8")
        old = "targets=config['targets'], scheme=config['scheme'], ignore=config['ignore']"
        assert code.count(old) == 1
        code = code.replace(old, "config_groups={'group_0': scheme}, ignore=config['ignore']")
        if lower == 'awq':
            # Match the stronger source-file checks in the frozen GPTQ driver.
            marker = "    records = read_json(manifest_path)"
            verification = """    common = read_json(ROOT / 'setup/navigation-stage2-common-20261109.json')
    if digest(source / 'model.safetensors') != common['model']['source_weights_sha256']:
        raise ValueError('Source weight hash mismatch')
    if any(version(name) != expected for name, expected in common['quantization_common']['quantization_versions'].items()):
        raise ValueError('Quantization environment changed')
"""
            code = code.replace(marker, verification + marker)
        code = code.replace('User-run '+algorithm+' lesson:', 'Isolated W8 '+algorithm+' calibration:')
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(code)
        config_path = ROOT / f'setup/navigation-{lower}-w8-20261110.json'
        with config_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(target.name, config_path.name)


if __name__ == '__main__':
    main()
