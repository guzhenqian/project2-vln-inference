"""Make independent service/controller copies, preserving the frozen protocol."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    service = (ROOT/'checks/serve_navigation_stage2.py').read_text(encoding='utf-8')
    service = service.replace('from navigation_stage2_common import', 'from navigation_precision_common import')
    service = service.replace('navigation_stage2_module_map.py', 'navigation_precision_module_map.py')
    service = service.replace("    source = ROOT / group['model']", "    source = ROOT / group['model']\n    kernel_gate = read(ROOT/'reports/navigation_precision_kernel_gate_linked_20261110.json')\n    assert kernel_gate['status'] == 'passed'\n    assert kernel_gate['kernel_source_sha256'] == digest(ROOT/'checks/navigation_precision_kernels.py')")
    marker = "    from lmdeploy import PytorchEngineConfig"
    service = service.replace(marker, "    assert digest(source/'model.safetensors') == audit['sha256'], 'Weight hash mismatch'\n"+marker)
    service = service.replace("    os.environ['PYTHONPATH']", "    os.environ['PROJECT2_PRECISION_WBITS'] = str(group['weight_bits'])\n    os.environ['PROJECT2_PRECISION_ABITS'] = str(group['activation_bits'])\n    os.environ['PYTHONPATH']")
    service = service.replace('group=args.group, recipe=group,', "precision_config_sha256=digest(ROOT/'setup/navigation-precision-20261110.json'),\n        group=args.group, recipe=group,")
    service = service.replace("instrumentation='Detailed runtime traces and original HTTP measurement protocol'", "instrumentation='Original detailed traces plus actual precision-kernel receipts; included in latency'")
    service = service.replace("        model_config_sha256=digest(source / 'config.json'),", "        model_config_sha256=digest(source / 'config.json'),\n        model_weights_sha256=audit['sha256'], model_bytes=audit['bytes'],\n        conversion_audit_sha256=digest(source/'conversion_audit.json'),")
    controller = (ROOT/'checks/run_navigation_stage2_validation.py').read_text(encoding='utf-8')
    controller = controller.replace('from navigation_stage2_common import', 'from navigation_precision_common import')
    controller = controller.replace('from audit_navigation_stage2 import', 'from audit_navigation_precision import')
    controller = controller.replace('serve_navigation_stage2.py', 'serve_navigation_precision.py')
    controller = controller.replace("dtype='W4A16' if GROUPS[group]['method'] in ('awq','gptq') else 'float16'", "dtype='W%dA%d' % (GROUPS[group]['weight_bits'], GROUPS[group]['activation_bits'])")
    controller = controller.replace('navigation_stage2_20261109', 'navigation_precision_20261110')
    controller = controller.replace("version='navigation_precision_20261110', model=model_name(group)", "version='navigation_precision_20261110', model=model_name(group)")
    controller = controller.replace("timeout=3600 if single else 21600", "timeout=7200 if single else 43200")
    controller = controller.replace("deadline = time.monotonic()+900", "deadline = time.monotonic()+1800")
    for name, text in [('serve_navigation_precision.py',service),('run_navigation_precision_validation.py',controller)]:
        with (ROOT/'checks'/name).open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)


if __name__ == '__main__':
    main()
