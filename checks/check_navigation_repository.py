"""Static delivery check: parse/compile source without importing it or invoking any runtime."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-tracked', action='store_true')
    parser.add_argument('--require-external', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    failures, external_missing, input_artifacts, dependencies = [], [], [], set()
    inventory = json.loads((ROOT/'setup/reproduction/source_inventory.json').read_text(encoding='utf-8'))
    paths = set(inventory['files'])
    modules = {Path(p).stem for p in paths if p.startswith('checks/') and p.endswith('.py')}
    if args.require_tracked:
        tracked = set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0'))
        failures.extend('untracked: '+p for p in sorted(paths-tracked))
    compiled = 0
    for name, expected in inventory['files'].items():
        path = ROOT/name
        if not path.is_file():
            failures.append('missing source: '+name)
            continue
        content = path.read_bytes()
        # Git's Windows checkout may use CRLF for the existing non-frozen PowerShell launcher.
        hashed_content = content.replace(b'\r\n', b'\n') if path.suffix == '.ps1' else content
        if hashlib.sha256(hashed_content).hexdigest() != expected:
            failures.append('source bytes changed: '+name)
        if path.suffix == '.py':
            try:
                tree = ast.parse(content, filename=name)
                compile(tree, name, 'exec')  # Compile only: no exec, imports, bytecode files or model code.
                compiled += 1
            except (SyntaxError, ValueError) as error:
                failures.append('syntax: '+name+': '+str(error))
                continue
            for node in ast.walk(tree):
                imports = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
                for module in imports:
                    top = module.split('.')[0]
                    if top in modules and not (ROOT/'checks'/(top+'.py')).is_file():
                        failures.append('missing local import: '+name+' -> '+top)
                    elif top:
                        dependencies.add(top)
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # Complete literal code/config paths; formatted paths are intentionally not guessed.
                    value = node.value
                    if re.fullmatch(r'(?:checks|scripts|setup)/[\w./-]+\.(?:py|sh|ps1|json|txt)', value):
                        if not (ROOT/value).is_file():
                            failures.append('missing literal dependency: '+name+' -> '+value)
    assets = json.loads((ROOT/'setup/reproduction/external_sources.json').read_text(encoding='utf-8'))['files']
    external = {row['path']:row for row in assets}
    contract = json.loads((ROOT/'setup/navigation-stage2-execution-20261109.json').read_text(encoding='utf-8'))
    for name, expected in contract['reference_files_sha256'].items():
        if name.startswith(('data/', 'models/')):
            input_artifacts.append(name)
            continue
        path = ROOT/name
        if not path.exists() and name in external:
            if external[name]['sha256'] != expected:
                failures.append('external pin disagrees with frozen contract: '+name)
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append('frozen reference mismatch: '+name)
    for row in assets:
        if row['group'] == 'habitat':
            continue
        path = ROOT/row['path']
        if not path.exists():
            external_missing.append(row['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            failures.append('external hash mismatch: '+row['path'])
    if args.require_external:
        failures.extend('missing external source: '+name for name in external_missing)
    result = dict(status='failed' if failures else 'passed', source_files=len(paths), python_files_compiled=compiled,
        failures=sorted(set(failures)), external_sources_not_restored=external_missing,
        data_and_model_artifacts_not_validated=input_artifacts,
        external_import_names=sorted(dependencies-modules),
        scope='Static source/hash/local-dependency checks only; no project imports, GPU, services, model weights, renderer or process control. Does not prove runtime/package compatibility or reproduce numerical results.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
