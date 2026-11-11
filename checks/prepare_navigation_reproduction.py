"""Create fresh evaluator manifests from the frozen batches without changing historical protocols."""
import argparse
import copy
import json
from pathlib import Path

from restore_navigation_sources import ROOT, inside

GROUPS = ('fp16', 'awq', 'gptq', 'fastv75', 'fastv50', 'sparsevlm75', 'sparsevlm50', 'dart75', 'awq_dart75')


def manifests(root, group, count):
    if group not in GROUPS or count not in (50, 100, 150):
        raise ValueError('Unsupported reproduction group or count')
    model = 'SimpleVLN-Qwen3VL-4B-RL-stage2-' + group + '-20261109'
    if group == 'dart75':
        model = 'SimpleVLN-Qwen3VL-4B-RL-FP16-DART-keep75-navigation-20261108'
    elif group == 'awq_dart75':
        model = 'SimpleVLN-Qwen3VL-4B-RL-AWQ-DART-keep75-navigation-20261108'
    result = []
    for index in range(1, count//50+1):
        source = root/'datasets/navigation_20261110/batches'/('batch_%02d.json' % index)
        item = copy.deepcopy(json.loads(source.read_text(encoding='utf-8')))
        item.update(model=model, version='independent_repository_reproduction', reproduction_group=group,
                    scope='New run on the original fixed development subset; never a historical result or independent test set')
        result.append(item)
    ids = [str(eid) for item in result for eid in item['episodes']]
    selection = json.loads((root/'datasets/navigation_20261110/selection.json').read_text(encoding='utf-8'))
    if ids != selection['development_pool150'][:count] or len(set(ids)) != count:
        raise ValueError('Published batch membership/order differs from the frozen selection')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--count', type=int, choices=(50, 100, 150), default=100)
    parser.add_argument('--output', required=True, help='Fresh directory under data/reproduction/')
    args = parser.parse_args()
    output = inside(ROOT, args.output)
    if not output.is_relative_to(ROOT/'data/reproduction') or output == ROOT/'data/reproduction':
        parser.error('Choose a fresh child directory under data/reproduction/')
    rows = manifests(ROOT, args.group, args.count)
    output.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows, 1):
        (output/('batch_%02d.json' % index)).write_text(json.dumps(row, indent=2)+'\n', encoding='utf-8')
    print('Prepared %d cases for %s; no model or simulator started' % (args.count, args.group))


if __name__ == '__main__':
    main()
