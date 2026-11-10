"""Run two owned W8 calibrations serially; preserve output and stop on failure."""
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path.home()/'.venvs/project2-quant312/bin/python')
RUN = ROOT/'logs/navigation_precision_w8_sequence_20261110'


def main():
    RUN.mkdir(exist_ok=False)
    for algorithm in ('awq', 'gptq'):
        state = dict(algorithm=algorithm, status='calibrating', utc=datetime.now(timezone.utc).isoformat())
        (RUN/'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        with (RUN/(algorithm+'.log')).open('x', encoding='utf-8') as stream:
            code = subprocess.call([PY, '-B', '-u', f'checks/quantize_navigation_{algorithm}_w8_20261110.py',
                                    '--config', f'setup/navigation-{algorithm}-w8-20261110.json'],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if code:
            state.update(status='failed', returncode=code)
            (RUN/'status.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
            raise RuntimeError(str(state))
        print('W8_CALIBRATION_COMPLETE', algorithm, flush=True)
    (RUN/'status.json').write_text(json.dumps(dict(status='complete',
             utc=datetime.now(timezone.utc).isoformat()), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
