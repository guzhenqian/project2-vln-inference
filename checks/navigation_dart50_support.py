"""C50 local file persistence and C75 completion gate; no model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
C75_MODEL = 'SimpleVLN-Qwen3VL-4B-RL-FP16-DART-keep75-navigation-20261108'


def read(path):
    for attempt in range(20):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except (PermissionError, json.JSONDecodeError):
            if attempt == 19:
                raise
            time.sleep(.1)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value, *, exclusive=False):
    """Retry transient Windows sharing violations, without rerunning a task."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
        return
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(encoded, encoding='utf-8')
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise  # Keep temporary evidence; never discard data on failure.
            time.sleep(.1)


def require_c75_complete(report_path):
    if report_path is None:
        raise ValueError('A completed C75 report is required before C50 model loading')
    path = Path(report_path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Choose the actual completed C75 report inside this project')
    report = read(path)
    if not (report.get('status') == 'complete' and report.get('completed_episode_count') == 150
            and report.get('all_runtime_audits_passed') is True and report.get('error_count') == 0):
        raise ValueError('C75 has not completed all150 tasks and all runtime audits')
    model = report.get('model', report.get('controller_state', {}).get('model'))
    if model != C75_MODEL:
        raise ValueError('Completed report is not FP16 C75')
    expected = {str(eid) for index in range(1, 4) for eid in
                read(ROOT / f'data/baseline/overnight200_fp16_20261031/batch_{index:02d}.json')['episodes']}
    actual = [str(row['episode_id']) for row in report['episodes']]
    if len(actual) != 150 or set(actual) != expected:
        raise ValueError('C75 report does not contain exactly the original150 episodes')
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
            'checked_utc': datetime.now(timezone.utc).isoformat()}
