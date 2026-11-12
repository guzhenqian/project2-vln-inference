"""Repeat the same fixed-input controller with an untimed idle between requests.

LMDeploy can finish a decode scheduling block after the HTTP response ends.
The one-second idle prevents that previous request's lookahead from entering
the next request's TTFT. All models and configurations use the same idle.
This is a separate confirmation condition, not pooled with back-to-back runs.
"""
from pathlib import Path
import time

import run_navigation_w4a16_fixes as controller

_request = controller.request
_write = controller.write


def settled_request(*args, **kwargs):
    time.sleep(1.0)
    return _request(*args, **kwargs)


def annotated_write(path, value, exclusive=True):
    if Path(path).name == 'protocol.json':
        value = dict(value)
        value['request_idle_seconds'] = 1.0
        value['idle_scope'] = 'Before every request and outside its timer; avoids prior decode lookahead queueing. Analyze separately from zero-idle screening.'
        value['sources_sha256'] = dict(value['sources_sha256'])
        value['sources_sha256'][Path(__file__).name] = controller.digest(Path(__file__))
    return _write(path, value, exclusive=exclusive)


if __name__ == '__main__':
    controller.request = settled_request
    controller.write = annotated_write
    controller.main()
