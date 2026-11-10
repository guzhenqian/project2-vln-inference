"""Verify that low-precision quality failures are counted, but outages cannot pass."""
from copy import deepcopy
from navigation_precision_quality import verify_step


def main():
    base = dict(model_called=True, action='MOVE_FORWARD', error=None, raw='move forward',
        finish_reason='stop', model_request_s=1., image_count=2,
        usage=dict(prompt_tokens=701, completion_tokens=3))
    assert verify_step(base) is None
    for error in ('truncated_model_output', 'invalid_action_random_fallback_rejected'):
        row = dict(base, action='STOP', error=error,
                   finish_reason='length' if error == 'truncated_model_output' else 'stop')
        assert verify_step(row) == error
    rejected = 0
    for update in (dict(error='HTTPError: 500'), dict(error='TimeoutError'), dict(model_called=None),
                   dict(model_request_s=0), dict(usage=None),
                   dict(error='truncated_model_output', action='MOVE_FORWARD'),
                   dict(error='truncated_model_output', action='STOP', finish_reason='stop'),
                   dict(model_called=False, error='invalid_action_random_fallback_rejected', action='STOP')):
        row = deepcopy(base)
        row.update(update)
        try:
            verify_step(row)
        except (AssertionError, KeyError, TypeError):
            rejected += 1
        else:
            raise AssertionError('Invalid execution accepted: '+str(update))
    assert rejected == 8
    print('PRECISION_QUALITY_BOUNDARY_PASSED valid=3 rejected=8')


if __name__ == '__main__':
    main()
