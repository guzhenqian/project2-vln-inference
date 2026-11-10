"""Distinguish model-quality failures from broken or missing execution."""
import math

QUALITY_ERRORS = frozenset(('invalid_action_random_fallback_rejected', 'truncated_model_output'))


def verify_step(row):
    assert type(row.get('model_called')) is bool, 'Missing real-request flag'
    error = row.get('error')
    assert error is None or error in QUALITY_ERRORS, 'Infrastructure/unclassified error: ' + str(error)
    if error:
        assert row['model_called'] and row['action'] == 'STOP', 'Quality failure must terminate the case'
        if error == 'truncated_model_output':
            assert row['finish_reason'] == 'length'
    if row['model_called']:
        assert isinstance(row.get('raw'), str)
        assert row['finish_reason'] in ('stop', 'length')
        assert math.isfinite(row['model_request_s']) and row['model_request_s'] > 0
        usage = row.get('usage')
        assert isinstance(usage, dict) and usage['prompt_tokens'] > 0 and usage['completion_tokens'] > 0
        assert row['image_count'] >= 2
    return error
