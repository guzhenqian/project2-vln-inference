"""Offline unit checks for recovery evidence boundaries; no model/HTTP/GPU."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_navigation_dart_recovery as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def row(eid, step, called, images=2, tokens=701):
    return dict(episode_id=eid, step=step, error=None, model_called=called,
                finish_reason='stop' if called else None,
                usage={'prompt_tokens': tokens} if called else None,
                image_count=images if called else 0, action='STOP',
                raw='stop' if called else None, model_request_s=1.0 if called else 0.0)


def outcome(eid, steps, prefix=None):
    result = dict(episode_id=eid, trajectory_id='trajectory_' + eid, steps=steps,
                  metrics={'success': 1.0, 'spl': 1.0, 'distance_to_goal': 1.0},
                  error_count=0, terminal_stop_source='model', elapsed_s=3.0,
                  start_distance_m=2.0, path_length_m=1.0)
    if prefix is not None:
        result.update(prefix_reused_steps=prefix,
                      timing_note='Completed original episode retained' if prefix == steps
                      else 'Interrupted episode lower-bound loop time' if prefix else '')
    return result


class RecoveryFixture:
    def __init__(self, root):
        self.root = root
        self.old, self.new, self.service = [root / name for name in ('old', 'new', 'service')]
        self.manifest = {'model': 'test-model', 'episodes': ['C', 'A', 'B']}
        for run in (self.old, self.new):
            write(run / 'manifest.json', self.manifest)
        write(self.service / 'launch.json', dict(model_name='test-model', profile='navigation',
              keep_ratio=.75, weight_quantization=None,
              engine_config=dict(dtype='float16', quant_policy=0, eager_mode=True, max_batch_size=1)))
        self.old_rows = {'A': [row('A', 0, True), row('A', 1, False)],
                         'B': [row('B', 0, True, tokens=711), row('B', 1, False)]}
        for eid, rows in self.old_rows.items():
            write_rows(self.old / ('episode_' + eid) / 'steps.jsonl', rows)
        write(self.old / 'episode_A/result.json', outcome('A', 2))
        self.rows = {eid: [dict(item, prefix_reused=True) for item in rows]
                     for eid, rows in self.old_rows.items()}
        # Continuation's first call differs from A's warmup prompt in both image
        # count and token count; accepting its signature for warmup is a bug.
        self.rows['B'].append(row('B', 2, True, images=7, tokens=2221))
        self.rows['C'] = [row('C', 0, True, tokens=721)]
        self.summary = dict(status='complete', error_count=0, model='test-model', episode_count=3,
                            warmup_requests_excluded=2,
                            episodes=[outcome('A', 2, 2), outcome('B', 3, 2), outcome('C', 1, 0)])
        self.save()
        self.trace_order = []
        self.add_trace('prior', 2, 701)
        write(self.old / 'runtime_before.json', ['prefill_prior.json'])
        for name, images, tokens in [('old_w1', 2, 701), ('old_w2', 2, 701),
                                    ('old_a', 2, 701), ('old_b', 2, 711)]:
            self.add_trace(name, images, tokens)
        self.before = ['prefill_' + name + '.json' for name in self.trace_order]
        for name, images, tokens in [('new_w1', 2, 701), ('new_w2', 2, 701),
                                    ('new_b', 7, 2221), ('new_c', 2, 721)]:
            self.add_trace(name, images, tokens)

    def save(self):
        for eid, rows in self.rows.items():
            write_rows(self.new / ('episode_' + eid) / 'steps.jsonl', rows)
        write(self.new / 'summary.json', self.summary)

    def add_trace(self, name, images, tokens):
        self.trace_order.append(name)
        trace = dict(trace_id=name, image_count=images, original_length=tokens,
                     retained_length=tokens - images * 75, original_image_tokens=images * 300,
                     retained_image_tokens=images * 225)
        path = self.service / ('prefill_' + name + '.json')
        write(path, trace)
        write(self.service / ('first_decode_' + name + '.json'), {'trace_id': name})
        timestamp_ns = (1_700_000_000 + len(self.trace_order)) * 1_000_000_000
        os.utime(path, ns=(timestamp_ns, timestamp_ns))

    def audit(self):
        # The lower-level numerical token/KV/mRoPE checks have their own suite;
        # these tests retain real inventories/files and mock only their payload.
        with patch.object(audit, 'audit_trace', side_effect=lambda path: audit.read(path)):
            return audit.audit_resumed_batch(self.new, self.old, self.service, self.before)


class RecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dart_recovery_unit_')
        self.fixture = RecoveryFixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def assert_rejected(self, expression=None):
        with self.assertRaises((ValueError, FileNotFoundError)) as error:
            self.fixture.audit()
        if expression:
            self.assertIn(expression, str(error.exception))

    def test_old_new_inventories_and_four_warmups(self):
        result = self.fixture.audit()
        self.assertEqual(result['actual_model_calls'], 4)
        self.assertEqual(result['prefix_model_calls'], 2)
        self.assertEqual(result['continuation_model_calls'], 2)
        self.assertEqual(result['old_runtime_trace_count'], 4)
        self.assertEqual(result['new_runtime_trace_count'], 4)
        self.assertEqual(result['runtime_trace_count'], 8)
        self.assertEqual(result['old_warmup_requests'], 2)
        self.assertEqual(result['new_warmup_requests'], 2)
        self.assertEqual(result['warmup_requests'], 4)
        self.assertEqual(result['prefix_reused_steps'], 4)
        self.assertEqual(result['completed_prefix_episodes'], 1)
        self.assertEqual(result['interrupted_episode_ids'], ['B'])
        self.assertEqual([trace['trace_id'] for trace in result['traces']],
                         self.fixture.trace_order[1:])
        self.assertNotIn('prior', [trace['trace_id'] for trace in result['traces']])

    def test_changed_old_response_rejected(self):
        self.fixture.rows['B'][0]['raw'] = 'different response'
        self.fixture.save()
        self.assert_rejected('Previously saved prefix rows changed')

    def test_missing_reused_marker_rejected(self):
        self.fixture.rows['A'][0].pop('prefix_reused')
        self.fixture.save()
        self.assert_rejected('Previously saved prefix rows changed')

    def test_suffix_cannot_be_falsely_marked_reused(self):
        self.fixture.rows['B'][-1]['prefix_reused'] = True
        self.fixture.save()
        self.assert_rejected('New steps falsely marked reused')

    def test_duplicated_prefix_row_rejected(self):
        self.fixture.rows['A'].insert(1, copy.deepcopy(self.fixture.rows['A'][0]))
        self.fixture.summary['episodes'][0]['steps'] = 3
        self.fixture.save()
        self.assert_rejected('Incomplete/nonsequential')

    def test_completed_old_case_cannot_receive_new_steps(self):
        self.fixture.rows['A'].append(row('A', 2, True))
        self.fixture.summary['episodes'][0]['steps'] = 3
        self.fixture.save()
        self.assert_rejected('Completed old episode received new steps')

    def test_completed_old_elapsed_cannot_change(self):
        self.fixture.summary['episodes'][0]['elapsed_s'] = 99.0
        self.fixture.save()
        self.assert_rejected('Completed prefix result changed: A/elapsed_s')

    def test_missing_old_prefill_rejected(self):
        (self.fixture.service / 'prefill_old_b.json').unlink()
        self.assert_rejected('Runtime trace inventory disappeared')

    def test_missing_new_prefill_rejected(self):
        (self.fixture.service / 'prefill_new_b.json').unlink()
        self.assert_rejected()

    def test_extra_new_prefill_rejected(self):
        self.fixture.add_trace('extra', 2, 721)
        self.assert_rejected('Segment trace count')

    def test_inventory_boundary_must_not_shift_one_old_call_to_new(self):
        self.fixture.before.remove('prefill_old_b.json')
        self.assert_rejected('Segment trace count')

    def test_new_warmups_use_first_episode_not_first_continuation(self):
        self.assertTrue(self.fixture.audit()['passed'])
        for name in ('new_w1', 'new_w2'):
            path = self.fixture.service / ('prefill_' + name + '.json')
            trace = audit.read(path)
            trace.update(image_count=7, original_length=2221)
            old_stat = path.stat()
            write(path, trace)
            os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        self.assert_rejected('Segment chronological input signatures')

    def test_new_summary_must_count_two_not_four_warmups(self):
        self.fixture.summary['warmup_requests_excluded'] = 4
        self.fixture.save()
        self.assert_rejected('two new warmups')

    def test_partial_prefix_needs_lower_bound_note(self):
        self.fixture.summary['episodes'][1]['timing_note'] = 'exact uninterrupted elapsed'
        self.fixture.save()
        self.assert_rejected('Missing interruption timing caveat')

    def test_new_row_episode_identity_must_match_folder(self):
        self.fixture.rows['B'][-1]['episode_id'] = 'C'
        self.fixture.save()
        self.assert_rejected()

    def test_numeric_reused_marker_is_not_boolean_true(self):
        self.fixture.rows['A'][0]['prefix_reused'] = 1
        self.fixture.save()
        self.assert_rejected('Prefix reuse markers must be boolean true')

    def test_missing_decode_rejected_despite_numerical_trace_mock(self):
        (self.fixture.service / 'first_decode_new_b.json').unlink()
        self.assert_rejected()

    def test_orphan_decode_rejected(self):
        write(self.fixture.service / 'first_decode_orphan.json', {'trace_id': 'orphan'})
        self.assert_rejected()


if __name__ == '__main__':
    unittest.main(verbosity=2)
