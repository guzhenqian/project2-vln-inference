"""User-run saved-input request to a profile-specific FP16/DART service.

--check-only validates local frozen input evidence and JSON without HTTP calls.
A real run checks /v1/models first, then sends at most one fresh chat request.
It does not start a service, load model weights, or evaluate navigation success.
"""

import argparse
import base64
import copy
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from navigation_dart50_profiles import profile_config, served_model_name


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 'data/calibration/navigation_awq_v1/samples.json'
SAMPLES_SHA256 = 'e5386370a0997453d8b27211b66f22897a2e31ad2e874d5f1e5cde0782ae87cc'
SAMPLE_ID = 'train_515_step_0'
SAMPLE_SPECS = {
    0: {'id': SAMPLE_ID, 'images': 2, 'prompt_tokens': 701},
    1: {'id': 'train_515_step_14', 'images': 9, 'prompt_tokens': 2815},
}
MODELS_URL = 'http://127.0.0.1:23333/v1/models'
CHAT_URL = 'http://127.0.0.1:23333/v1/chat/completions'


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def reject_nonfinite_json(value):
    raise ValueError('Nonfinite value in HTTP JSON: ' + value)


def model_name(keep_ratio, profile='short'):
    return served_model_name(keep_ratio, profile)


def prepare_input(keep_ratio, sample_index=0, profile='short'):
    expected_model = model_name(keep_ratio, profile)
    limits = profile_config(profile)
    if sample_index not in SAMPLE_SPECS:
        raise ValueError('Only frozen sample-index 0 (two images) or 1 (nine images) is supported')
    if profile == 'short' and sample_index != 0:
        raise ValueError('The short profile only supports sample-index 0; use navigation explicitly for nine images')
    spec = SAMPLE_SPECS[sample_index]
    if (not limits['min_images'] <= spec['images'] <= limits['max_images']
            or spec['prompt_tokens'] > limits['max_prompt_tokens']
            or spec['prompt_tokens'] + 512 > limits['session_len']):
        raise ValueError('Saved sample exceeds the selected profile limits')
    source_bytes = (ROOT / SAMPLES).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SAMPLES_SHA256:
        raise ValueError('Frozen source samples.json SHA256 mismatch')
    samples = json.loads(source_bytes)
    if not isinstance(samples, list) or len(samples) <= sample_index:
        raise ValueError('Saved sample is missing from the frozen JSON list')
    sample = samples[sample_index]
    if sample.get('id') != spec['id']:
        raise ValueError('Saved sample ID differs from the frozen teaching input')
    messages = copy.deepcopy(sample['messages'])
    image_hashes = []
    image_paths = []
    for message in messages:
        for item in message['content']:
            if item['type'] == 'image':
                relative = item['image']
                path = (ROOT / relative).resolve()
                if not path.is_relative_to(ROOT) or path.suffix.lower() not in ('.jpg', '.jpeg'):
                    raise ValueError('Saved image must be a JPEG inside the project')
                image_bytes = path.read_bytes()
                image_hashes.append(hashlib.sha256(image_bytes).hexdigest())
                image_paths.append(relative)
                encoded = base64.b64encode(image_bytes).decode('ascii')
                item.clear()
                item.update(type='image_url', image_url={'url': 'data:image/jpeg;base64,' + encoded})
    if len(image_hashes) != spec['images'] or image_hashes != sample['image_sha256']:
        raise ValueError('The input image count or hashes do not match the frozen sample')
    settings = dict(model=expected_model, max_tokens=512,
                    temperature=0.3, top_p=0.95, seed=7, stream=False)
    # Preserve the original text and image order. Do not add session/history IDs.
    payload = dict(settings, messages=messages)
    encoded_payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if json.loads(encoded_payload) != payload:
        raise ValueError('Request JSON roundtrip failed')
    return sample, settings, image_paths, image_hashes, encoded_payload


def validate_model_listing(listing, expected):
    if not isinstance(listing, dict) or not isinstance(listing.get('data'), list):
        raise ValueError('Malformed /v1/models response')
    entries = listing['data']
    if len(entries) != 1 or not isinstance(entries[0], dict) or entries[0].get('id') != expected:
        actual = [item.get('id') if isinstance(item, dict) else None for item in entries]
        raise ValueError(f'Expected exactly one served model {expected!r}; got {actual!r}. No chat request sent.')
    return entries[0]['id']


def latest_launch_hint(expected, keep_ratio, profile='short'):
    """Auxiliary provenance only: a saved launch file may describe an old process."""
    paths = sorted((ROOT / 'logs').glob('navigation_dart50_service_*/launch.json'), reverse=True)
    for path in paths:
        try:
            raw = path.read_bytes()
            launch = json.loads(raw)
            if (launch.get('model_name') != expected or launch.get('keep_ratio') != keep_ratio
                    or launch.get('profile', 'short') != profile):
                continue
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        return {
            'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'model_name': launch['model_name'],
            'keep_ratio': launch['keep_ratio'],
            'profile': launch.get('profile', 'short'),
            'server': launch.get('server'),
            'model_source': launch.get('model_source'),
            'sources_sha256': launch.get('sources_sha256'),
            'identity_evidence': False,
            'note': 'May be stale; not proof that this process is running or that DART executed.',
        }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep-ratio', type=float, choices=(1.0, 0.5), default=0.5)
    parser.add_argument('--profile', choices=('short', 'navigation'), default='navigation',
                        help='short: two images only; navigation: experimental full-prefill 2-9-image service')
    parser.add_argument('--sample-index', type=int, choices=(0, 1), default=0,
                        help='Sample 0 has two images; sample 1 has nine and requires --profile navigation.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate frozen images and request JSON locally, without HTTP or output writes.')
    parser.add_argument('--run', action='store_true', help='Explicitly send one request; default is offline')
    args = parser.parse_args()
    if args.run and args.check_only:
        parser.error('--run and --check-only conflict')
    sample, settings, image_paths, image_hashes, payload = prepare_input(
        args.keep_ratio, args.sample_index, args.profile)
    limits = profile_config(args.profile)
    expected = settings['model']
    print('Sample:', sample['id'])
    print('Images:', len(image_hashes))
    print('Profile:', args.profile)
    print('Frozen preprocessing prompt tokens:', SAMPLE_SPECS[args.sample_index]['prompt_tokens'])
    print('Expected served model:', expected)
    print('API:', CHAT_URL, flush=True)
    if args.profile == 'navigation':
        print('Experimental full-prefill profile; GPU memory and formal navigation evaluation remain unverified', flush=True)
    if not args.run:
        print('DART_REQUEST_INPUT_READY (CPU input/JSON checks only; no HTTP request or output write)')
        return

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    log = ROOT / 'logs' / ('navigation_dart50_request_' + args.profile + '_' + run_id + '.json')
    log.parent.mkdir(exist_ok=True)
    record = {
        'purpose': 'single_request_functional_check_not_navigation_evaluation_or_benchmark',
        'started_at_utc': started_at.isoformat(),
        'sample_index': args.sample_index, 'sample_id': sample['id'],
        'profile': args.profile, 'profile_limits': limits,
        'expected_prompt_tokens_from_frozen_preprocessing': SAMPLE_SPECS[args.sample_index]['prompt_tokens'],
        'source_samples': SAMPLES, 'source_samples_sha256': SAMPLES_SHA256,
        'source_messages': sample['messages'],
        'image_paths': image_paths, 'image_sha256': image_hashes,
        'keep_ratio': args.keep_ratio, 'settings': settings,
        'expected_model_id': expected, 'served_model_id': None,
        'served_model_ids': None, 'response_model_id': None,
        'models_url': MODELS_URL, 'request_url': CHAT_URL,
        'request_sha256': hashlib.sha256(payload).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'session_policy': 'Fresh independent request; no session_id or history-continuation parameters',
        'chat_request_attempted': False, 'model_check_s': None,
        'request_started_at_utc': None, 'request_s': None,
        'reply': None, 'finish_reason': None, 'usage': None, 'error': None,
        'status': 'checking_service_identity',
        'latest_matching_launch_hint': latest_launch_hint(expected, args.keep_ratio, args.profile),
        'runtime_activation_verified': False,
        'limitations': [
            'The HTTP model identifier is a service identity check, not proof of runtime token pruning',
            'Runtime DART counters and model/cache behavior require separate log verification',
            'finish_reason=stop means normal text completion, not a navigation STOP action or success',
            'One request latency is not a benchmark or an equal-input acceleration result',
        ],
    }
    opener = build_opener(ProxyHandler({}), NoRedirectHandler())
    phase = 'model_identity'
    # Reserve a unique output before HTTP; never overwrite an older request log.
    with log.open('x', encoding='utf-8') as output_stream:
        try:
            check_started = time.perf_counter()
            try:
                with opener.open(Request(MODELS_URL, method='GET'), timeout=15) as response:
                    listing = json.load(response, parse_constant=reject_nonfinite_json)
                record['model_listing'] = listing
                if isinstance(listing, dict) and isinstance(listing.get('data'), list):
                    actual_ids = [item.get('id') if isinstance(item, dict) else None
                                  for item in listing['data']]
                    record['served_model_ids'] = actual_ids
                    if len(actual_ids) == 1:
                        record['served_model_id'] = actual_ids[0]
                record['served_model_id'] = validate_model_listing(listing, expected)
            finally:
                record['model_check_s'] = time.perf_counter() - check_started

            phase = 'chat_request'
            record['status'] = 'requesting'
            request = Request(CHAT_URL, data=payload, method='POST',
                              headers={'Content-Type': 'application/json'})
            record['request_started_at_utc'] = datetime.now(timezone.utc).isoformat()
            request_started = time.perf_counter()
            record['chat_request_attempted'] = True
            try:
                # Exactly one POST attempt, with no application retry or redirect.
                with opener.open(request, timeout=180) as response:
                    answer = json.load(response, parse_constant=reject_nonfinite_json)
            finally:
                record['request_s'] = time.perf_counter() - request_started
            record['response'] = answer
            phase = 'response_validation'
            if not isinstance(answer, dict):
                raise ValueError('Chat response must be a JSON object')
            record['response_model_id'] = answer.get('model')
            record['usage'] = answer.get('usage')
            if answer.get('model') != expected:
                raise ValueError('Chat response model ID differs from the expected FP16/DART service')
            choices = answer.get('choices')
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise ValueError('Expected exactly one completion choice')
            choice = choices[0]
            message = choice.get('message')
            record['reply'] = message.get('content') if isinstance(message, dict) else None
            record['finish_reason'] = choice.get('finish_reason')
            if (record['finish_reason'] != 'stop' or not isinstance(record['reply'], str)
                    or not record['reply'].strip()):
                raise ValueError('Expected a nonempty, normally finished text response')
            record['status'] = 'functional_request_verified'
        except HTTPError as error:
            body = error.read(65536).decode('utf-8', errors='replace')
            record.update(status='failed', error=f'HTTP {error.code}: {body}', error_phase=phase)
        except (URLError, TimeoutError, ValueError, OSError, TypeError) as error:
            record.update(status='failed', error=str(error), error_phase=phase)
        except KeyboardInterrupt:
            record.update(status='interrupted', error='Interrupted by the user', error_phase=phase)
        finally:
            record['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
            json.dump(record, output_stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('Saved:', log, flush=True)
    if record['error'] is not None:
        raise SystemExit('DART_REQUEST_FAILED: ' + record['error'])
    print('Served model:', record['served_model_id'])
    print('Model reply:', record['reply'])
    print('Finish reason:', record['finish_reason'])
    print('Request seconds: %.3f' % record['request_s'])
    print('Usage:', record['usage'])
    print('DART_SERVICE_REQUEST_OK')
    print('Single-request connectivity verified; runtime pruning and navigation success remain separate checks.')


if __name__ == '__main__':
    main()
