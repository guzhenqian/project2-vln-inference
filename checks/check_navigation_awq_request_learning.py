"""Send one saved navigation input to the running AWQ service for learning."""
import argparse
import base64
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'SimpleVLN-Qwen3VL-4B-RL-AWQ-learning-20261104'
URL = 'http://127.0.0.1:23333/v1/chat/completions'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-index', type=int, default=0,
                        help='Zero-based sample index; 0 has two images, 1 has nine.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate input locally without sending a request.')
    args = parser.parse_args()
    samples = json.loads((ROOT / 'data/calibration/navigation_awq_v1/samples.json').read_text(encoding='utf-8'))
    if not 0 <= args.sample_index < len(samples):
        parser.error('sample-index must be between 0 and %d' % (len(samples) - 1))
    sample = samples[args.sample_index]
    messages = copy.deepcopy(sample['messages'])
    image_hashes = []
    for message in messages:
        for item in message['content']:
            if item['type'] == 'image':
                image_bytes = (ROOT / item['image']).read_bytes()
                image_hashes.append(hashlib.sha256(image_bytes).hexdigest())
                encoded = base64.b64encode(image_bytes).decode('ascii')
                item.clear()
                item.update(type='image_url', image_url={'url': 'data:image/jpeg;base64,' + encoded})
    if not image_hashes or image_hashes != sample['image_sha256']:
        raise SystemExit('Input image hashes do not match the saved sample.')
    # Preserve the original text, image order and navigation generation settings.
    settings = dict(model=MODEL, max_tokens=512, temperature=.3, top_p=.95, seed=7, stream=False)
    payload = dict(settings, messages=messages)
    print('Sample:', sample['id'])
    print('Images:', len(image_hashes))
    print('API:', URL, flush=True)
    if args.check_only:
        print('AWQ_REQUEST_INPUT_READY (no request sent)')
        return

    record = dict(purpose='single_request_functional_check_not_benchmark',
                  sample_id=sample['id'], image_sha256=image_hashes, settings=settings, url=URL)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        request = Request(URL, data=json.dumps(payload).encode('utf-8'),
                          headers={'Content-Type': 'application/json'})
        with build_opener(ProxyHandler({})).open(request, timeout=180) as response:
            record['response'] = json.load(response)
    except HTTPError as error:
        record['error'] = 'HTTP %s: %s' % (error.code, error.read().decode('utf-8', errors='replace'))
    except (URLError, TimeoutError, ValueError, OSError) as error:
        record['error'] = str(error)
    record['request_s'] = time.perf_counter() - started
    record['started_at_utc'] = started_at.isoformat()
    log = ROOT / 'logs' / ('navigation_awq_learning_request_' + started_at.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    log.parent.mkdir(exist_ok=True)
    with log.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print('Saved:', log)
    if 'error' in record:
        raise SystemExit('REQUEST_FAILED: ' + record['error'])
    answer = record['response']
    choices = answer.get('choices') or []
    choice = choices[0] if choices else {}
    reply = (choice.get('message') or {}).get('content') or ''
    print('Model reply:', reply)
    print('Finish reason:', choice.get('finish_reason'))
    print('Request seconds: %.3f' % record['request_s'])
    print('Usage:', answer.get('usage'))
    if choice.get('finish_reason') != 'stop' or not isinstance(reply, str) or not reply.strip():
        raise SystemExit('REQUEST_NOT_VERIFIED: expected a nonempty, normally finished response.')
    print('AWQ_SERVICE_REQUEST_OK')


if __name__ == '__main__':
    main()
