"""Real-GPU execution acceptance, retaining low-precision quality failures."""
import argparse
import json
import os
import socket
from pathlib import Path
import subprocess
import time
import traceback
from urllib.request import Request, ProxyHandler, build_opener

from navigation_precision_common import ROOT, RUN, PLAN, GROUPS, SOURCES, read, write, digest, model_name
from audit_navigation_precision import audit_service, audit_trace, audit_batch
from run_navigation_dart50_night import stop_owned, now, processes

PYTHON = str(Path.home() / '.venvs/project2/bin/python')
OPENER = build_opener(ProxyHandler({}))


def status(**values):
    data = read(RUN / 'status.json')
    data.update(values, updated_utc=now())
    write(RUN / 'status.json', data, exclusive=False)


def make_manifest(group, batch, path, single=False):
    manifest = read(PLAN / f'source_batch_{batch:02d}.json')
    if single:
        # The same preselected original first episode; quality does not decide acceptance.
        index = manifest['episodes'].index('1250')
        manifest['episodes'] = ['1250']
        manifest['trajectory_ids'] = [manifest['trajectory_ids'][index]]
    manifest.update(version='navigation_precision_20261110', model=model_name(group),
        scope='stage2 integration single1250' if single else f'stage2 development batch{batch}: 50 fixed cases',
        dtype='W%dA%d' % (GROUPS[group]['weight_bits'], GROUPS[group]['activation_bits']),
        backend='LMDeploy0.11.1 PyTorch eager full-prefill3072 fixed64x64 KV',
        stage2_group=group, stage2_recipe=GROUPS[group],
        differences_from_author_benchmark='Same original navigation policy; stage2 inference recipe',
        execution_gate='Authorized after per-method runtime validation' if not single else 'Authorized integration validation')
    write(path, manifest)
    return manifest


class OwnedSession:
    def __init__(self, group, directory):
        self.group, self.directory = group, directory
        self.child = self.service_child = None
        self.service = directory / 'service'
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LIBGL_ALWAYS_SOFTWARE='1')
        self.env.pop('DISPLAY', None)
        self.env.pop('CUDA_VISIBLE_DEVICES', None)

    def command(self, argv, label, timeout=10800):
        with (self.directory / (label + '.log')).open('x', encoding='utf-8') as stream:
            self.child = subprocess.Popen(argv, cwd=ROOT, env=self.env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        status(active_group=self.group, active_step=label, active_child_pid=self.child.pid)
        deadline = time.monotonic()+timeout
        while self.child.poll() is None:
            if self.service_child is not None and self.service_child.poll() is not None:
                raise RuntimeError('Owned model service exited during ' + label)
            if time.monotonic() > deadline:
                raise TimeoutError(label)
            time.sleep(3)
        code, self.child = self.child.returncode, None
        if code:
            raise RuntimeError(f'{label} exited {code}; retained logs')

    def start(self):
        # Match the service's conservative bind probe. Closed HTTP connections
        # may leave TIME_WAIT even after our old model process has exited.
        status(active_group=self.group, active_step='waiting_for_released_port')
        deadline = time.monotonic()+120
        while True:
            try:
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1',23333))
                break
            except OSError:
                if time.monotonic()>deadline:
                    raise TimeoutError('Port 23333 did not become free; no unrelated process will be stopped')
                time.sleep(2)
        with (self.directory / 'service_console.log').open('x', encoding='utf-8') as stream:
            self.service_child = subprocess.Popen([PYTHON, '-B', '-u', 'checks/serve_navigation_precision.py',
                '--group', self.group, '--run-dir', str(self.service)], cwd=ROOT, env=self.env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status(active_group=self.group, active_step='model_loading', service_pid=self.service_child.pid)
        deadline = time.monotonic()+1800
        while True:
            if self.service_child.poll() is not None:
                raise RuntimeError('Service exited before readiness: ' + self.group)
            try:
                with OPENER.open('http://127.0.0.1:23333/v1/models', timeout=3) as response:
                    listing = json.load(response)
                if [row['id'] for row in listing['data']] != [model_name(self.group)]:
                    raise ValueError('Unexpected service identity')
                break
            except (OSError, TimeoutError):
                if time.monotonic() > deadline:
                    raise TimeoutError('Service startup')
                time.sleep(3)
        audit_service(self.service, self.group)
        write(self.directory/'ownership.json', {'service_pid':self.service_child.pid,
            'snapshot': processes().get(self.service_child.pid), 'created_utc':now()})

    def smoke(self, sample_index):
        # Reuse the original frozen messages/images; replace only the served model ID.
        from check_navigation_awq_dart_request import prepare_input
        sample, settings, paths, hashes, encoded = prepare_input(1., sample_index, 'navigation')
        payload = json.loads(encoded)
        payload['model'] = model_name(self.group)
        before = {path.name for path in self.service.glob('prefill_*.json')}
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        started = time.perf_counter()
        request = Request('http://127.0.0.1:23333/v1/chat/completions', data=encoded,
            headers={'Content-Type':'application/json'}, method='POST')
        with OPENER.open(request, timeout=300) as response:
            answer = json.load(response)
        record = dict(sample_id=sample['id'], sample_index=sample_index, images_sha256=hashes,
            response=answer, request_s=time.perf_counter()-started, request_sha256=__import__('hashlib').sha256(encoded).hexdigest())
        write(self.directory / f'smoke{sample_index}.json', record)
        assert answer['model'] == model_name(self.group)
        assert len(answer['choices']) == 1 and answer['choices'][0]['finish_reason'] in ('stop', 'length')
        assert answer['choices'][0]['message']['content'].strip()
        fresh = [path for path in self.service.glob('prefill_*.json') if path.name not in before]
        assert len(fresh) == 1
        trace = audit_trace(fresh[0], self.group)
        assert (trace['image_count'], trace['original_length']) == ((2,701) if sample_index == 0 else (9,2815))
        trace['finish_reason'] = answer['choices'][0]['finish_reason']
        trace['truncated_model_output'] = trace['finish_reason'] == 'length'
        trace['scope'] = 'Technical execution gate; output quality is retained, not an acceptance criterion'
        write(self.directory / f'smoke{sample_index}_audit.json', trace)

    def episode_batch(self, batch, single=False):
        label = 'single1250' if single else f'batch_{batch:02d}'
        folder = self.directory / label
        folder.mkdir()
        manifest = self.directory / (label + '_manifest.json')
        make_manifest(self.group, batch, manifest, single)
        before = [path.name for path in self.service.glob('prefill_*.json')]
        write(folder/'runtime_before.json', before)
        self.command(['bash','scripts/run_habitat_task_python.sh','-u','checks/run_navigation_precision_diagnostic.py',
            '--policy','memory','--manifest',str(manifest),'--output',str(folder)], label,
            timeout=7200 if single else 43200)
        audit = audit_batch(folder, self.service, self.group, before)
        write(folder/'runtime_audit.json', audit)
        return folder

    def close(self):
        stop_owned(self.child)
        stop_owned(self.service_child)
        self.child = self.service_child = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args()
    validation = RUN / 'validation'
    validation.mkdir(exist_ok=True)
    passed = []
    for group in args.groups:
        success = validation / (group+'_passed.json')
        if success.exists():
            old = read(success)
            audit_service(ROOT/old['service'], group)
            passed.append(group)
            continue
        directory = validation / (group+'_'+time.strftime('%Y%m%dT%H%M%S'))
        directory.mkdir()
        owner = OwnedSession(group, directory)
        try:
            status(status='validating', active_group=group, formal_started=False, failure=None)
            owner.start()
            for sample in (0,1):
                status(active_step='smoke'+str(sample))
                owner.smoke(sample)
            folder = owner.episode_batch(1, single=True)
            write(success, dict(passed=True, group=group, validated_utc=now(),
                service=owner.service.relative_to(ROOT).as_posix(),
                single=folder.relative_to(ROOT).as_posix(),
                single_audit_sha256=digest(folder/'runtime_audit.json'),
                source_hashes=read(owner.service/'launch.json')['sources_sha256']))
            passed.append(group)
            status(validated_groups=passed)
            print('STAGE2_VALIDATED', group, flush=True)
        except BaseException as error:
            failure = dict(group=group, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(), utc=now())
            write(directory/'failure.json', failure)
            status(status='validation_failed', failure=failure, validated_groups=passed)
            raise
        finally:
            owner.close()
    status(status='validated_ready', validated_groups=passed, formal_started=False)


if __name__ == '__main__':
    main()
