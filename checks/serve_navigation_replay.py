"""Read-only, loopback-only viewer for a single navigation run. No model calls."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_episode(run, eid):
    folder = run / ('episode_' + eid)
    rows = []
    try:
        lines = (folder / 'steps.jsonl').read_text(encoding='utf-8').splitlines()
    except FileNotFoundError:
        lines = []
    last_call = None
    for line in lines:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            break  # writer may still be appending the last line
        if r.get('model_called', True):
            last_call = r
        rows.append(dict(step=r['step'], action=r['action'], model_called=r.get('model_called', True),
                         raw=(last_call or r).get('raw'), call_step=(last_call or r)['step'],
                         request_s=r.get('model_request_s'), error=r.get('error'),
                         distance=r['metrics']['distance_to_goal'],
                         collision=r['metrics'].get('collisions', {}).get('is_collision', False),
                         before=r['before']['position'], after=r['after']['position']))
    result = read_json(folder / 'result.json')
    reason = '等待任务开始' if not rows else '运行中，尚无最终结果'
    status = 'pending' if not rows else 'running'
    if result:
        status = 'success' if result['metrics']['success'] else 'failed'
        distance = result['metrics']['distance_to_goal']
        if result['error_count']:
            reason = '出现请求或动作解析错误；需先检查错误记录'
        elif status == 'success':
            reason = '模型主动STOP，最终距离 {:.2f}m，小于3m'.format(distance)
        elif result['stop_called']:
            reason = '模型已STOP，但最终距离 {:.2f}m，不满足小于3m'.format(distance)
        else:
            reason = '达到{}步上限，未作出有效STOP'.format(result['steps'])
        if status == 'failed' and rows and min(r['distance'] for r in rows) < 3:
            reason += '；曾进入3m范围，但没有在范围内有效停止'
    instruction = ''
    if lines:
        try:
            texts = json.loads(lines[0]).get('request_text', [])
            text = ' '.join(texts)
            match = re.search(r"Your assigned task is: '(.*?)'\. Analyze", text, re.S)
            instruction = match.group(1) if match else text
        except (ValueError, TypeError):
            pass
    return dict(id=eid, status=status, reason=reason, result=result, rows=rows,
                instruction=instruction, final_available=(folder/'final.png').exists())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--port', type=int, default=8766)
    args = p.parse_args()
    run = args.run.resolve()
    run.relative_to((ROOT/'logs').resolve())
    manifest = read_json(run/'manifest.json')
    assert manifest and all(re.fullmatch(r'\d+', str(e)) for e in manifest['episodes'])
    ids = [str(e) for e in manifest['episodes']]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/':
                self.send_data((ROOT/'checks/navigation_replay.html').read_bytes(), 'text/html; charset=utf-8')
            elif url.path == '/api/run':
                data = dict(run=run.name, model=manifest['model'], max_steps=manifest['max_episode_steps'],
                            summary=read_json(run/'summary.json'), episodes=[load_episode(run,e) for e in ids])
                self.send_data(json.dumps(data, ensure_ascii=False).encode(), 'application/json; charset=utf-8')
            elif url.path == '/frame':
                params = parse_qs(url.query)
                eid = params.get('episode', [''])[0]; step = params.get('step', [''])[0]
                if eid not in ids or not (step == 'final' or re.fullmatch(r'\d{1,5}', step)):
                    self.send_error(400); return
                path = run/('episode_'+eid)/('final.png' if step=='final' else 'frame_{:04d}.png'.format(int(step)))
                try:
                    self.send_data(path.read_bytes(), 'image/png')
                except FileNotFoundError:
                    self.send_error(404)
            else:
                self.send_error(404)

        def send_data(self, data, kind):
            self.send_response(200); self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def log_message(self, *args):
            pass

    print('NAVIGATION_REPLAY http://127.0.0.1:{} | {}'.format(args.port,run.name), flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__ == '__main__':
    main()
