"""Finish the pinned public RL checkpoint using small, independently resumable Range parts."""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'tmp/habitat_setup/navigation_source_review/simplevln_model_rl.json').read_text(encoding='utf8'))
assert meta['sha']=='4078dab77f3c93a0711b57549ac54a59c1c1e1ea'
folder=ROOT/'models/SimpleVLN_Qwen3VL_4B_RL'
item=next(x for x in meta['siblings'] if x['rfilename']=='model.safetensors')
size=item['size'];expected=item['lfs']['sha256']
target=folder/'model.safetensors';prefix=folder/'model.safetensors.part'
parts=folder/'range_parts';parts.mkdir(exist_ok=True)
plan_file=parts/'plan.json'
if plan_file.exists():plan=json.loads(plan_file.read_text(encoding='utf8'))
else:
    with prefix.open('rb') as stream:stream.seek(0,2);start=stream.tell()
    plan={'prefix_bytes':start,'total_bytes':size,'chunk_bytes':16*1024*1024,'revision':meta['sha']}
    plan_file.write_text(json.dumps(plan,indent=2),encoding='utf8')
assert plan['total_bytes']==size and plan['revision']==meta['sha']
url='https://huggingface.co/{}/resolve/{}/model.safetensors'.format(meta['id'],meta['sha'])
ranges=[(start,min(size-1,start+plan['chunk_bytes']-1)) for start in range(plan['prefix_bytes'],size,plan['chunk_bytes'])]


def get_part(bounds):
    start,end=bounds;dest=parts/('{}-{}.bin'.format(start,end))
    for attempt in range(12):
        have=dest.stat().st_size if dest.exists() else 0
        if have==end-start+1:return dest
        assert have<end-start+1
        offset=start+have
        request=urllib.request.Request(url+'?range_part='+str(start)+'&try='+str(time.time_ns()),
            headers={'Range':'bytes={}-{}'.format(offset,end)})
        try:
            begun=time.monotonic()
            with urllib.request.urlopen(request,timeout=12) as response:
                assert response.status==206 and response.headers['Content-Range']=='bytes {}-{}/{}'.format(offset,end,size)
                with dest.open('ab') as output:
                    while True:
                        block=response.read1(65536)
                        if not block:break
                        output.write(block)
                        if time.monotonic()-begun>35:raise TimeoutError('Part deadline')
            if dest.stat().st_size==end-start+1:return dest
        except Exception as error:
            if attempt==11:raise
            print('PART_RETRY',start,attempt+1,type(error).__name__,flush=True)
            time.sleep(1)
    raise RuntimeError('Incomplete part '+str(start))


if not target.exists():
    print('RETAINED_PREFIX',plan['prefix_bytes'],'REMAINING_PARTS',len(ranges),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(get_part,bounds) for bounds in ranges]
        for count,future in enumerate(concurrent.futures.as_completed(futures),1):
            future.result()
            if count%8==0 or count==len(ranges):print('PARTS_READY',count,'/',len(ranges),flush=True)
    with prefix.open('rb') as stream:stream.seek(0,2);assert stream.tell()==plan['prefix_bytes']
    digest=hashlib.sha256()
    with prefix.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    with prefix.open('ab') as output:
        for start,end in ranges:
            with (parts/('{}-{}.bin'.format(start,end))).open('rb') as source:
                for block in iter(lambda:source.read(8*1024*1024),b''):
                    output.write(block);digest.update(block)
    assert prefix.stat().st_size==size and digest.hexdigest()==expected,'Final LFS checksum mismatch'
    prefix.rename(target)
    print('RL_WEIGHT_LFS_SHA256_OK',expected,flush=True)
records=[]
for record in meta['siblings']:
    name=record['rfilename']
    if name.startswith('.'):continue
    path=folder/name;assert path.stat().st_size==record['size']
    if name=='model.safetensors':sha=expected
    else:sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if record.get('lfs'):assert sha==record['lfs']['sha256']
    records.append({'name':name,'bytes':path.stat().st_size,'sha256':sha,'source_lfs_sha256':record.get('lfs',{}).get('sha256')})
(folder/'source_manifest.json').write_text(json.dumps({'repo':meta['id'],'revision':meta['sha'],'purpose':'public navigation RL checkpoint, loaded BF16 for same-runtime comparison; no local training',
    'author_code_revision':'a450c57cff323587f54b0654c8e20e530a7202ac','files':records},indent=2),encoding='utf8')
print('SIMPLEVLN_RL_CHECKPOINT_READY',flush=True)
