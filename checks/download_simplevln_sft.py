"""Download the pinned author's BF16 navigation checkpoint with Range resume and SHA checks."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
metadata=json.loads((ROOT/'tmp/habitat_setup/navigation_source_review/simplevln_model_sft.json').read_text(encoding='utf8'))
assert metadata['sha']=='2b2ae6e30f3e344fb9c6d2ed11ee9c117ac24566'
directory=ROOT/'models/SimpleVLN_Qwen3VL_4B_SFT'
directory.mkdir(parents=True,exist_ok=True)


def download(item):
    name=item['rfilename']
    if name.startswith('.'):return None
    target=directory/name
    expected=item.get('lfs',{}).get('sha256')
    if not target.exists():
        partial=target.with_suffix(target.suffix+'.part')
        url='https://huggingface.co/{}/resolve/{}/{}'.format(metadata['id'],metadata['sha'],name)
        print('DOWNLOAD',name,item.get('size'),flush=True)
        for attempt in range(8):
            offset=partial.stat().st_size if partial.exists() else 0
            if offset==item['size']:break
            request=urllib.request.Request(url,headers={'Range':'bytes={}-'.format(offset)} if offset else {})
            try:
                with urllib.request.urlopen(request,timeout=120) as response:
                    if offset:
                        assert response.status==206 and response.headers.get('Content-Range','').startswith('bytes {}-'.format(offset)), 'Unsafe resume response'
                    last=time.monotonic()
                    with partial.open('ab' if offset else 'wb') as output:
                        while True:
                            block=response.read(1024*1024)
                            if not block:break
                            output.write(block);offset+=len(block)
                            if time.monotonic()-last>15:
                                print('PROGRESS',name,offset,'/',item['size'],flush=True);last=time.monotonic()
                if offset==item['size']:break
            except Exception as error:
                print('RETRY',name,attempt+1,type(error).__name__,str(error)[:180],flush=True)
                if attempt==7:raise
                time.sleep(min(2**attempt,10))
        assert partial.stat().st_size==item['size'],name
        partial.rename(target)
    assert target.stat().st_size==item['size'],name
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    actual=digest.hexdigest()
    if expected:assert expected==actual,name
    print('VERIFIED',name,actual,flush=True)
    return {'name':name,'bytes':target.stat().st_size,'sha256':actual,'source_lfs_sha256':expected}


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    files=list(pool.map(download,metadata['siblings']))
(directory/'source_manifest.json').write_text(json.dumps({'repo':metadata['id'],'revision':metadata['sha'],
    'purpose':'navigation-trained SFT checkpoint; independent from original generic 2B baseline',
    'author_code_revision':'a450c57cff323587f54b0654c8e20e530a7202ac','files':[f for f in files if f]},indent=2),encoding='utf8')
print('SIMPLEVLN_SFT_CHECKPOINT_READY',flush=True)
