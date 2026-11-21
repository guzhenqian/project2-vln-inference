"""Tiny stdlib/AST/fake-tensor QA; never imports torch/LMDeploy or reads images.

This validates Python layout/control flow only. It is not numerical or GPU
evidence. Actual candidate bytewise warmup gates remain mandatory.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


def require(value,message):
    if not value: raise AssertionError(message)


def load_stdlib_module(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class Tensor:
    def __init__(self,rows,device='cuda',dtype='float16'):
        self.rows = list(rows); self.device = SimpleNamespace(type=device); self.dtype = dtype; self.shape = (len(self.rows),1)
    def numel(self): return len(self.rows)
    def element_size(self): return 2
    def detach(self): return self
    def contiguous(self): return self
    def to(self,device,non_blocking=False):
        require(non_blocking is False,'Probe expects blocking transfers')
        return Tensor(self.rows,device.type if hasattr(device,'type') else device,self.dtype)
    def new_tensor(self,rows): return Tensor(rows,self.device.type,self.dtype)


class FakeTorch:
    @staticmethod
    def split(tensor,sizes):
        require(sum(sizes) == len(tensor.rows),'Split sizes differ')
        result = []; cursor = 0
        for size in sizes:
            result.append(Tensor(tensor.rows[cursor:cursor+size],tensor.device.type,tensor.dtype)); cursor += size
        return tuple(result)
    @staticmethod
    def cat(tensors,dim=0):
        tensors = list(tensors); require(dim == 0 and tensors,'Invalid fake cat')
        return Tensor([row for tensor in tensors for row in tensor.rows],tensors[0].device.type,tensors[0].dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path); args = parser.parse_args()
    require('torch' not in sys.modules and 'lmdeploy' not in sys.modules,'Start without model packages')
    names = ['navigation_latency_vision_cpu_cache_20261120.py','navigation_latency_vision_cpu_cache_module_map_20261120.py',
             'serve_navigation_latency_vision_cache_20261120.py','run_navigation_latency_vision_cache_replay_20261120.py']
    trees = {name:ast.parse((HERE/name).read_text(encoding='utf-8')) for name in names}
    module = load_stdlib_module('cache_static_candidate',HERE/names[0])
    checks = ['four new candidate files parse without imports','cache module top level imports stdlib only']
    # Extract the real layout function, replacing only its lazy torch import by
    # a tiny in-memory stand-in. No torch-named module is imported or installed.
    function = next(n for n in trees[names[0]].body if isinstance(n,ast.FunctionDef) and n.name == '_cached_visual')
    function.body = [n for n in function.body if not isinstance(n,ast.Import)]
    namespace = dict(module.__dict__,torch=FakeTorch)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(HERE/names[0]),'exec'),namespace)
    cached = namespace['_cached_visual']
    state = SimpleNamespace(lru=module.CpuLRU(96),identity={'weights':'fake-fixed','dtype':'float16'})
    calls = []

    def packet(labels):
        grids = [[1,2,4] if label != 'large' else [1,4,4] for label in labels]
        raw = [t*h*w for t,h,w in grids]
        pixels = Tensor([(label,j) for label,n in zip(labels,raw) for j in range(n)])
        positions = Tensor([('pos',label,j) for label,n in zip(labels,raw) for j in range(n)])
        rotary = (Tensor([('cos',label,j) for label,n in zip(labels,raw) for j in range(n)]),
                  Tensor([('sin',label,j) for label,n in zip(labels,raw) for j in range(n)]))
        cumulative=[]; total=0
        for n in raw: total+=n; cumulative.append(total)
        identities=[dict(rgb_sha256=label,rgb_bytes=24,native_grid=grid) for label,grid in zip(labels,grids)]
        return grids,identities,pixels,Tensor(cumulative,dtype='int32'),rotary,positions

    def original(pixels,cu_seqlens,rotary_pos_emb,pos_embeds):
        calls.append(list(cu_seqlens.rows)); begin=0; labels=[]; sizes=[]
        for end in cu_seqlens.rows:
            part=pixels.rows[begin:end]; label=part[0][0]
            require(part == [(label,j) for j in range(end-begin)],'Raw image patch order changed')
            require(pos_embeds.rows[begin:end] == [('pos',label,j) for j in range(end-begin)],'Position order changed')
            for rotary,kind in zip(rotary_pos_emb,('cos','sin')):
                require(rotary.rows[begin:end] == [(kind,label,j) for j in range(end-begin)],'Rotary order changed')
            labels.append(label); sizes.append((end-begin)//4); begin=end
        require(begin == len(pixels.rows) and cu_seqlens.rows[0] > 0,'Invalid miss cumulative lengths/leading zero')
        features = [Tensor([(feature,label,j) for label,n in zip(labels,sizes) for j in range(n)]) for feature in range(4)]
        return features[0],features[1:]

    def invoke(labels):
        args=packet(labels); output,stats=cached(original,state,*args)
        wanted=original(args[2],args[3],args[4],args[5]); calls.pop()
        require([t.rows for t in module.flatten_features(output)] == [t.rows for t in module.flatten_features(wanted)],'Four-feature original-slot layout mismatch')
        require(all(t.device.type == 'cpu' for entry,_ in state.lru.entries.values() for t in entry),'GPU tensor retained in cache')
        return output,stats

    _,stats=invoke(['a','a'])
    require(stats['encoded_distinct_misses'] == 1 and stats['within_request_deduplicated_miss_slots'] == 1 and calls[-1] == [8],'Duplicate-first input did not batch/dedup correctly')
    _,stats=invoke(['a','b','large'])
    require(stats['historical_hit_slots'] == 1 and stats['encoded_distinct_misses'] == 2 and calls[-1] == [8,24],'Mixed-hit variable-grid miss batch wrong')
    before=len(calls); _,stats=invoke(['large','a','large'])
    require(len(calls) == before and stats['native_vision_forward_calls'] == 0 and stats['historical_hit_slots'] == 3,'All-hit unexpectedly executed encoder')
    checks += ['duplicate first input encodes one distinct image','mixed hits batch all distinct misses once','raw/cos/sin/pos and cumulative lengths preserve variable-grid ordering','all-hit skips native encoder and restores repeated slots','all stored feature tensors are CPU owned']
    state.lru.clear(); labels=[str(i) for i in range(9)]; _,stats=invoke(labels); keys=stats['ordered_key_sha256']
    for missing in (1,3,4,8):
        state.lru.discard_for_numeric_gate(keys[:missing]); _,stats=invoke(labels)
        require(stats['encoded_distinct_misses'] == missing and stats['historical_hit_distinct'] == 9-missing,'Partial mask counts wrong')
    checks.append('partial-hit masks1/3/4/8 preserve all four feature arrays')
    # Exercise real LRU eviction bookkeeping with scalar fake arrays.
    lru=module.CpuLRU(96); feature=tuple(Tensor([1],'cpu') for _ in range(4))
    for i in range(96): lru.put(str(i),feature)
    lru.lookup(['0','1'])
    for i in range(96,103): lru.put(str(i),feature)
    lru.touch_slots(['0','1'])
    require(len(lru.entries) == 96 and lru.evictions == 7 and lru.bytes == 96*8,'LRU eviction/byte accounting failed')
    old=lru.clear(); require(old == dict(entries=96,bytes=768) and not lru.entries and lru.bytes == 0,'Actual clear failed')
    checks += ['current hits protected during miss insertion','LRU capacity/bytes/actual clear match']
    # Parse runner helpers rather than importing its dependencies.
    runner=trees[names[3]]; build=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name == 'build_cache_plan')
    scope={}; exec(compile(ast.Module(body=[build],type_ignores=[]),'cache_runner_plan','exec'),scope)
    entries=[dict(index=i,image_count=2 if i==0 else 9,source_prompt_tokens=701 if i==0 else 2815,episode_id='a' if i<2 else 'b',is_episode_start=i in (0,2)) for i in range(4)]
    plan=[dict(phase='input_warmup',sample=i) for i in (0,1)]+[dict(phase='measured',sample=i) for i in range(4)]+[dict(phase='flush',sample=0)]
    document=scope['build_cache_plan'](entries,plan,'native','manifest')
    require([r['index'] for r in document['requests'] if r['reset_before']] == [2,4] and document['requests'][-1]['expected_input_tokens'] == 701,'Episode/warmup/tail plan mismatch')
    checks.append('cache plan clears exactly episode starts after excluded warmups, never on tail')
    smoke=scope['build_cache_plan'](entries,plan[:2]+[plan[-1]],'native','manifest')
    require(len(smoke['requests']) == 3 and not any(r['reset_before'] or r['replay_index'] is not None for r in smoke['requests']),
            'Smoke plan contains historical requests or episode resets')
    runner_text=(HERE/names[3]).read_text(encoding='utf-8')
    require("prepared = [] if smoke_only else" in runner_text and "status='smoke_only'" in runner_text and
            "historical_payloads_loaded=0,historical_requests_executed=0" in runner_text and "measured_requests=0,no_latency_claim=True,no_quality_claim=True" in runner_text,
            'Smoke protocol does not explicitly exclude replay and claims')
    checks.append('smoke plan has only two excluded warmups plus tail; historical payload loading bypassed and claims disabled')
    audit_nodes=[n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name in ('audit_feature_comparison','audit_numeric_execution')]
    audit_scope=dict(require=require,math=math)
    exec(compile(ast.Module(body=audit_nodes,type_ignores=[]),'cache_numeric_audit','exec'),audit_scope)
    metadata=[dict(shape=[18,2560],dtype='torch.float16',device='cuda:0') for _ in range(4)]
    compared=dict(passed=True,features=[dict(feature=label,reference_shape=[18,2560],candidate_shape=[18,2560],
        reference_dtype='torch.float16',candidate_dtype='torch.float16',reference_device='cuda:0',candidate_device='cuda:0',
        metadata_equal=True,finite=True,distinct_storage=True,torch_equal=True,bitwise_equal=True,passed=True,max_abs_error=0.,relative_l2=0.)
        for label in ('main','deepstack0','deepstack1','deepstack2')])
    audit_scope['audit_feature_comparison'](compared,metadata)
    def expect_reject(fn):
        try: fn()
        except AssertionError: return
        raise AssertionError('Corrupt numerical evidence was accepted')
    for field,value in [('bitwise_equal',False),('finite',False),('distinct_storage',False),('max_abs_error',1e-7),('feature','main')]:
        changed=json.loads(json.dumps(compared)); changed['features'][1][field]=value
        expect_reject(lambda:audit_scope['audit_feature_comparison'](changed,metadata))
    ref=dict(ordered_key_sha256=[str(i) for i in range(9)],grids=[[1,2,4]]*9,output_features=metadata,image_slots=9,
        distinct_images=9,within_request_duplicate_slots=0,historical_hit_distinct=0,encoded_distinct_misses=9,
        historical_hit_slots=0,within_request_deduplicated_miss_slots=0,native_vision_forward_calls=1)
    part=dict(ref,historical_hit_distinct=6,historical_hit_slots=6,encoded_distinct_misses=3)
    audit_scope['audit_numeric_execution'](ref,ref)
    wrong_calls=dict(ref,native_vision_forward_calls=2)
    expect_reject(lambda:audit_scope['audit_numeric_execution'](wrong_calls,wrong_calls))
    audit_scope['audit_numeric_execution'](part,ref,['0','1','2'])
    expect_reject(lambda:audit_scope['audit_numeric_execution'](dict(part,historical_hit_slots=5),ref,['0','1','2']))
    checks += ['per-array numeric audit accepts all4 exact independent finite arrays','per-array numeric audit rejects trusted-summary-only byte/finite/storage/error/label corruption',
               'partial-mask numeric audit recomputes slots and distinct hit/miss counts',
               'initial numeric execution rejects multiple native visual forwards for one miss batch']
    # Exercise the actual visual wrapper with a fake comparator. Its frozen
    # comparator's real torch implementation is never imported or executed.
    wrapper=next(n for n in trees[names[0]].body if isinstance(n,ast.FunctionDef) and n.name == '_install_visual_wrapper')
    class RemoveComparatorImport(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            return None if node.module == 'navigation_latency_vision_qmax_20261120' else node
    wrapper=RemoveComparatorImport().visit(wrapper); ast.fix_missing_locations(wrapper)
    comparisons=[]; fail_on={'number':None}
    def fake_compare(reference,candidate):
        equal=[t.rows for t in module.flatten_features(reference)] == [t.rows for t in module.flatten_features(candidate)]
        comparisons.append(equal)
        return dict(passed=equal and len(comparisons) != fail_on['number'],scope='fake layout comparison, not actual bytewise evidence')
    namespace.update(_compare_features=fake_compare,_cached_visual=cached,_MODE='cpu',_MODULE_SHA='fake-source')
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(HERE/names[0]),'exec'),namespace)
    with tempfile.TemporaryDirectory(prefix='vision_cache_tiny_') as temp:
        # write_receipt is a module function, so configure its own globals too.
        module._MODE='cpu'; module._MODULE_SHA='fake-source'
        state=SimpleNamespace(lru=module.CpuLRU(96),identity={'weights':'fake-fixed'},directory=Path(temp),plan_sha='fake-plan',numeric_passed=set())
        visual=SimpleNamespace(forward=original)
        model=SimpleNamespace(visual=visual,_fix_time=lambda name,fn:fn())
        namespace['_install_visual_wrapper'](model,state)
        def visual_request(index,phase,labels):
            grids,identities,pixels,cu,rotary,positions=packet(labels)
            context=dict(state=state,request_index=index,spec=dict(spec=dict(phase=phase,sample=index,tokens=0,repeat=0),expected_images=len(labels),
                expected_input_tokens=701 if index == 0 else 2815),grids=grids,identities=identities,stats=None,
                numerical_gate=dict(status='not_performed_for_formal_request'))
            token=module._ACTIVE.set(context)
            try: value=model.visual.forward(pixels,cu,rotary,positions)
            finally: module._ACTIVE.reset(token)
            return value,context
        for index,length in enumerate((701,2815)):
            (Path(temp)/f'input_{length}.json').write_text(json.dumps(dict(request_index=index)),encoding='utf-8')
        visual_request(0,'input_warmup',['a','a'])
        _,context=visual_request(1,'input_warmup',[str(i) for i in range(9)])
        numeric=json.loads((Path(temp)/'vision_cache_numeric_0001.json').read_text())
        require(numeric['status'] == 'passed' and [v['distinct_miss_count'] for v in numeric['partial_hit_gates']] == [1,3,4,8] and
                state.numeric_passed == {0,1},'Actual wrapper did not finish eligible gates')
        before=len(comparisons); native_before=len(calls)
        _,context=visual_request(2,'measured',['0','1','0'])
        require(len(comparisons) == before and len(calls) == native_before and context['stats']['native_vision_forward_calls'] == 0,
                'Formal request performed numerical repetition or unnecessary encoder work')
        # Force a comparator failure in a fresh warmup receipt namespace.
        (Path(temp)/'vision_cache_numeric_0000.json').unlink()
        state.numeric_passed.clear(); fail_on['number']=len(comparisons)+1
        try: visual_request(0,'input_warmup',['a','a'])
        except RuntimeError as error: require('bytewise gate failed' in str(error),'Wrong failure mode')
        else: raise AssertionError('Failed numerical gate was accepted')
        failed=json.loads((Path(temp)/'vision_cache_numeric_0000.json').read_text())
        require(failed['status'] == 'failed' and 0 not in state.numeric_passed,'Failure receipt/gate state missing')
    checks += ['actual wrapper runs all eligible index1 partial masks with fake comparator','formal all-hit request performs zero comparisons and zero encoder calls',
               'numerical failure writes failed receipt and raises before accepting warmup']
    # Narrow source-level requirements complement dynamic fake layout checks.
    text=(HERE/names[0]).read_text(encoding='utf-8')
    require("index in (0, 1) and context['spec']['spec']['phase'] == 'input_warmup'" in text and "if index == 1 else []" in text,'Numerical eligibility broadened')
    require("from navigation_latency_vision_qmax_20261120 import _compare_features" in text and "require(passed, 'CPU visual cache bytewise gate failed" in text,'Strict bytewise helper/fail-closed missing')
    checks += ['numerical repetitions restricted to first two excluded warmups; partial masks only index1','strict frozen four-feature bytewise helper and fail-closed required']
    require('torch' not in sys.modules and 'lmdeploy' not in sys.modules,'Real model package unexpectedly imported')
    report=dict(passed=True,checks=checks,torch_imported=False,lmdeploy_imported=False,gpu_executed=False,images_read=False,
        scope='Tiny stdlib/AST/fake-tensor layout/control checks only; no real numeric or performance evidence',
        source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names})
    if args.output:
        with args.output.open('x',encoding='utf-8') as stream: json.dump(report,stream,ensure_ascii=False,indent=2); stream.write('\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': main()
