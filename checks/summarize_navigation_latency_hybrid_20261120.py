"""Independent hybrid-capable copy of the frozen latency-study auditor.

Recomputes metrics from request/timing files. Does not start services, import
torch or pool quantiles across repetitions. Wide-kernel validation independently
reads selected packed checkpoint tensors; run audits outside active GPU timings.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import mmap
from pathlib import Path
import statistics

from navigation_latency_w4_hybrid_audit_20261120 import (
    audit_hybrid_install, audit_hybrid_request, hybrid_output_gate, apply_hybrid_output_gates)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    'fp16': 'b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b',
    'awq': '31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
    'gptq': '2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f',
}
MODEL_PATHS = {'awq':'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104',
               'gptq':'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_stage2_20261109'}
PROJECTIONS = {'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
               'o':['self_attn.o_proj'], 'gate_up':['mlp.gate_proj','mlp.up_proj'], 'down':['mlp.down_proj']}
PROJECTION_SHAPES = {'qkv':(2560,6144),'o':(4096,2560),'gate_up':(2560,19456),'down':(9728,2560)}
_WIDE_PROBE_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def sha_object(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def distribution(values):
    values = sorted(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'Invalid metric samples')
    pos = (len(values)-1)*.95
    lo, hi = math.floor(pos), math.ceil(pos)
    return dict(n=len(values), p50=statistics.median(values),
                p95=values[lo]+(values[hi]-values[lo])*(pos-lo), min=values[0], max=values[-1])


def source_audit(run, protocol, launch):
    verified = {}
    for name, expected in protocol['source_sha256'].items():
        snapshot = run/'source_snapshot'/Path(name).name
        require(snapshot.is_file() and digest(snapshot) == expected, 'Protocol snapshot SHA mismatch: '+name)
    for name, expected in launch['sources_sha256'].items():
        snapshot = run/'source_snapshot'/Path(name).name
        if snapshot.is_file():
            require(digest(snapshot) == expected, 'Launch vs snapshot SHA mismatch: '+name)
            verified[name] = dict(sha256=expected, source='retained_snapshot')
        else:
            path = ROOT/'checks'/name
            require(path.is_file() and digest(path) == expected, 'Unretained source differs from launch SHA: '+name)
            verified[name] = dict(sha256=expected, source='current_file_matches_launch_receipt')
    return verified


def source_constant(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Guard constant not found: '+name)


def guarded_package_sources(expected):
    # Metadata lookup does not import LMDeploy or torch.
    from importlib.metadata import distribution
    package = distribution('lmdeploy')
    for name, sha in expected.items():
        require(digest(package.locate_file('lmdeploy/'+name)) == sha, 'Guarded LMDeploy source changed: '+name)


def recorded_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    require(path.is_relative_to(ROOT), 'Recorded evidence is outside project: '+str(path))
    return path


def fused_tensor_hashes(checkpoint, layer, projection):
    """Hash the exact dimension-1 fusion using raw mmap bytes, without torch.

    Only the requested real-layer tensors are read, never the full checkpoint.
    Schedule this independent verification outside active GPU timing intervals.
    """
    formats = {'I32':(4,'torch.int32'), 'F16':(2,'torch.float16')}
    result = {}
    with Path(checkpoint).open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        require(0 < size < 32*1024*1024, 'Invalid safetensors header length')
        header = json.loads(stream.read(size))
        data_start = size+8
        with mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
            with memoryview(mapped) as view:
                for suffix in ('qweight','scales','qzeros'):
                    parts = [header[f'model.language_model.layers.{layer}.{part}.{suffix}'] for part in PROJECTIONS[projection]]
                    dtype = 'F16' if suffix == 'scales' else 'I32'
                    require(all(v['dtype'] == dtype and len(v['shape']) == 2 for v in parts), 'Packed tensor dtype/rank differs')
                    rows = parts[0]['shape'][0]
                    require(all(v['shape'][0] == rows for v in parts), 'Cannot fuse tensors with differing row dimensions')
                    itemsize, torch_dtype = formats[dtype]
                    for part in parts:
                        start,end = part['data_offsets']
                        require(0 <= start <= end and end-start == math.prod(part['shape'])*itemsize and
                                data_start+end <= len(view), 'Tensor offset/shape differs')
                    checksum = hashlib.sha256()
                    for row in range(rows):
                        for part in parts:
                            width = part['shape'][1]*itemsize
                            offset = data_start+part['data_offsets'][0]+row*width
                            checksum.update(view[offset:offset+width])
                    result[suffix] = dict(shape=[rows,sum(v['shape'][1] for v in parts)],
                                          dtype=torch_dtype,sha256=checksum.hexdigest())
    return result


def wide_probe_audit(table_path, launch, verified_sources, service):
    table_path = recorded_path(table_path)
    protocol_path, result_path = table_path.parent/'protocol.json', table_path.parent/'result.json'
    require(protocol_path.is_file() and result_path.is_file(), 'Wide table lacks adjacent completed numerical probe evidence')
    require(not (table_path.parent/'failure.json').exists(), 'Wide probe retained a failure; not eligible')
    table, protocol, probe = read(table_path), read(protocol_path), read(result_path)
    method = launch['recipe']['method']; checkpoint = ROOT/MODEL_PATHS[method]/'model.safetensors'
    file_stat = checkpoint.stat()
    key = (digest(table_path),digest(result_path),digest(protocol_path),str(checkpoint),
           file_stat.st_size,file_stat.st_mtime_ns,launch['model_weights_sha256'],
           sha_object({k:v['sha256'] for k,v in verified_sources.items()}))
    require(probe['status'] == 'passed' and probe['numerical_gate_passed'] and probe['protocol'] == protocol,
            'Wide numerical probe is not complete or protocol differs')
    require(probe['launch_table'] == table and table['scope'] == 'exact_m_only' and table['numerical_gate_passed'],
            'Wide table differs from completed numerical result')
    require(table['algorithm'] == method == protocol['algorithm'] and
            table['model_weights_sha256'] == FROZEN[method] == protocol['model_weights_sha256'] == launch['model_weights_sha256'],
            'Wide numerical/model checkpoint identity differs')
    require(protocol['relative_l2_limit'] == .005 and protocol['layers'] == [0], 'Unexpected wide numerical threshold/search scope')
    require(set(protocol['rows']) == {1,701,2815} and set(protocol['projections']) == set(PROJECTIONS), 'Wide search shape coverage incomplete')
    require(set(protocol['sources_sha256']) == {'navigation_latency_w4_kernels_20261120.py',
            'probe_navigation_latency_w4_20261120.py','navigation_w4a16_repair_kernels.py',
            'navigation_w4a16_candidate_kernels.py'}, 'Wide probe source inventory differs')
    for name, expected in protocol['sources_sha256'].items():
        snapshot = service.parent.parent/'source_snapshot'/name
        chosen = snapshot if snapshot.is_file() and digest(snapshot) == expected else ROOT/'checks'/name
        require(chosen.is_file() and digest(chosen) == expected, 'Wide probe source cannot be verified: '+name)
        if name in verified_sources:
            require(verified_sources[name]['sha256'] == expected, 'Wide service/probe kernel source differs: '+name)
    baseline_path = recorded_path(protocol['baseline_table'])
    require(digest(baseline_path) == protocol['baseline_table_sha256'], 'Wide probe baseline table changed')
    baseline = read(baseline_path); baseline = baseline.get('frozen_launch_table',baseline)
    replayed = read(table_path.parent/'baseline_replayed_table.json')
    require(replayed['frozen'] and replayed['actual_m_preserved'] and replayed['generic_m_specialization_disabled'] and
            not replayed['runtime_autotuning_allowed'] and not replayed['runtime_cold_compilation_allowed'] and
            replayed['selection_mode'] == 'replayed_frozen_table' and
            replayed['frozen_table_source']['sha256'] == protocol['baseline_table_sha256'] and
            replayed['shapes'] == baseline['shapes'] and replayed['representatives'] == baseline['representatives'],
            'Wide probe baseline replay/configuration evidence differs')
    for name,key_name in (('numerical_validation.json','numerical_validation'),('measurements.json','measurements')):
        require(read(table_path.parent/name) == probe[key_name], 'Wide probe sidecar disagrees with final result: '+name)
    if key in _WIDE_PROBE_CACHE:
        return _WIDE_PROBE_CACHE[key]
    expected_shapes = {(m,k,n) for k,n in PROJECTION_SHAPES.values() for m in (1,701,2815)}
    shapes = {(v['m'],v['k'],v['n']):v['config'] for v in table['shapes']}
    require(len(shapes) == len(table['shapes']) == 12 and set(shapes) == expected_shapes, 'Wide table exact-M/layout coverage differs')
    validations = probe['numerical_validation']
    require(table['validation_all_passed'] and table['validation_cases'] == len(validations) == 72 and
            table['validation_layers'] == protocol['validation_layers'] == [0,18,35] and
            table['validation_input_magnitudes'] == protocol['validation_input_magnitudes'] == [.1,1.],
            'Wide independent real-layer numerical coverage incomplete')
    expected_cases = {(l,p,m,a) for l in (0,18,35) for p in PROJECTIONS for m in (1,701,2815) for a in (.1,1.)}
    observed_cases, tensor_refs = set(), {}
    for value in validations:
        identity = (value['layer'],value['projection'],value['m'],value['input_magnitude'])
        require(identity in expected_cases and identity not in observed_cases, 'Missing/duplicate wide numerical case')
        observed_cases.add(identity)
        k,n = PROJECTION_SHAPES[value['projection']]
        require((value['k'],value['n']) == (k,n) and value['config'] == shapes[(value['m'],k,n)], 'Wide numerical selected configuration differs')
        require(value['passed'] and math.isfinite(value['relative_l2']) and 0 <= value['relative_l2'] < .005,
                'Wide numerical tolerance failed')
        tid = (value['layer'],value['projection'])
        if tid in tensor_refs: require(value['tensors'] == tensor_refs[tid], 'Real-layer tensor hashes changed between numerical cases')
        tensor_refs[tid] = value['tensors']
    require(observed_cases == expected_cases, 'Wide real-layer numerical coverage incomplete')
    independently_verified = {}
    for (layer,projection), expected in tensor_refs.items():
        actual = fused_tensor_hashes(checkpoint,layer,projection)
        k,n = PROJECTION_SHAPES[projection]
        require(actual['qweight']['shape'] == [k,n//8] and actual['scales']['shape'] == [k//128,n] and
                actual['qzeros']['shape'] == [k//128,n//8], 'Real packed checkpoint layout differs')
        require(actual == expected, f'Wide numerical tensors are not the real checkpoint fusion: {layer}/{projection}')
        independently_verified[f'{layer}/{projection}'] = actual
    measurements = probe['measurements']; require(len(measurements) == 12, 'Wide timing shape coverage incomplete')
    observed_shapes, cold_samples = set(), 0
    for value in measurements:
        shape = (value['m'],value['k'],value['n'])
        require(shape in shapes and shape not in observed_shapes and value['selected_config'] == shapes[shape], 'Wide measured shape/config differs')
        observed_shapes.add(shape)
        for numeric in ('numerical','baseline_numerical'):
            require(value[numeric]['passed'] and 0 <= value[numeric]['relative_l2'] < .005, 'Wide search winner numerical check failed')
        for prefix, condition in (('confirmation','hot'),('cold_confirmation','eviction_before_each_sample')):
            for mode in ('baseline','candidate'):
                blocks = value[prefix][mode]
                require(len(blocks) == 2, 'Missing independent confirmation rounds')
                samples = []
                for block in blocks:
                    require(block['cache_condition'] == condition and block['mode'] == ('cuda_graph_replay' if value['m'] == 1 else 'eager'),
                            'Hot/cold or decode measurement mode differs')
                    require(all(math.isfinite(v) and v > 0 for v in block['samples_ms']), 'Invalid projection timings')
                    expected_count = protocol['confirm_repeats'] if condition == 'hot' else protocol['cold_repeats']
                    require(len(block['samples_ms']) == expected_count, 'Projection confirmation repeat count differs')
                    samples.extend(block['samples_ms'])
                    if condition != 'hot':
                        require(block['flush_gpu_ms_in_kernel_timing'] is False and block['eviction_buffer_bytes'] == protocol['cold_cache_mib']*1048576,
                                'Cold cache eviction entered operation timing or buffer differs')
                        cold_samples += len(block['samples_ms'])
                require(math.isclose(statistics.median(samples),value[prefix+'_p50_ms'][mode],rel_tol=1e-12), 'Microbenchmark median recomputation differs')
    require(observed_shapes == expected_shapes, 'Missing wide timing layout')
    expected_boundaries = {(p,m) for p in PROJECTIONS for m in protocol['boundary_rows']}
    seen = set()
    for b in probe['boundary_checks']:
        identity = (b['projection'],b['m']); k,n = PROJECTION_SHAPES[b['projection']]
        require(identity in expected_boundaries and identity not in seen and b['config'] == shapes[(b['selected_at_m'],k,n)],
                'Wide boundary coverage/config differs')
        require(b['passed'] and 0 <= b['relative_l2'] < .005, 'Wide boundary numerical check failed'); seen.add(identity)
    require(seen == expected_boundaries and {2,16,17,33,65,713,1737,3001,3072}.issubset(protocol['boundary_rows']),
            'Wide boundary checks incomplete')
    eviction = probe['cache_flush_evidence']
    require(eviction['verified'] and eviction['flush_count'] == cold_samples and
            eviction['buffer_bytes'] == protocol['cold_cache_mib']*1048576 > 48*1048576 and
            eviction['full_buffer_checksum'] == eviction['expected_checksum'] ==
            (cold_samples % 256)*eviction['buffer_bytes'], 'Cold cache device checksum/counter differs')
    result = dict(status='verified', table_sha256=digest(table_path), table=table,
        probe_result_sha256=digest(result_path), probe_protocol_sha256=digest(protocol_path),
        numerical_cases=72, max_relative_l2=max(v['relative_l2'] for v in validations),
        real_checkpoint_tensor_hashes=independently_verified,
        tensor_verification='Independent safetensors mmap byte hashing of dimension-1 fused qweight/scales/qzeros on layers0/18/35; no torch or GPU',
        baseline_table_sha256=protocol['baseline_table_sha256'],
        limitations='Search timings use layer0; numerical validation uses real layers0/18/35. Fixed-input and dynamic navigation execution coverage are audited separately.')
    _WIDE_PROBE_CACHE[key] = result
    return result


def wide_install_audit(service, launch, verified_sources):
    env = launch['candidate_environment']; phase = env.get('PROJECT2_LATENCY_W4_PHASE','all')
    require(phase in ('all','prefill'), 'Unknown wide W4 phase')
    require(launch['recipe']['weight_bits'] == 4 and launch['recipe']['activation_bits'] == 16, 'Wide requires W4A16')
    for name in ('navigation_latency_w4_kernels_20261120.py','navigation_latency_w4_adapter_20261120.py','navigation_latency_w4_module_map_20261120.py'):
        require(name in verified_sources, 'Wide source absent from launch receipt: '+name)
    require(Path(launch['engine']['custom_module_map']).name == 'navigation_latency_w4_module_map_20261120.py', 'Wide model mapping differs')
    probe = wide_probe_audit(env['PROJECT2_LATENCY_W4_TABLE'],launch,verified_sources,service)
    path = service/'wide_launch_install.json'; install = read(path)
    require(install['algorithm'] == launch['recipe']['method'] and install['checkpoint_sha256'] == launch['model_weights_sha256'], 'Wide installation checkpoint identity differs')
    require(install['candidate_table_sha256'] == probe['table_sha256'] and
            recorded_path(install['candidate_table_path']) == recorded_path(env['PROJECT2_LATENCY_W4_TABLE']), 'Wide installed table differs')
    # Older all-phase adapter receipts are accepted only when their verified source
    # and explicit all-candidate coverage agree. Prefill mode needs new fields.
    observed_phase = install.get('phase','all')
    require(observed_phase == phase and install.get('mandatory_table_m',install['mandatory_candidate_m']) == [1,701,2815], 'Wide mandatory table M/phase differs')
    require(install['mandatory_candidate_m'] == ([1,701,2815] if phase == 'all' else [701,2815]), 'Wide mandatory candidate M differs')
    require(install.get('decode_route','candidate') == ('candidate' if phase == 'all' else 'baseline_decode'), 'Wide installed decode route differs')
    require(install['actual_projection_count'] == 144 and install['cold_compile_guard_enabled'], 'Wide installation coverage/cold compile guard missing')
    frozen = install['table']
    require(frozen['frozen'] and frozen['scope'] == 'exact_m_only' and frozen['algorithm'] == probe['table']['algorithm'] and
            frozen['model_weights_sha256'] == launch['model_weights_sha256'], 'Installed wide table is not frozen/identified')
    selected = lambda values:{(v['m'],v['k'],v['n']):v['config'] for v in values}
    require(selected(frozen['shapes']) == selected(probe['table']['shapes']), 'Installed wide configurations differ from numeric gate')
    return dict(status='verified_install_and_numerical_probe', phase=phase,
                implementation_identity=dict(backend='wide_exact_m', phase=phase, table_sha256=probe['table_sha256']),
                install_receipt_sha256=digest(path), numerical_probe=probe)


def wide_execution_audit(service, index, timing, contract, native_receipt):
    wide = contract.get('wide_evidence')
    if not wide: return None
    path = service/f'wide_execution_{index:03d}.json'; receipt = read(path)
    phase = wide['phase']; route = 'candidate' if phase == 'all' else 'baseline_decode'
    modules = {r['name'] for r in native_receipt['module_calls']}
    require(len(modules) == 144, 'Native projection inventory incomplete')
    require(receipt['index'] == index and receipt.get('phase','all') == phase, 'Wide request/phase mapping differs')
    prefill = [s['rows'] for s in timing['steps'] if s['phase'] == 'prefill']
    require(receipt['prefill_m'] == prefill and len(prefill) == 1, 'Wide prefill mapping differs')
    require(receipt['outside_graph_execution_modes'] == dict(Counter(s['execution_mode'] for s in timing['steps'])) and
            receipt['graph_replay_python_counters_increment'] is False, 'Wide graph execution evidence differs')
    captures = receipt.get('candidate_capture_projection_calls',{}) if phase == 'all' else receipt.get('baseline_decode_capture_projection_calls',{})
    require(set(captures) == modules and all(isinstance(v,int) and v>0 for v in captures.values()), 'Wide expected decode capture coverage incomplete')
    other = receipt.get('baseline_decode_capture_projection_calls',{}) if phase == 'all' else receipt.get('candidate_capture_projection_calls',{})
    require(not other and not receipt.get('unexpected_capture_projection_calls',receipt.get('noncandidate_capture_projection_calls',{})), 'Wrong projection route captured')
    require(receipt.get('expected_decode_route','candidate') == route, 'Wide decode route differs')
    require(receipt.get('mandatory_prefill_route_coverage_passed',receipt.get('mandatory_prefill_candidate_coverage_passed',False)), 'Wide prefill coverage flag failed')
    m = prefill[0]; table_ms={v['m'] for v in wide['numerical_probe']['table']['shapes']}
    expected_route = 'baseline_decode' if m == 1 and phase == 'prefill' else 'candidate' if m in table_ms else 'baseline_fallback'
    calls = receipt['python_projection_calls']; keys=set()
    for c in calls:
        require(c['module'] in modules and c['m'] == m and c['route'] == expected_route and c['calls'] == 1,
                'Wide prefill used unexpected route/count/M')
        require(c['module'] not in keys, 'Duplicate wide projection execution'); keys.add(c['module'])
    require(keys == modules, 'Wide prefill did not cover all 144 projections')
    return dict(index=index, prefill_m=m, prefill_route=expected_route, decode_route=route,
                actual_projection_count=144, receipt_sha256=digest(path))


def vision_qmax_audit(service, launch, verified_sources):
    """Verify first-grid host receipts and the two excluded numerical warmups.

    This is deliberately not a per-request kernel execution counter. The patch
    records each distinct CPU image grid once, and stays installed afterwards.
    """
    env = launch.get('candidate_environment',{})
    enabled = env.get('PROJECT2_LATENCY_VISION_QMAX') == '1'
    paths = sorted(service.glob('vision_qmax_*.json'))
    if not enabled:
        require(not paths, 'Unexpected vision QMAX receipts without opt-in')
        return None
    require(env.get('PROJECT2_LATENCY_VISION_QMAX_VALIDATE') == '1',
            'Vision QMAX requires the two excluded same-run numerical warmups; external proof inheritance is unsupported')
    module = 'navigation_latency_vision_qmax_20261120.py'
    require(module in verified_sources, 'Vision QMAX module absent from launch source receipt')
    snapshot = service.parent.parent/'source_snapshot'/module
    source = snapshot if snapshot.is_file() else ROOT/'checks'/module
    require(digest(source) == verified_sources[module]['sha256'], 'Vision QMAX source differs from launch')
    guarded = source_constant(source,'SOURCE_SHA256')
    require(set(guarded) == {'pytorch/models/qwen3_vl.py','pytorch/models/qwen2_5_vl.py',
            'pytorch/nn/attention.py','pytorch/backends/cuda/flash_attention.py',
            'pytorch/kernels/cuda/flashattention.py'}, 'Vision QMAX source guard inventory differs')
    guarded_package_sources(guarded)
    expected_specs = [dict(phase='input_warmup',sample=i,tokens=0,repeat=0) for i in (0,1)]
    plan = read(service/'request_plan.json')
    require(plan[:2] == expected_specs, 'Vision QMAX numerical validation is not restricted to the two excluded warmups')

    def function(value, expected_module, expected_name, filename):
        require(value['module'] == expected_module and value['name'] == expected_name and
                Path(value['code_filename']).name == filename, 'Vision QMAX function identity differs')

    def layer_calls(calls, grid, native=False):
        total = sum(t*h*w for t,h,w in grid); bound = total if native else max(h*w for t,h,w in grid)
        require([c['layer'] for c in calls] == list(range(24)), 'Vision QMAX must execute all 24 layers once')
        shapes = []
        for c in calls:
            require(len(c['query_shape']) == 3 and c['query_shape'][0] == total and
                    all(isinstance(v,int) and v > 0 for v in c['query_shape']) and
                    c['query_dtype'] == 'torch.float16' and c['query_device'].startswith('cuda'),
                    'Vision QMAX query metadata differs')
            require(c['original_default_max_q_seqlen'] == total and c['effective_max_q_seqlen'] == bound,
                    'Vision QMAX used the wrong total/per-image bound')
            require(c['attention_class'] == 'lmdeploy.pytorch.nn.attention.FlashAttention' and
                    c['backend_class'] == 'lmdeploy.pytorch.backends.cuda.flash_attention.TritonFlashAttentionImpl',
                    'Vision QMAX attention/backend differs')
            function(c['installed_forward'],'lmdeploy.pytorch.nn.attention','forward',module)
            function(c['original_forward'],'lmdeploy.pytorch.nn.attention','forward','attention.py')
            function(c['native_kernel'],'lmdeploy.pytorch.kernels.cuda.flashattention','flash_attention_fwd','flashattention.py')
            shapes.append((c['query_shape'],c['query_dtype'],c['query_device']))
        return shapes

    installs, executions, numerical, hashes = {}, [], {}, {}
    for path in paths:
        value = read(path); pid = value['pid']
        require(isinstance(pid,int) and pid > 0 and value['time_ns'] > 0 and
                value['patch_module'] == module.removesuffix('.py') and
                value['patch_module_sha256'] == verified_sources[module]['sha256'] and
                value['original_source_sha256'] == guarded and value['enabled_env'] == '1' and
                value['validation_enabled_env'] == '1', 'Vision QMAX receipt source/environment/process differs')
        hashes[relative(path)] = digest(path)
        if value['kind'] == 'install_receipt':
            require(path.name == f'vision_qmax_install_{pid}.json' and pid not in installs, 'Duplicate/invalid QMAX install receipt')
            function(value['actual_installed_model_forward'],'lmdeploy.pytorch.models.qwen3_vl','forward',module)
            function(value['original_model_forward'],'lmdeploy.pytorch.models.qwen3_vl','forward','qwen3_vl.py')
            installs[pid] = value
        elif value['kind'] == 'first_shape_execution_receipt':
            require(path.name.startswith(f'vision_qmax_execute_{pid}_'), 'Vision QMAX execution filename differs')
            executions.append((path,value))
        elif value['kind'] == 'vision_numerical_gate':
            index = value['warmup']['request_index']
            require(index in (0,1) and index not in numerical and
                    path.name == f'vision_qmax_numeric_{pid}_{2 if index == 0 else 9:02d}images.json',
                    'Unexpected/duplicate QMAX numerical gate outside warmups')
            numerical[index] = (path,value)
        else:
            raise ValueError('Unknown vision QMAX receipt kind')
    require(executions and set(numerical) == {0,1}, 'Vision QMAX actual execution/numerical warmups missing')
    seen_grids, warmups, coverage = set(), {}, []
    for path,value in executions:
        require(value['pid'] in installs, 'Vision QMAX execution lacks same-process installation')
        grid = value['grid_thw']; signature = canonical(grid)
        require(grid and all(len(r) == 3 and all(isinstance(v,int) and v > 0 for v in r) and r[0] == 1 for r in grid),
                'Vision QMAX grid is not a positive static-image grid')
        require(signature not in seen_grids, 'Duplicate QMAX first-grid execution'); seen_grids.add(signature)
        total = sum(t*h*w for t,h,w in grid); bound = max(h*w for t,h,w in grid)
        require(value['cpu_grid_reused'] and value['regular_path_device_scalar_reads_added'] == 0 and
                value['regular_path_added_cuda_synchronizations'] == 0 and value['layer_count'] == 24 and
                value['total_patches'] == total and value['segments'] == len(grid) and
                value['effective_max_q_seqlen'] == bound, 'Vision QMAX grid/regular-path contract differs')
        launch_grid = value['launch_grid_bound']
        require(launch_grid['old_max_q_seqlen'] == total and launch_grid['new_max_q_seqlen'] == bound and
                launch_grid['kernel_meta_selection_unchanged'] is True and launch_grid['per_image_grid_thw'] == grid and
                launch_grid['grid_formula'] == '(ceil(max_q_seqlen/native_BLOCK_M), query_heads, segment_count)',
                'Vision QMAX changed kernel metadata or grid bound')
        layer_calls(value['actual_layer_calls'],grid)
        index = value['request_index']
        require(isinstance(index,int) and index >= 0 and value['input_tokens'] > 0, 'Vision QMAX request identity invalid')
        if index in (0,1):
            require(index not in warmups and len(grid) == (2 if index == 0 else 9) and
                    value['request_spec'] == expected_specs[index] and value['input_tokens'] == (701 if index == 0 else 2815),
                    'Vision QMAX excluded input warmup identity differs')
            warmups[index] = (path,value)
        else:
            require(value['numerical_gate']['status'] in ('skipped_outside_explicit_warmup_scope','not_repeated_after_first_shape'),
                    'Vision QMAX ran numerical double-forward outside excluded warmups')
        coverage.append(dict(receipt_sha256=digest(path),pid=value['pid'],request_index=index,
            input_tokens=value['input_tokens'],grid_thw=grid,total_patches=total,max_q_seqlen=bound,layers=24))
    require(set(warmups) == {0,1}, 'Vision QMAX first execution of both fixed grids missing')
    for index,(path,value) in numerical.items():
        _, executed = warmups[index]
        require(value['pid'] == executed['pid'] and value['grid_thw'] == executed['grid_thw'] and
                value['status'] == 'passed' and value['passed'] is True and
                value['additional_visual_forwards'] == 1 and value['scalar_comparisons_synchronize_cuda'] is True,
                'Vision QMAX numerical gate failed or runtime mapping differs')
        require(value['warmup']['request_spec'] == expected_specs[index] and
                recorded_path(value['warmup']['input_sha_receipt']) == service/f'input_{executed["input_tokens"]}.json',
                'Vision QMAX numerical gate is not bound to actual input SHA receipt')
        input_receipt = read(recorded_path(value['warmup']['input_sha_receipt']))
        require(input_receipt['request_index'] == index and
                input_receipt['input_ids']['shape'] == [1,executed['input_tokens']] and
                input_receipt['pixel_values']['shape'][0] == executed['total_patches'],
                'Vision QMAX input SHA identity differs')
        require(layer_calls(value['native_layer_calls'],value['grid_thw'],native=True) ==
                layer_calls(value['candidate_layer_calls'],value['grid_thw']) and
                value['candidate_layer_calls'] == executed['actual_layer_calls'], 'Vision QMAX numerical layer inputs differ')
        features = value['features']
        require([f['feature'] for f in features] == ['main','deepstack0','deepstack1','deepstack2'],
                'Vision QMAX numerical gate omitted a visual/DeepStack output')
        for feature in features:
            require(all(feature[k] is True for k in ('metadata_equal','finite','distinct_storage','torch_equal','bitwise_equal','passed')) and
                    feature['reference_shape'] == feature['candidate_shape'] and
                    feature['reference_dtype'] == feature['candidate_dtype'] == 'torch.float16' and
                    feature['reference_device'] == feature['candidate_device'] and feature['reference_device'].startswith('cuda') and
                    feature['max_abs_error'] == feature['relative_l2'] == 0, 'Vision QMAX visual features differ or alias storage')
        saved = executed['numerical_gate']
        require(all(saved[k] == value[k] for k in saved) and saved['receipt'] == path.name,
                'Vision QMAX numerical and completed-forward receipts disagree')
    return dict(status='verified_two_excluded_warmups',source_sha256=verified_sources[module]['sha256'],
        guarded_package_sha256=guarded,execution_receipt_sha256=hashes,first_grid_coverage=coverage,
        numerical_warmups={str(i):dict(receipt_sha256=digest(p),features=v['features'],warmup=v['warmup']) for i,(p,v) in numerical.items()},
        scope='First successful host execution per distinct grid, not per-request GPU counters. Two 2/9-image warmups prove bitwise visual+DeepStack outputs; later calls remain single-forward by guarded source.',
        timing_exclusion='Indices0/1 double visual forward and scalar synchronization excluded. Whole-card peak memory samples may include numerical validation and are not a steady-state memory comparison.')


def execution_contract(service, launch, verified_sources):
    """Separate requested settings from observed construction/transport paths."""
    requested = dict(launch['engine'])
    effective = dict(requested)
    env = launch.get('candidate_environment', {})
    inprocess = env.get('PROJECT2_LATENCY_INPROCESS') == '1'
    linear = env.get('PROJECT2_LATENCY_LINEAR_IPC') == '1'
    wide = env.get('PROJECT2_LATENCY_W4_TABLE')
    hybrid = env.get('PROJECT2_LATENCY_HYBRID_TABLE')
    vision = env.get('PROJECT2_LATENCY_VISION_QMAX') == '1'
    receipts = {}
    identity = dict(executor=requested['distributed_executor_backend'],
        prefill_interval=requested['prefill_interval'], requested_mp_engine=requested['enable_mp_engine'],
        inprocess_requested=inprocess, linear_ipc_requested=linear, wide_w4_requested=bool(wide),
        hybrid_requested=bool(hybrid),
        vision_qmax_requested=vision, vision_qmax_validation_enabled=env.get('PROJECT2_LATENCY_VISION_QMAX_VALIDATE') == '1',
        profile_enabled=env.get('PROJECT2_LATENCY_PROFILE') == '1')
    vision_evidence = vision_qmax_audit(service,launch,verified_sources)
    identity['vision_qmax_status'] = vision_evidence['status'] if vision_evidence else 'disabled'
    if vision_evidence:
        receipts.update(vision_evidence['execution_receipt_sha256'])
    wide_evidence = wide_install_audit(service,launch,verified_sources) if wide else None
    hybrid_evidence = audit_hybrid_install(service,launch,verified_sources)
    if wide_evidence:
        identity['kernel_implementation'] = wide_evidence['implementation_identity']
        receipts[relative(service/'wide_launch_install.json')] = wide_evidence['install_receipt_sha256']
    elif hybrid_evidence:
        require(not (service/'wide_launch_install.json').exists() and not list(service.glob('wide_execution_*.json')),
                'Hybrid service unexpectedly emitted old wide adapter receipts')
        identity['kernel_implementation'] = hybrid_evidence['implementation_identity']
        receipts[relative(service/'hybrid_launch_install.json')] = hybrid_evidence['install_receipt_sha256']
    else:
        require(not (service/'wide_launch_install.json').exists() and not list(service.glob('wide_execution_*.json')),
                'Unexpected wide execution receipts without opt-in')
        require(Path(requested['custom_module_map']).name == 'navigation_repair_module_map.py', 'Unverified native model mapping')
        identity['kernel_implementation'] = dict(backend='native_frozen')
    path = service/'effective_inprocess_engine.json'
    if inprocess:
        module = 'navigation_latency_inprocess_20261120.py'
        require(module in verified_sources, 'In-process shim absent from launch source receipt')
        require(path.is_file(), 'In-process requested but observed engine receipt missing')
        value = read(path)
        require(value['native_engine_observed'] and value['only_bridge_setting_changed'] == 'enable_mp_engine',
                'In-process receipt does not prove native engine construction')
        require(value['engine_class'] == 'lmdeploy.pytorch.engine.engine.Engine' and
                value['executor_class'] == 'lmdeploy.pytorch.engine.executor.uni_executor.UniExecutor',
                'In-process engine/executor classes differ')
        require(value['builder_input_config'] == requested, 'HTTP builder input differs from requested launch config')
        delegated = dict(requested, enable_mp_engine=False)
        require(value['delegated_config'] == delegated, 'In-process bridge changed unexpected settings')
        require(value['effective_engine_config'] == delegated, 'Native effective engine differs from delegated settings')
        require(delegated['distributed_executor_backend'] == 'uni', 'In-process experiment requires UniExecutor')
        ownership = read(service.parent/'ownership.json')
        require(value['pid'] == ownership['service_pid'], 'In-process engine was not constructed in HTTP service process')
        snapshot = service.parent.parent/'source_snapshot'/module
        guarded = source_constant(snapshot if snapshot.is_file() else ROOT/'checks'/module, '_SOURCE_SHA256')
        require(value['source_sha256'] == guarded, 'In-process LMDeploy guard receipt differs from verified shim')
        guarded_package_sources(guarded)
        effective = value['effective_engine_config']
        identity['inprocess_status'] = 'observed_native_engine_in_http_process'
        receipts[relative(path)] = digest(path)
    else:
        require(not path.exists(), 'Unexpected in-process engine receipt without opt-in')
        identity['inprocess_status'] = 'native_mp_engine_launch_contract_no_override'
    identity['effective_mp_engine'] = effective['enable_mp_engine']
    if linear:
        module = 'navigation_latency_ipc_20261120.py'
        require(module in verified_sources, 'Linear IPC shim absent from launch source receipt')
        require(effective['distributed_executor_backend'] == 'mp', 'Linear IPC requested without MP executor; no executed transport optimization established')
        snapshot = service.parent.parent/'source_snapshot'/module
        guarded = source_constant(snapshot if snapshot.is_file() else ROOT/'checks'/module, 'SOURCE_SHA256')
        guarded_package_sources({'pytorch/engine/executor/mp_executor.py':guarded})
        module_sha = verified_sources[module]['sha256']

        def installed_function(value, qualname):
            while value:
                if (value.get('module') == 'navigation_latency_ipc_20261120' and
                    value.get('qualname') == qualname and Path(value.get('code_filename','')).name == module):
                    return True
                value = value.get('wrapped')
            return False

        installs, packs, receives = {}, [], []
        for receipt_path in sorted(service.glob('linear_ipc_*.json')):
            value = read(receipt_path)
            require(value['patch_module_sha256'] == module_sha and value['original_mp_source_sha256'] == guarded,
                    'Linear IPC runtime receipt source identity differs')
            require(value['enabled_env'] == '1' and isinstance(value['pid'],int), 'Invalid linear IPC runtime process receipt')
            require(receipt_path.name == f"linear_ipc_{value['kind']}_{value['pid']}.json", 'Linear IPC receipt filename identity differs')
            require(installed_function(value['installed_pack_function'], 'install.<locals>.pack_data') and
                    installed_function(value['installed_receive_function'], 'install.<locals>.receive_step1'),
                    'Linear IPC receipt does not prove installed patch functions')
            if value['kind'] == 'install':
                installs[value['pid']] = value
            elif value['kind'] in ('first_large_pack','first_large_receive'):
                require(value['completed'] and value['payload_bytes'] >= 1<<20 and
                        value['blocks'] == math.ceil(value['payload_bytes']/(1<<20)), 'Large IPC execution incomplete/invalid')
                expected_path = 'send' if value['kind'] == 'first_large_pack' else 'receive'
                require(value['path'] == expected_path, 'Large IPC execution direction differs')
                (packs if expected_path == 'send' else receives).append(value)
            else:
                raise ValueError('Unknown linear IPC execution receipt kind')
            receipts[relative(receipt_path)] = digest(receipt_path)
        require(packs and receives, 'Linear IPC requested but successful large pack/receive receipts missing')
        require(all(v['pid'] in installs for v in packs+receives), 'Linear IPC execution lacks same-process install receipt')
        require(any(p['pid'] != r['pid'] and p['payload_bytes'] == r['payload_bytes'] and p['blocks'] == r['blocks']
                    for p in packs for r in receives), 'No matching cross-process large linear IPC transfer observed')
        identity['linear_ipc_status'] = 'observed_large_pack_and_receive'
    else:
        require(not list(service.glob('linear_ipc_*.json')), 'Unexpected linear IPC execution receipts without opt-in')
        identity['linear_ipc_status'] = 'disabled'
    return dict(requested_engine=requested, effective_engine=effective,
                execution_identity=identity, execution_receipt_sha256=receipts, wide_evidence=wide_evidence,
                vision_qmax_evidence=vision_evidence, hybrid_evidence=hybrid_evidence)


def profile_audit(service, requests):
    files = sorted(service.glob('latency_frontend_*.jsonl'))
    if not files:
        return dict(status='not_recorded', reason='Profiling disabled or no profile logs', requests=[])
    events = []
    for path in files:
        for line in path.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            require(event['end'] >= event['start'], 'Negative profile interval')
            require(math.isclose(event['wall_ms'], 1000*(event['end']-event['start']), abs_tol=.01),
                    'Profile duration mismatch')
            events.append(event)
    events.sort(key=lambda e:e['start'])
    prompt = [e for e in events if e['event'] == 'frontend_prompt']
    worker = [e for e in events if e['event'] == 'worker_prepare_inputs']
    if len(prompt) != len(requests) or len(worker) != len(requests):
        return dict(status='unmapped', reason='Profile/request counts differ; no per-request attribution made',
                    counts=dict(requests=len(requests), frontend_prompt=len(prompt), worker_prepare=len(worker)),
                    events_by_type=dict(Counter(e['event'] for e in events)), requests=[])
    rows = []
    for index, (request, front, work) in enumerate(zip(requests, prompt, worker)):
        finish = prompt[index+1]['start'] if index+1 < len(prompt) else float('inf')
        if not (front['end'] <= work['start'] < finish and work['input_tokens'] == request['usage']['prompt_tokens']):
            return dict(status='unmapped', reason='Profile order or input token identity differs',
                        failed_index=index, requests=[])
        if request['spec']['phase'] != 'measured':
            continue
        selected = [e for e in events if front['start'] <= e['start'] < finish]
        pieces = defaultdict(list)
        for event in selected:
            pieces[event['event']].append(event['wall_ms'])
        rows.append(dict(index=index, sample=request['spec']['sample'], tokens=request['spec']['tokens'],
            frontend_prompt_ms=front['wall_ms'], frontend_to_worker_ms=1000*(work['start']-front['end']),
            worker_prepare_ms=work['wall_ms'], pixel_bytes=work['pixel_bytes'],
            event_duration_sums_ms={k:sum(v) for k,v in pieces.items()},
            event_counts={k:len(v) for k,v in pieces.items()}))
    return dict(status='mapped_by_serial_request_order', requests=rows,
        files_sha256={relative(path):digest(path) for path in files},
        caveat='CPU intervals may overlap/nest across processes; receive includes block waiting. Do not sum them as exclusive total latency or call them pure memcpy time.')


def audit_session(run, protocol, number, session):
    directory = run/f"{number:02d}_{session['label']}"
    require((directory/'result.json').is_file(), 'Session result missing')
    service = directory/'service'
    launch, command = read(service/'launch.json'), read(directory/'command.json')
    sources = source_audit(run, protocol, launch)
    contract = execution_contract(service, launch, sources)
    if contract['vision_qmax_evidence']:
        coverage = contract['vision_qmax_evidence']['first_grid_coverage']
        require(len(coverage) == 2 and {v['request_index'] for v in coverage} == {0,1},
                'Fixed-input QMAX run encountered an unexpected new grid outside excluded warmups')
    require(launch['group'] == session['group'], 'Group identity differs')
    method = session['group'].split('_')[0]
    require(launch['model_weights_sha256'] == FROZEN[method] == protocol['weight_sha256'][method],
            'Weight identity differs')
    require(launch['recipe']['activation_bits'] == 16 and
            launch['recipe']['weight_bits'] == (16 if method == 'fp16' else 4), 'Precision differs')
    engine = contract['effective_engine']
    for key, expected in dict(dtype='float16', tp=1, dp=1, session_len=4096, max_batch_size=1,
        max_prefill_token_num=3072, num_gpu_blocks=64, block_size=64, eager_mode=False,
        enable_prefix_caching=False, enable_microbatch=False, quant_policy=0).items():
        require(engine[key] == expected, 'Engine contract differs: '+key)
    if session.get('frozen_table'):
        path = ROOT/session['frozen_table']
        require(digest(path) == command['frozen_table_sha256'], 'Frozen launch table source changed')
    execution = read(service/'execution_audit.json')
    require(execution['first_native_forward_verified'], 'Native execution not verified')
    require(read(service/'startup_graph_capture.json')['actual_capture_observed'], 'Graph capture not verified')
    if method != 'fp16':
        receipt = execution['native_awq_receipt']
        require(receipt['verified'] and receipt['module_count'] == 144 and
                receipt['loaded_quantized_tensor_count'] == 756, 'Quantized execution not verified')
    repair = read(service/'repair_execution_audit.json')
    require(not repair['extension'] and repair['activation_bits'] == 16, 'Unexpected repair precision')
    launch_table_sha = None
    if method != 'fp16':
        table = repair['frozen_launch_table']
        require(table['frozen'] and table['actual_m_preserved'] and table['generic_m_specialization_disabled'] and
                not table['runtime_autotuning_allowed'] and not table['runtime_cold_compilation_allowed'],
                'Quantized launch configuration is not frozen')
        require(table['model_weights_sha256'] == FROZEN[method] and table['algorithm'] == method,
                'Executed launch table identity differs')
        launch_table_sha = sha_object(dict(shapes=table['shapes'], representatives=table['representatives'],
                                          max_rows=table['max_rows']))
        if session.get('frozen_table'):
            previous = read(ROOT/session['frozen_table'])
            previous = previous.get('frozen_launch_table', previous)
            require(table['selection_mode'] == 'replayed_frozen_table' and
                    table['frozen_table_source']['sha256'] == command['frozen_table_sha256'],
                    'Actual runtime did not replay the specified table')
            require(table['shapes'] == previous['shapes'] and table['representatives'] == previous['representatives'],
                    'Executed launch configurations differ from frozen table')
        if contract['wide_evidence']:
            require(table['selection_mode'] == 'replayed_frozen_table' and
                    table['frozen_table_source']['sha256'] == contract['wide_evidence']['numerical_probe']['baseline_table_sha256'],
                    'Wide service fallback differs from numerically checked baseline table')
    input_tensors = {}
    for length in (701, 2815):
        value = read(service/f'input_{length}.json')
        require(value['request_index'] == (0 if length == 701 else 1), 'Input SHA request identity differs')
        input_tensors[str(length)] = {k:value[k] for k in ('input_ids', 'pixel_values')}
    for sample in (0, 1):
        require(read(directory/f'natural_output_gate_{sample}.json')['passed'], 'Natural output gate failed')
    requests = read(directory/'requests.json')
    plan = read(service/'request_plan.json')
    require(len(requests) == len(plan) and plan[-1]['phase'] == 'flush', 'Incomplete requests/tail')
    require([r['spec'] for r in requests] == plan, 'Request/plan disagreement')
    require([p['sample'] for p in plan[:2]] == [0, 1] and
            all(p['phase'] == 'input_warmup' for p in plan[:2]), 'Missing input warmups')
    measured, evidence, wide_coverage, hybrid_coverage = [], {}, [], []
    for index, row in enumerate(requests[:-1]):
        require(row['index'] == index and row['http_mode'] == protocol['http_mode'], 'Request identity/mode differs')
        require(row['idle_seconds_outside_timer'] == protocol['idle_seconds_outside_timer'], 'Request idle differs')
        require(row['output_tokens_sha256'] == sha_object(row['output_tokens']), 'Output token SHA mismatch')
        require(len(row['output_tokens']) == row['usage']['completion_tokens'], 'Output token count differs')
        require(row['response']['usage'] == row['usage'], 'Raw response usage differs')
        require(row['response']['choices'][0]['message']['gen_tokens'] == row['output_tokens'], 'Raw response tokens differ')
        if row['spec']['phase'] == 'input_warmup':
            from navigation_repair_output_gate import inspect_answer
            gate = inspect_answer(row['response'])
            require(gate['passed'], 'Natural raw response failed pinned parser replay')
            saved = read(directory/f"natural_output_gate_{row['spec']['sample']}.json")
            require(canonical(gate) == canonical(saved), 'Natural output receipt differs from parser replay')
        value = read(service/f'timing_{index:03d}.json')
        require(value['index'] == index and value['plan'] == row['spec'], 'Timing/request mapping differs')
        steps = value['steps']; first, decode = steps[0], steps[1:]
        require(first['phase'] == 'prefill' and first['rows'] == row['usage']['prompt_tokens'], 'Prefill differs')
        require(first['rows'] == (701 if row['spec']['sample'] == 0 else 2815), 'Input length differs')
        require(decode and all(d['phase'] == 'decode' and d['rows'] == 1 and d['execution_mode'] == 'replay'
                              for d in decode), 'Measured/warm decode contains eager/capture')
        if contract['wide_evidence']:
            wide_coverage.append(wide_execution_audit(service,index,value,contract,execution['native_awq_receipt']))
        if contract['hybrid_evidence']:
            hybrid_coverage.append(audit_hybrid_request(service,index,value,contract,execution['native_awq_receipt']))
        if row['spec']['tokens']:
            require(row['usage']['completion_tokens'] == row['spec']['tokens'], 'Forced output length differs')
            require(row['response']['choices'][0]['finish_reason'] == 'length', 'Forced output ended early')
        if row['spec']['phase'] != 'measured':
            continue
        metrics = dict(http_s=row['http_total_s'], ttft_s=row['ttft_s'],
            stream_decode_ms=row['stream_decode_ms_per_token'],
            language_prefill_ms=first['timers']['language']['cuda_interval_ms'],
            vision_prefill_ms=first['timers']['vision']['cuda_interval_ms'],
            model_prefill_ms=first['timers']['model']['cuda_interval_ms'],
            decode_ms_per_actual_forward=statistics.mean(d['timers']['model']['cuda_interval_ms'] for d in decode))
        measured.append(dict(index=index, cell=f"{row['spec']['sample']}_{row['spec']['tokens']}",
            repeat=row['spec']['repeat'], metrics=metrics, actual_decode_forwards=len(decode),
            decode_model_total_ms=sum(d['timers']['model']['cuda_interval_ms'] for d in decode),
            output_tokens=row['output_tokens'], output_sha256=row['output_tokens_sha256'],
            semantic_payload_sha256=row['semantic_payload_sha256']))
    profile = profile_audit(service, requests)
    cells = {}
    for cell in protocol['cells']:
        chosen = [r for r in measured if r['cell'] == cell]
        require(len(chosen) == protocol['measured_repeats'], 'Incomplete measured cell: '+cell)
        require(len({r['semantic_payload_sha256'] for r in chosen}) == 1, 'Measured inputs changed within cell')
        values = {key:distribution([r['metrics'][key] for r in chosen]) if chosen[0]['metrics'][key] is not None
                  else None for key in chosen[0]['metrics']}
        matching_profile = [r for r in profile['requests'] if f"{r['sample']}_{r['tokens']}" == cell]
        psummary = None
        if matching_profile:
            psummary = {key:distribution([r[key] for r in matching_profile]) for key in (
                'frontend_prompt_ms', 'frontend_to_worker_ms', 'worker_prepare_ms', 'pixel_bytes')}
            event_names = set().union(*(r['event_duration_sums_ms'] for r in matching_profile))
            psummary['event_duration_sums_ms'] = {
                name:distribution([r['event_duration_sums_ms'].get(name, 0.) for r in matching_profile])
                for name in sorted(event_names)}
        cells[cell] = dict(n=len(chosen), metrics=values, actual_decode_forwards=sum(r['actual_decode_forwards'] for r in chosen),
            aggregate_decode_ms_per_actual_forward=sum(r['decode_model_total_ms'] for r in chosen)/
                                                  sum(r['actual_decode_forwards'] for r in chosen),
            output_sequences_identical=len({r['output_sha256'] for r in chosen}) == 1,
            output_tokens=chosen[0]['output_tokens'], output_sha256=chosen[0]['output_sha256'],
            semantic_payload_sha256=chosen[0]['semantic_payload_sha256'], profile=psummary)
    require(len(measured) == len(protocol['cells'])*protocol['measured_repeats'], 'Unexpected measured cells')
    if contract['wide_evidence']:
        require({v['prefill_m'] for v in wide_coverage} == {701,2815} and
                all(v['prefill_route'] == 'candidate' for v in wide_coverage),
                'Wide fixed-input mandatory prefill coverage incomplete or fallback occurred')
    if contract['hybrid_evidence']:
        require({v['prefill_m'] for v in hybrid_coverage} == {701,2815} and
                all(v['prefill_route_projection_counts'] == {'wide':144} for v in hybrid_coverage) and
                len({v['capture_signature_sha256'] for v in hybrid_coverage}) == 1,
                'Hybrid fixed-input prefill route/capture coverage changed or fallback occurred')
    # The original controller's summary is checked rather than trusted.
    controller = read(directory/'result.json')
    for cell, value in cells.items():
        original = controller['cells'][cell]
        for q in ('p50', 'p95'):
            require(math.isclose(value['metrics']['http_s'][q], original['metrics']['http_total_s'][q], rel_tol=1e-12),
                    'Recomputed HTTP percentile disagrees with controller')
        require(value['actual_decode_forwards'] == original['actual_decode_forwards'], 'Controller decode count differs')
    evidence_paths = [directory/'requests.json', directory/'command.json', directory/'ownership.json',
        directory/'cleanup.json', directory/'gpu_samples.json', service/'launch.json', service/'request_plan.json',
        service/'execution_audit.json', service/'repair_execution_audit.json', service/'startup_graph_capture.json']
    evidence_paths += [service/f'timing_{i:03d}.json' for i in range(len(requests)-1)]
    for path in evidence_paths:
        require(path.is_file(), 'Missing evidence: '+str(path))
        evidence[relative(path)] = digest(path)
    samples = read(directory/'gpu_samples.json')
    memory = [float(x['value'].split(',')[0]) for x in samples if 'value' in x]
    env = launch.get('candidate_environment', {})
    return dict(status='audited', id=relative(directory), study=relative(run), label=session['label'],
        group=session['group'], engine=engine, versions=launch['versions'],
        **contract,
        model_weights_sha256=launch['model_weights_sha256'], model_config_sha256=launch['model_config_sha256'],
        frozen_launch_configuration_sha256=launch_table_sha,
        candidate_environment=env, source_verification=sources, actual_input_tensors=input_tensors,
        http_mode=protocol['http_mode'], idle_seconds=protocol['idle_seconds_outside_timer'],
        warmups=protocol['warmups'], profile_enabled=env.get('PROJECT2_LATENCY_PROFILE') == '1',
        cells=cells, measurements=measured, profile=profile, evidence_sha256=evidence,
        wide_request_coverage=wide_coverage,
        hybrid_request_coverage=hybrid_coverage,
        sampled_peak_includes_vision_numeric_gate=contract['vision_qmax_evidence'] is not None,
        sampled_whole_card_peak_mib=max(memory) if memory else None,
        unknowns=[] if profile['status'] in ('mapped_by_serial_request_order','not_recorded') else [profile['reason']])


def comparable(a, b, ignore_executor=False, ignore_inprocess=False, ignore_linear_ipc=False,
               ignore_interval=False, allow_kernel_change=False, ignore_vision_qmax=False):
    reasons = []
    for key in ('versions', 'http_mode', 'idle_seconds', 'warmups', 'profile_enabled', 'actual_input_tensors'):
        if a[key] != b[key]: reasons.append(key+' differs')
    ea, eb = dict(a['engine']), dict(b['engine'])
    if ignore_executor:
        ea.pop('distributed_executor_backend', None); eb.pop('distributed_executor_backend', None)
    if ignore_inprocess:
        ea.pop('enable_mp_engine', None); eb.pop('enable_mp_engine', None)
    if ignore_interval:
        ea.pop('prefill_interval', None); eb.pop('prefill_interval', None)
    if allow_kernel_change:
        # Both mappings were independently checked by execution_contract.
        ea.pop('custom_module_map', None); eb.pop('custom_module_map', None)
    if ea != eb: reasons.append('engine configuration differs')
    ia, ib = dict(a['execution_identity']), dict(b['execution_identity'])
    if ignore_executor:
        ia.pop('executor', None); ib.pop('executor', None)
    if ignore_inprocess:
        for key in ('inprocess_requested','inprocess_status','effective_mp_engine'):
            ia.pop(key, None); ib.pop(key, None)
    if ignore_linear_ipc:
        for key in ('linear_ipc_requested','linear_ipc_status'):
            ia.pop(key, None); ib.pop(key, None)
    if ignore_interval:
        ia.pop('prefill_interval',None); ib.pop('prefill_interval',None)
    if ignore_vision_qmax:
        for key in ('vision_qmax_requested','vision_qmax_validation_enabled','vision_qmax_status'):
            ia.pop(key,None); ib.pop(key,None)
    if allow_kernel_change:
        for key in ('wide_w4_requested','hybrid_requested','kernel_implementation'):
            ia.pop(key,None); ib.pop(key,None)
    if ia != ib: reasons.append('observed execution identity differs')
    common = set(a['source_verification']) & set(b['source_verification'])
    for name in sorted(common):
        if a['source_verification'][name]['sha256'] != b['source_verification'][name]['sha256']:
            reasons.append('common source differs: '+name)
    # Other candidate switches must match; executor is handled by actual engine.
    excluded = {'PROJECT2_LATENCY_EXECUTOR', 'PROJECT2_LATENCY_PREFILL_INTERVAL', 'PROJECT2_LATENCY_PROFILE'}
    if ignore_inprocess: excluded.add('PROJECT2_LATENCY_INPROCESS')
    if ignore_linear_ipc: excluded.add('PROJECT2_LATENCY_LINEAR_IPC')
    if allow_kernel_change: excluded.update(('PROJECT2_LATENCY_W4_TABLE','PROJECT2_LATENCY_W4_PHASE','PROJECT2_LATENCY_HYBRID_TABLE'))
    if ignore_vision_qmax: excluded.update(('PROJECT2_LATENCY_VISION_QMAX','PROJECT2_LATENCY_VISION_QMAX_VALIDATE'))
    va = {k:v for k,v in a['candidate_environment'].items() if k not in excluded}
    vb = {k:v for k,v in b['candidate_environment'].items() if k not in excluded}
    if va != vb: reasons.append('other candidate switches differ')
    return reasons


def comparison_rows(sessions):
    rows = []
    for i, a in enumerate(sessions):
        for b in sessions[i+1:]:
            ignore_inprocess = ignore_linear = ignore_interval = ignore_vision = False
            allow_kernel_change = False
            changes = []
            if a['group'] == 'fp16' and b['group'] != 'fp16':
                baseline, candidate, kind, ignore = a, b, 'matched_quantization', False
                allow_kernel_change = True
            elif b['group'] == 'fp16' and a['group'] != 'fp16':
                baseline, candidate, kind, ignore = b, a, 'matched_quantization', False
                allow_kernel_change = True
            elif a['group'] == b['group']:
                baseline, candidate = a, b
                ea, eb = a['engine']['distributed_executor_backend'], b['engine']['distributed_executor_backend']
                ignore = ea != eb
                ignore_inprocess = a['execution_identity']['inprocess_requested'] != b['execution_identity']['inprocess_requested']
                ignore_linear = a['execution_identity']['linear_ipc_requested'] != b['execution_identity']['linear_ipc_requested']
                ignore_interval = a['engine']['prefill_interval'] != b['engine']['prefill_interval']
                ignore_vision = a['execution_identity'].get('vision_qmax_requested',False) != b['execution_identity'].get('vision_qmax_requested',False)
                allow_kernel_change = a['execution_identity']['kernel_implementation'] != b['execution_identity']['kernel_implementation']
                changes = [k for k,v in dict(executor=ignore,inprocess=ignore_inprocess,linear_ipc=ignore_linear,
                                            interval=ignore_interval,vision_qmax=ignore_vision,kernel_implementation=allow_kernel_change).items() if v]
                kind = ('same_model_'+changes[0]+'_change' if len(changes) == 1 else
                        'same_model_system_and_kernel_bundle' if allow_kernel_change else 'same_model_system_bundle') if changes else 'same_model_repetition'
                # Prefer original MP/native path as reference; otherwise preserve declared session order.
                if (ignore and ea == 'uni' and eb == 'mp') or (len(changes)==1 and
                    ((ignore_inprocess and a['execution_identity']['inprocess_requested']) or
                     (ignore_linear and a['execution_identity']['linear_ipc_requested']) or
                     (ignore_vision and a['execution_identity'].get('vision_qmax_requested',False)) or
                     (allow_kernel_change and
                      (2 if a.get('hybrid_evidence') else 1 if a.get('wide_evidence') else 0) >
                      (2 if b.get('hybrid_evidence') else 1 if b.get('wide_evidence') else 0)))):
                    baseline, candidate = b, a
            else:
                continue
            reasons = comparable(baseline, candidate, ignore_executor=ignore,
                                 ignore_inprocess=ignore_inprocess, ignore_linear_ipc=ignore_linear,
                                 ignore_interval=ignore_interval, allow_kernel_change=allow_kernel_change,
                                 ignore_vision_qmax=ignore_vision)
            if baseline['group'] == candidate['group'] and baseline['model_weights_sha256'] != candidate['model_weights_sha256']:
                reasons.append('same-group weight identity differs')
            if baseline['group'] == candidate['group'] and baseline['frozen_launch_configuration_sha256'] != candidate['frozen_launch_configuration_sha256']:
                reasons.append('frozen kernel launch configurations differ')
            for cell in sorted(set(baseline['cells']) & set(candidate['cells'])):
                av, bv = baseline['cells'][cell], candidate['cells'][cell]
                invalid = reasons + ([] if av['semantic_payload_sha256'] == bv['semantic_payload_sha256'] else ['request payload differs'])
                same_weight = baseline['model_weights_sha256'] == candidate['model_weights_sha256']
                token_equal = av['output_sha256'] == bv['output_sha256'] if same_weight else None
                repeats_stable = av['output_sequences_identical'] and bv['output_sequences_identical']
                if same_weight and (not token_equal or not repeats_stable):
                    invalid += ['same-weight output equality not established']
                ratio = bv['metrics']['http_s']['p50']/av['metrics']['http_s']['p50']
                rows.append(dict(kind=kind, baseline=baseline['id'], candidate=candidate['id'], cell=cell,
                    changed_system_or_kernel_factors=changes,
                    kernel_implementation=dict(baseline=baseline['execution_identity']['kernel_implementation'],
                                               candidate=candidate['execution_identity']['kernel_implementation']),
                    attribution=('Combined configuration effect; not a single-factor or pure-quantization attribution'
                                 if 'bundle' in kind else 'Matched execution contract; quantized kernel implementation is explicitly identified'
                                 if kind == 'matched_quantization' else 'Only listed configuration factors may differ'),
                    comparable=not invalid, reasons=invalid, exact_output_tokens_equal=token_equal,
                    both_repeats_identical=repeats_stable, http_p50_ratio=ratio,
                    http_p50_reduction_percent=100*(1-ratio),
                    http_p95_ratio=bv['metrics']['http_s']['p95']/av['metrics']['http_s']['p95'],
                    ratio_scope='Descriptive medians from independent sessions; no statistical significance claim. Do not pool repeated-round percentiles.'))
    return rows


def wide_output_gate(session, sessions, comparisons=None):
    """Wide numerical safety also needs same-weight native fixed output evidence."""
    if not session.get('wide_evidence'):
        return dict(status='not_applicable')
    comparisons = comparison_rows(sessions) if comparisons is None else comparisons
    native = {s['id'] for s in sessions if s['group'] == session['group'] and not s.get('wide_evidence') and not s.get('hybrid_evidence')}
    matches = [c for c in comparisons if c['kind'] == 'same_model_kernel_implementation_change' and
               c['candidate'] == session['id'] and c['baseline'] in native and c['comparable'] and
               c['exact_output_tokens_equal'] is True and c['both_repeats_identical']]
    covered = {c['cell'] for c in matches}
    missing = sorted(set(session['cells'])-covered)
    return dict(status='verified' if not missing else 'unverified_missing_matching_native_cells',
                missing_cells=missing, matching_native_comparisons=matches,
                scope='Same weight, actual inputs, common engine and original fallback launch table; wide kernel is the only implementation change')


def apply_wide_comparison_gates(sessions, comparisons):
    """Propagate every wide output gate before permitting any benefit claim.

    Build all gates from the unmodified comparisons first. One missing native
    cell invalidates every comparison involving that wide session, including
    FP16 comparisons whose weights differ and therefore cannot check tokens.
    Raw session timings remain unchanged; unvalidated ratios remain separately
    available as observations, never in the benefit fields.
    """
    blocked = {}
    for session in sessions:
        if session.get('wide_evidence'):
            gate = session.get('wide_native_output_gate',{})
            if gate.get('status') != 'verified':
                blocked[session['id']] = dict(status=gate.get('status','missing_gate'),
                                              missing_cells=gate.get('missing_cells',[]))
    for row in comparisons:
        failures = {sid:blocked[sid] for sid in (row['baseline'],row['candidate']) if sid in blocked}
        row['benefit_claim_allowed'] = row['comparable'] and not failures
        if not failures:
            continue
        row['comparable'] = False
        row['wide_output_gate_failures'] = failures
        for sid,gate in failures.items():
            reason = ('Wide native output gate not verified: '+sid+'; status='+gate['status']+
                      '; missing cells='+','.join(gate['missing_cells']))
            if reason not in row['reasons']:
                row['reasons'].append(reason)
        if 'unvalidated_latency_observation' not in row:
            row['unvalidated_latency_observation'] = {key:row[key] for key in
                ('http_p50_ratio','http_p50_reduction_percent','http_p95_ratio')}
        for key in ('http_p50_ratio','http_p50_reduction_percent','http_p95_ratio'):
            row[key] = None
        row['ratio_scope'] = 'Wide output gate is not verified. Retained unvalidated observations cannot support any speedup or quantization-benefit claim.'
    return comparisons


def audit_runs(paths):
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), studies=[], sessions=[], failures=[], unknowns=[],
        limitations=[
            'Fixed-input tests do not establish closed-loop navigation quality, success rate or generalization.',
            'All timing comparisons keep their HTTP mode; nonstream has no TTFT.',
            'Quantization comparisons require FP16 and quantized configurations with the same executor and engine settings.',
            'MP-to-uni improvements are execution-path changes, not solely quantization gains.',
            'Requested and effective engine configurations are separate. In-process HTTP uses a guarded private bridge and removes process isolation; concurrency responsiveness is untested.',
            'Linear IPC requires actual successful large-message send/receive receipts. Wide kernels require independent real packed-tensor hashes, numerical gates and all 144 projection routes per request.',
            'Wide kernels use saved exact-M configurations only; other navigation lengths retain native fallback. Microbenchmark numerical coverage uses layers0/18/35, not all layers.',
            'Hybrid uses GEMV only for the 36 M1 down projections, with 108 wide decode projections. Full-model approval requires all four cells to match a same-engine wide-all reference whose native output gate also passed.',
            'Vision QMAX requires bitwise main/three-DeepStack checks in excluded 2/9-image warmups. First-grid host receipts are not per-request GPU counters; sampled peak memory may include the diagnostic double-forward.',
            'Quantized/FP16 comparisons require the same vision QMAX state and numerical-validation flag; QMAX gains are reported separately as vision/system changes.',
            'CUDA event intervals can include dispatch gaps; decode uses actual forwards, including post-response tail work.',
            'Profile intervals overlap and include waiting; no exclusive overhead decomposition is claimed.',
            'Full checkpoint identities use prior SHA receipts; wide numerical tensors are additionally verified by independent mmap byte hashes. Audit only outside timed GPU runs.',
            'Small repeated fixed inputs and raw P50/P95 are descriptive; repetitions are not pooled or treated as significance tests.'])
    for run in paths:
        run = Path(run).resolve()
        require(run.is_relative_to(ROOT/'logs'), 'Study must be under project logs')
        state = read(run/'status.json')
        require(state['status'] in ('complete', 'failed'), 'Study is still running; audit only terminal runs')
        protocol = read(run/'protocol.json') if (run/'protocol.json').is_file() else None
        result['studies'].append(dict(directory=relative(run), status=state['status'],
            protocol_sha256=digest(run/'protocol.json') if protocol else None))
        if state['status'] == 'failed':
            result['failures'].append(dict(study=relative(run), failure=read(run/'failure.json')
                if (run/'failure.json').is_file() else 'Failure details absent'))
        if not protocol:
            continue
        for number, session in enumerate(protocol['sessions']):
            directory = run/f"{number:02d}_{session['label']}"
            if not (directory/'result.json').is_file():
                result['unknowns'].append(dict(session=relative(directory), reason='No completed session result'))
                continue
            try:
                result['sessions'].append(audit_session(run, protocol, number, session))
            except Exception as error:
                result['failures'].append(dict(session=relative(directory), audit_error=repr(error)))
    inputs = {}
    for session in result['sessions']:
        for length, value in session['actual_input_tensors'].items():
            if length in inputs and inputs[length] != value:
                result['failures'].append(dict(session=session['id'], audit_error='Cross-session input tensor SHA differs'))
            inputs.setdefault(length, value)
    result['actual_input_tensors'] = inputs
    result['comparisons'] = comparison_rows(result['sessions'])
    for session in result['sessions']:
        session['wide_native_output_gate'] = wide_output_gate(session,result['sessions'],result['comparisons'])
        if session['wide_native_output_gate']['status'].startswith('unverified'):
            result['unknowns'].append(dict(session=session['id'],reason='Wide candidate lacks same-engine native fixed-output equivalence',
                                           missing_cells=session['wide_native_output_gate']['missing_cells']))
    apply_wide_comparison_gates(result['sessions'],result['comparisons'])
    for session in result['sessions']:
        session['hybrid_fixed_output_gate'] = hybrid_output_gate(session,result['sessions'],result['comparisons'])
        if session['hybrid_fixed_output_gate']['status'].startswith('unverified'):
            result['unknowns'].append(dict(session=session['id'],reason='Hybrid candidate lacks complete matched wide-all/native output approval',
                missing_cells=session['hybrid_fixed_output_gate']['missing_cells']))
    apply_hybrid_output_gates(result['sessions'],result['comparisons'])
    result['status'] = 'audited_with_failures_or_incomplete_attempts' if result['failures'] or result['unknowns'] else 'audited'
    result['auditor_sha256'] = digest(__file__)
    result['audit_dependencies_sha256'] = {name:digest(ROOT/'checks'/name) for name in (
        'navigation_latency_w4_hybrid_audit_20261120.py','navigation_latency_w4_hybrid_contract_20261120.py')}
    return result


def fixed_evidence(paths, session):
    """Future navigation/replay binding API; audit terminal studies first.

    Run only in an idle audit window: inherited wide-reference verification can
    independently mmap selected packed tensors. This function starts no service.
    Old frozen runners are not patched to call this new API automatically.
    """
    def key(value):
        return dict(group=value['group'],service_entry=str(recorded_path(value['service_entry'])),
            model_name=value.get('model_name','navigation-repair-'+value['group']),
            env=value.get('env',{}),extra_args=value.get('extra_args',[]),
            frozen_table_sha256=digest(recorded_path(value['frozen_table'])) if value.get('frozen_table') else None)
    require(bool(session.get('env',{}).get('PROJECT2_LATENCY_HYBRID_TABLE')), 'This binding API is specifically for hybrid candidates')
    paths = [recorded_path(p) for p in paths]
    require(all(read(p/'status.json')['status'] == 'complete' for p in paths), 'Hybrid fixed binding requires completed studies')
    report = audit_runs(paths)
    by_id = {s['id']:s for s in report['sessions']}
    matches = []
    for run in paths:
        protocol = read(run/'protocol.json')
        for number,tested in enumerate(protocol['sessions']):
            if key(tested) != key(session): continue
            sid = relative(run/f"{number:02d}_{tested['label']}")
            require(sid in by_id, 'Matching hybrid session failed independent audit: '+sid)
            audited = by_id[sid]
            require(audited['hybrid_fixed_output_gate']['status'] == 'verified',
                    'Hybrid has not passed all four cells against an approved same-engine wide-all reference')
            for name,item in audited['source_verification'].items():
                require(digest(ROOT/'checks'/name) == item['sha256'], 'Hybrid source changed after fixed study: '+name)
            matches.append(dict(study=relative(run),fixed_session=sid,fixed_result_sha256=digest(ROOT/sid/'result.json'),
                engine=audited['engine'],requested_engine=audited['requested_engine'],execution_identity=audited['execution_identity'],
                model_weights_sha256=audited['model_weights_sha256'],actual_input_tensors=audited['actual_input_tensors'],
                frozen_launch_configuration_sha256=audited['frozen_launch_configuration_sha256'],
                source_sha256={n:v['sha256'] for n,v in audited['source_verification'].items()},
                wide_evidence=audited['wide_evidence'],hybrid_evidence=audited['hybrid_evidence'],
                vision_qmax_evidence=audited['vision_qmax_evidence'],hybrid_fixed_output_gate=audited['hybrid_fixed_output_gate'],
                auditor_sha256=report['auditor_sha256'],audit_dependencies_sha256=report['audit_dependencies_sha256'],
                scope='Eligible fixed-model hybrid evidence for a future separately authorized navigation/replay run; no navigation success or quality approval'))
    require(matches, 'No eligible matching hybrid four-cell fixed study')
    return matches[-1]


def markdown(result):
    lines = ['# 量化时延研究：独立固定输入审计', '',
        f"状态：`{result['status']}`。保留每轮原始测量；不平均不同轮的分位数。", '',
        '| 运行/配置 | 格子 | HTTP模式 | 执行器/MP引擎/间隔 | IPC补丁 | N | P50/s | P95/s | TTFT/s | 语言prefill/ms | 视觉/ms | Decode/ms每实际前向 |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for session in result['sessions']:
        for cell, value in session['cells'].items():
            m = value['metrics']; ttft = f"{m['ttft_s']['p50']:.4f}" if m['ttft_s'] else '未测'
            lines.append(f"| {session['study'].split('/')[-1]}/{session['label']} | {cell} | {session['http_mode']} | "
                f"{session['engine']['distributed_executor_backend']}/{session['engine']['enable_mp_engine']}/{session['engine']['prefill_interval']} | "
                f"{session['execution_identity']['linear_ipc_requested']} | {value['n']} | "
                f"{m['http_s']['p50']:.4f} | {m['http_s']['p95']:.4f} | {ttft} | {m['language_prefill_ms']['p50']:.2f} | "
                f"{m['vision_prefill_ms']['p50']:.2f} | {value['aggregate_decode_ms_per_actual_forward']:.3f} |")
    lines += ['', '格子：0=两图701输入token，1=九图2815输入token；后缀为强制返回token数。', '',
        '## 匹配比较', '', '| 比较类型 | 基线 → 候选 | 格子 | 可比较 | P50时延降低 | P95候选/基线 | 同权重输出一致 |',
        '|---|---|---|---|---:|---:|---|']
    comparison_limits = []
    for c in result['comparisons']:
        same = '不适用' if c['exact_output_tokens_equal'] is None else str(c['exact_output_tokens_equal'])
        reduction = f"{c['http_p50_reduction_percent']:.2f}%" if c['comparable'] else '不作归因'
        p95 = f"{c['http_p95_ratio']:.4f}" if c['http_p95_ratio'] is not None else '未通过门槛'
        lines.append(f"| {c['kind']} | {c['baseline'].split('/')[-1]} → {c['candidate'].split('/')[-1]} | {c['cell']} | "
                     f"{'是' if c['comparable'] else '否'} | {reduction} | {p95} | {same} |")
        if c['reasons']:
            comparison_limits.append(c['baseline'].split('/')[-1]+' → '+c['candidate'].split('/')[-1]+': '+'；'.join(c['reasons']))
    if comparison_limits:
        lines += ['', '不可直接比较的原因：', '']+['- '+x for x in comparison_limits]
    lines += ['', '## 前置路径诊断', '',
        '| 运行/配置 | 格子 | Frontend prompt/ms | Prompt结束→worker准备开始/ms | Worker准备/ms | 输入像素MiB |',
        '|---|---|---:|---:|---:|---:|']
    for s in result['sessions']:
        for cell, value in s['cells'].items():
            p = value['profile']
            if p:
                lines.append(f"| {s['study'].split('/')[-1]}/{s['label']} | {cell} | {p['frontend_prompt_ms']['p50']:.2f} | "
                    f"{p['frontend_to_worker_ms']['p50']:.2f} | {p['worker_prepare_ms']['p50']:.2f} | {p['pixel_bytes']['p50']/1048576:.2f} |")
    lines += ['', '这些CPU区间可能交叠；MP接收包括等待，不可相加成互斥总耗时。完整事件分段保存在JSON。', '', '## 限制、失败与未完成项', '']
    lines += ['- '+x for x in result['limitations']]
    lines += ['- '+canonical(x) for x in result['failures']+result['unknowns']]
    lines += ['', '本报告不宣称完整导航质量通过。各组实际输入tensor SHA、源码/launch SHA、原始请求和逐前向计时已在JSON逐项保留。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New JSON path; matching .md is also written')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and output.suffix == '.json', 'Output must be a project JSON path')
    require(not output.exists() and not output.with_suffix('.md').exists(), 'Output already exists; choose a fresh path')
    result = audit_runs(args.runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    output.with_suffix('.md').write_text(markdown(result), encoding='utf-8')
    print(canonical(dict(status=result['status'], audited_sessions=len(result['sessions']),
                         failures=len(result['failures']), unknowns=len(result['unknowns']), output=str(output))))


if __name__ == '__main__':
    main()
