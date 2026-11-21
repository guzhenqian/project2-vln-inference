"""CPU-only candidate shortlist for a new, separately validated W4 table.

Historical timings determine search order only. They never issue numerical,
repeatability, native-equivalence, or service-level approval.
"""
import json
import math
from pathlib import Path

SHAPES={'qkv':(2560,6144),'o':(4096,2560),'gate_up':(2560,19456),'down':(9728,2560)}
CONFIG_KEYS={'block_m','block_n','split_k','num_warps','pipeline_stages','compiler_stages','group_m'}


def native_config(table,projection):
    table=table.get('frozen_launch_table',table)
    k,n=SHAPES[projection]
    values=[v for v in table['shapes'] if (v['k'],v['n'])==(k,n)]
    if len(values)!=1:
        raise ValueError('Native layout is missing or duplicated')
    value=values[0];config=value['configs']['1']
    if value['split_k']!=max(1,k//4096):
        raise ValueError('Unexpected original split count')
    return dict(block_m=16,block_n=config['BLOCK_SIZE_N'],split_k=value['split_k'],
        num_warps=config['num_warps'],pipeline_stages=2,compiler_stages=config['num_stages'],
        group_m=config['GROUP_SIZE_M'])


def admissible(config,projection,policy):
    if policy not in ('native_split','split_le_2'):
        raise ValueError('Unknown decode policy')
    if set(config)!=CONFIG_KEYS or any(type(v) is not int for v in config.values()):
        return False
    if (config['block_m'] not in (16,32,64,128) or config['block_n'] not in (64,128,256)
        or config['num_warps'] not in (4,8) or config['pipeline_stages'] not in (2,3)
        or config['compiler_stages'] not in (2,3) or config['group_m'] not in (4,8)):
        return False
    original=max(1,SHAPES[projection][0]//4096)
    return config['split_k']==original if policy=='native_split' else config['split_k'] in (1,2)


def shortlist(attempts,baseline_table,projection,policy='native_split',limit=8):
    """Always include the original launch; rank eligible saved hints after it."""
    if not 1<=limit<=64:
        raise ValueError('Candidate limit must be1..64')
    k,n=SHAPES[projection]
    original=native_config(baseline_table,projection)
    result=[dict(config=original,origin='original_native_configuration',historical_hot_p50_ms=None)]
    seen={json.dumps(original,sort_keys=True)}
    eligible=[]
    for value in attempts:
        if (value.get('m'),value.get('k'),value.get('n'),value.get('projection'))!=(1,k,n,projection):
            continue
        if not value.get('numerical',{}).get('passed') or 'timing' not in value:
            continue
        config=value['config'];elapsed=value['timing']['cuda_p50_ms']
        if not admissible(config,projection,policy) or not math.isfinite(elapsed) or elapsed<=0:
            continue
        eligible.append(value)
    ranked=sorted(eligible,key=lambda v:v['timing']['cuda_p50_ms'])
    # With the broader policy, explicitly test both split counts when budget
    # permits instead of letting a hot-only ranking hide one complete family.
    coverage=[]
    if policy=='split_le_2':
        for split in (1,2):
            match=next((v for v in ranked if v['config']['split_k']==split),None)
            if match is not None:coverage.append(match)
    for value in coverage+ranked:
        key=json.dumps(value['config'],sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(dict(config=value['config'],origin='saved_hot_search_hint_only',
                               historical_hot_p50_ms=value['timing']['cuda_p50_ms']))
    return result[:limit]


def load_attempts(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
