"""Seal the completed navigation investigation and selected candidate without rehashing huge weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rl-suite',required=True,type=Path)
args=parser.parse_args()
result=json.loads((ROOT/'reports/navigation_sr_final_20261030.json').read_text(encoding='utf8'))
prior=json.loads((ROOT/'reports/closed_loop_baseline_v1_20261030_freeze.json').read_text(encoding='utf8'))
for name,record in prior['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record['sha256'],name
for group in ['compare','confirmation']:
    provenance=json.loads((args.rl_suite/group/'provenance.json').read_text(encoding='utf8'))
    for name,digest in provenance['scripts'].items():
        assert hashlib.sha256((ROOT/'checks'/name).read_bytes()).hexdigest()==digest,name
if result['rl_candidate_adopted']:
    candidate={'version':'navigation_rl4b_bf16_candidate_v1','status':'small-sample quality baseline candidate; not full benchmark acceptance',
        'model_repo':'honglyhly/SimpleVLN_Qwen3VL_4B_RL','revision':'4078dab77f3c93a0711b57549ac54a59c1c1e1ea',
        'runtime_precision':'BF16','server_script':'scripts/start_simplevln_rl_4b.sh','server_model_name':'SimpleVLN-Qwen3VL-4B-RL',
        'policy_source':'checks/simplevln_policy.py','run_script':'scripts/run_simplevln_rl_diagnostic.sh',
        'groups':{'A_diagnostic':'data/baseline/simplevln_4b_rl_compare/manifest.json',
                  'B_confirmation':'data/baseline/simplevln_4b_rl_confirmation/manifest.json'},
        'reference_runs':{'A_diagnostic':result['rl_same_tasks_h400']['path'],'B_confirmation':result['rl_fresh_scenes_h400']['path']},
        'report':'reports/navigation_sr_final_20261030.md',
        'next_stage':'freeze calibration/evaluation scene separation before quantization or token compression; keep all policy inputs/settings fixed for comparisons',
        'limitations':['six tasks per group','one stochastic seeded rollout per task','Habitat0.1.7/LMDeploy differs from author evaluator','not all failures solved']}
    (ROOT/'setup/navigation-baseline-profile.json').write_text(json.dumps(candidate,indent=2),encoding='utf8')
folders=[args.rl_suite.resolve(),ROOT/'logs/simplevln_diagnostic_20261030T090743Z_47822',
         ROOT/'logs/simplevln_holdout_h400_20261030T092900Z',
         ROOT/'tmp/habitat_setup/metric_controls_holdout_20261030',ROOT/'tmp/habitat_setup/metric_controls_confirmation_20261030']
files=[]
for folder in folders:files.extend(p for p in folder.rglob('*') if p.is_file())
files.extend((ROOT/'checks').glob('*navigation*.py'))
files.extend((ROOT/'checks').glob('*simplevln*.py'))
files.extend((ROOT/'checks').glob('*holdout_controls.py'))
files.extend((ROOT/'checks').glob('*confirmation_controls.py'))
files.extend((ROOT/'scripts').glob('*simplevln*.sh'))
files.extend((ROOT/'scripts').glob('*navigation*.sh'))
for folder in (ROOT/'data/baseline').glob('*'):
    if any(word in folder.name for word in ['simplevln','memory_policy','rgb_v1_holdout']):files.extend(folder.rglob('manifest.json'))
for name in ['SimpleVLN_Qwen3VL_4B_SFT','SimpleVLN_Qwen3VL_4B_RL']:
    folder=ROOT/'models'/name
    files.extend(folder/name for name in ['source_manifest.json','config.json','tokenizer_config.json','processor_config.json'])
    files.append(ROOT/'models'/(name+'_lmdeploy')/'serving_adapter.json')
for pattern in ['navigation_sr_*.md','navigation_sr_*.json','navigation_sr_*.png','navigation_horizon_*.md','navigation_horizon_*.json']:
    files.extend((ROOT/'reports').glob(pattern))
files.extend((ROOT/'tmp/habitat_setup/navigation_source_review').glob('*comparison.json'))
files.append(ROOT/'setup/navigation-baseline-profile.json')
files.append(Path(__file__))
records={}
for path in sorted(set(p.resolve() for p in files if p.is_file())):
    records[path.relative_to(ROOT).as_posix()]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
sealed={'date':'2026-10-30','purpose':'completed quality diagnosis, all wins/losses retained; candidate profile frozen',
    'original_v1_files_verified_unchanged':len(prior['files']),
    'weight_verification':'full SHA256 against HF LFS completed at download; exact tensor key/shape and runtime image/token comparison also recorded; weights not redundantly re-read here',
    'files':records}
(ROOT/'reports/navigation_quality_20261030_freeze.json').write_text(json.dumps(sealed,indent=2),encoding='utf8')
print('NAVIGATION_QUALITY_FROZEN',len(records),'original_v1_unchanged',len(prior['files']))
