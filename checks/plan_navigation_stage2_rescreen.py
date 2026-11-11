"""Record the conservative initial50 decision before running the second50."""
from datetime import datetime, timezone
from navigation_stage2_common import ROOT, RUN, GROUPS, read, write, digest

source = ROOT / 'reports/navigation_stage2_same50_20261109.json'
audit_path = ROOT / 'reports/navigation_stage2_delivery_audit_50_20261109.json'
audit = read(audit_path)
assert audit['status'] == 'passed' and audit['report_sha256'] == digest(source)
report = read(source)
assert report['status'] == 'complete' and len(report['metrics']) == 9
assert all(row['error_count'] == 0 for row in report['metrics'].values())
assert read(RUN / 'status.json')['status'] == 'initial50_complete'
assert not (RUN / 'screen_stage2').exists(), 'Decision must precede rescreening'
decision = dict(status='selected_for_cumulative100', created_utc=datetime.now(timezone.utc).isoformat(),
    initial50_report_sha256=digest(source), independent_audit_sha256=digest(audit_path),
    new_groups=list(GROUPS), reused_groups=['dart75', 'dart50'],
    additional_cases_per_new_group=50, cumulative_cases=100, extra_new_episodes=350,
    roster='Original frozen batch02; preserve batch01 results; no resampling',
    reasons={
        'fp16': 'Matched reference required for paired cumulative100 comparisons.',
        'awq_gptq': '35 vs36 successes on the same50; no stable basis for choosing between them from one extra success.',
        'token75': 'DART37, Sparse35, FastV32; quality gaps coexist with different measured latency. Keep FastV75 as the uncertain comparison rather than eliminating from SR alone.',
        'token50': 'FastV30 and Sparse29 vs DART27; lower P50/P95 creates a quality/latency tradeoff. Sparse50 collision rate requires closer examination.',
        'stability': 'All seven new groups completed with zero request/parser errors and passed full runtime audits.',
    },
    decision_scope='Conservative exploratory screening. No formal inferiority/noninferiority test or universal quality threshold was pre-registered; retain all viable/uncertain new configurations.',
    limits=['Single fixed development pool and one run per setting.',
            'No automatic expansion to150 or quantization-plus-token combinations.',
            'Rescreening continues during the day; no completion deadline promised.'])
write(RUN / 'rescreen_decision.json', decision)
write(ROOT / 'reports/navigation_stage2_rescreen_plan_20261110.json', decision)
print('RESCREEN_DECISION_RECORDED', ' '.join(GROUPS))
