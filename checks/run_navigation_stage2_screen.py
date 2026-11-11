"""Run one authorized 50-case stage sequentially, preserving completed results."""
import argparse
import time
import traceback
from navigation_stage2_common import ROOT, RUN, GROUPS, read, write, digest
from run_navigation_stage2_validation import OwnedSession, status
from audit_navigation_stage2 import audit_service
from collect_navigation_stage2 import collect


def export_table(stage):
    count=stage*50
    rows={}
    for group in list(GROUPS)+['dart75','dart50']:
        path=(RUN/'reused_dart_v2'/f'{group}_{count}.json' if group.startswith('dart')
              else RUN/'results'/f'{group}_{count}.json')
        if path.exists():
            row=read(path)
            rows[group]={key:row[key] for key in ('case_count','reused','success_count','sr','spl',
                'actual_model_calls','request_median_s','request_p95_s','request_total_s',
                'sampled_whole_card_peak_mib','error_count','collisions')}
    result=dict(stage='initial50' if stage==1 else 'rescreen100',
        status='all_nine_available' if len(rows)==9 else 'partial_or_selected_candidates',
        metrics=rows,note='SparseVLM is pinned PriorTR v1.0 single-layer no-merge; DART timings reused with original recovery limitations.')
    write(RUN/f'comparison_{count}.json',result,exclusive=False)
    lines=[f'# 第二阶段：同批{count}条开发案例', '',
        '| 配置 | 成功数 | SR | SPL | P50/秒 | P95/秒 | 整卡峰值/MiB | 来源 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for group,row in rows.items():
        name='SparseVLM-v1 单层无合并 '+group[-2:]+'%' if group.startswith('sparse') else group
        lines.append(f"| {name} | {row['success_count']}/{count} | {row['sr']:.1%} | {row['spl']:.4f} | {row['request_median_s']:.3f} | {row['request_p95_s']:.3f} | {row['sampled_whole_card_peak_mib']:.0f} | {'复用' if row['reused'] else '新运行'} |")
    lines += ['', '原150条开发池内相同子集；仅真实HTTP请求，排除预热与队列零秒。显存为200ms整卡采样；DART75保留原恢复采样范围。详细审计和算法额外计算均计入耗时。',
        '', '本表随已完成且审计通过的组更新，缺少配置时不能视为最终九组结论。']
    (RUN/f'comparison_{count}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=int,choices=(1,2),required=True)
    parser.add_argument('--groups',nargs='+',choices=GROUPS,default=list(GROUPS))
    args=parser.parse_args()
    integrity=read(RUN/'input_integrity.json')
    assert integrity['status']=='passed'
    kernel=read(ROOT/'reports/navigation_gptq_kernel_stage2_20261109.json')
    assert kernel['status']=='passed'
    assert kernel['model_sha256']==read(ROOT/GROUPS['gptq']['model']/'conversion_audit.json')['sha256']
    # Gate initial screening on completed acceptance of every planned method.
    for group in GROUPS:
        evidence=read(RUN/'validation'/(group+'_passed.json'))
        assert evidence['passed']
        audit_service(ROOT/evidence['service'],group)
        assert digest(ROOT/evidence['single']/'runtime_audit.json')==evidence['single_audit_sha256']
    from export_navigation_stage2_integration import export
    export()
    results=RUN/'results'
    results.mkdir(exist_ok=True)
    stage_dir=RUN/f'screen_stage{args.stage}'
    stage_dir.mkdir(exist_ok=True)
    completed=[]
    export_table(args.stage)
    for group in args.groups:
        result_path=results/f'{group}_{args.stage*50}.json'
        if result_path.exists():
            assert read(result_path)['status']=='complete'
            completed.append(group)
            continue
        previous_attempts=list(stage_dir.glob(group+'_*'))
        if previous_attempts:
            raise RuntimeError('Unfinished prior attempt exists; inspect and resume saved cases instead of silently rerunning: '+str(previous_attempts))
        previous=[]
        if args.stage==2:
            first=read(results/f'{group}_50.json')
            previous=[ROOT/path for path in first['batch_directories']]
        directory=stage_dir/(group+'_'+time.strftime('%Y%m%dT%H%M%S'))
        directory.mkdir()
        owner=OwnedSession(group,directory)
        try:
            status(status='screening',formal_started=True,screening_stage=args.stage,
                active_group=group,completed_groups_current_stage=completed)
            owner.start()
            folder=owner.episode_batch(args.stage)
            directories=previous+[folder]
            result=collect(group,directories,args.stage*50)
            result['batch_directories']=[path.relative_to(ROOT).as_posix() for path in directories]
            result['runtime_audit_sha256']={str(path.relative_to(ROOT)):digest(path/'runtime_audit.json') for path in directories}
            result['controller_sha256']=digest(__file__)
            result['execution_contract_sha256']=digest(ROOT/'setup/navigation-stage2-execution-20261109.json')
            result['input_integrity_sha256']=digest(RUN/'input_integrity.json')
            write(result_path,result)
            completed.append(group)
            export_table(args.stage)
            status(completed_groups_current_stage=completed)
            print('STAGE2_GROUP_COMPLETE',group,args.stage*50,result['success_count'],flush=True)
        except BaseException as error:
            failure=dict(group=group,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
            write(directory/'failure.json',failure)
            status(status='screening_failed',failure=failure)
            raise
        finally:
            owner.close()
    status(status='initial50_complete' if args.stage==1 else 'rescreen100_complete',
        completed_groups_current_stage=completed,active_group=None)


if __name__=='__main__':
    main()
