"""Post-hoc delay-removal diagnostic; does not replace the frozen policy test."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]))
from auditembodied.l4_lift import LiftConfig,_job
from auditembodied.duration_pilot import STATE_KEYS,prefix_frames
from auditembodied.duration_acquire import before_relevel
from auditembodied.duration_closed_loop import trial_summary


def searches(row):
    return sum(f['stage']=='local_arc_approach' and (i==0 or row['trace'][i-1]['stage']!='local_arc_approach') for i,f in enumerate(row['trace']))


def main():
    if (root/'no-wait-diagnostic-trials.jsonl').exists():raise ValueError('diagnostic exists')
    source=[json.loads(s) for s in (root/'trials.jsonl').read_text().splitlines()]
    policy=[r for r in source if r['method']=='vlm_with_wait']
    baseline={(r['seed'],r['scenario']):r for r in source if r['method']=='oracle_role_fixed60'}
    jobs=[(r['seed'],r['scenario'],r['candidate'],LiftConfig(local_relevel_steps=r.get('local_steps',{}).get('local_relevel_steps',40))) for r in policy]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(_job,jobs))
    cases=[]
    for old,new in zip(policy,rows):
        new['method']='selected_action_no_wait_diagnostic'
        b=baseline[old['seed'],old['scenario']]
        cases.append({'seed':old['seed'],'scenario':old['scenario'],'candidate':old['candidate'],
            'relevel_steps':new.get('local_steps',{}).get('local_relevel_steps'),
            'prefix_identical':all(old[k]==new[k] for k in STATE_KEYS) and prefix_frames(old)==prefix_frames(new),
            'no_wait_pre_relevel_equals_fixed60':before_relevel(new)==before_relevel(b),
            'with_wait_success':old['success'],'with_wait_violation':old['protection_violation'],
            'without_wait_success':new['success'],'without_wait_violation':new['protection_violation'],
            'with_wait_total_seconds':old.get('total_post_fault_seconds'),'without_wait_total_seconds':new.get('total_post_fault_seconds'),
            'with_wait_searches':searches(old),'without_wait_searches':searches(new),
            'extra_skill_seconds':old.get('skill_seconds',0)-new.get('skill_seconds',0)})
    (root/'no-wait-diagnostic-trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report={'posthoc_diagnostic':True,'new_policy_inference':False,'model_or_threshold_changed':False,
        'description':'Replay exactly the selected candidate and duration at the same seed/fault prefix, with inference wait removed. This is not the measured online policy result.',
        'summary':trial_summary(rows),'all_prefixes_identical':all(c['prefix_identical'] for c in cases),
        'all_no_wait_pre_relevel_match_fixed60':all(c['no_wait_pre_relevel_equals_fixed60'] for c in cases),'cases':cases}
    (root/'no-wait-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
