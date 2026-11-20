"""Real unchosen suffixes for frozen independent-policy decisions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .duration_continuation import run_continuation
from .mid_recovery_data import read,outcome


def execute(source):
    from .l0 import _import_robosuite
    _import_robosuite(True)
    count=sum(f['stage']=='local_relevel' for f in source['trace'])
    generated=[]
    if count<60:
        local=run_continuation((source,'continue'),transition='defer_rotation',account_control_latency=True,boundary=f'relevel{count}')
        if local['diagnostic_errors']:raise RuntimeError(local['diagnostic_errors'])
        generated.append(local)
    else:local=source
    branches={'LOCAL60':local}
    for step in (30,40):
        if source['candidate']=='R_global' and count==step:glob=source
        else:
            glob=run_continuation((local,'R_global'),transition='defer_rotation',account_control_latency=True,boundary=f'relevel{step}')
            if glob['diagnostic_errors']:raise RuntimeError(glob['diagnostic_errors'])
            generated.append(glob)
        branches[f'GLOBAL{step}']=glob
    key={k:source[k] for k in ('seed','scenario','method')}
    for r in generated:r['policy_source']=key
    case={**key,'labels':{k:outcome(v) for k,v in branches.items()},
        'branch_seconds':{k:v['total_post_fault_seconds'] for k,v in branches.items()},
        'actual_label':outcome(source),'actual_seconds':source['total_post_fault_seconds'],
        'actual_branch':f'GLOBAL{count}' if source['candidate']=='R_global' else 'reject' if source.get('abstained') else 'LOCAL60',
        'windows':source.get('mid_selection',{}).get('windows',[])}
    return generated,case


def analyze(cases):
    reports={}
    for method in sorted({r['method'] for r in cases}):
        rows=[r for r in cases if r['method']==method]
        available=[r for r in rows if 'A' in r['labels'].values()]
        needed=[r for r in rows if r['labels']['LOCAL60']!='A' and any(r['labels'][k]=='A' for k in ('GLOBAL30','GLOBAL40'))]
        false=[];admitted=[]
        for r in rows:
            for w in r['windows']:
                if not w['forecast_ready'] or not w['continuation_matches_forecast']:continue
                for p in w['decision'].get('records',[]):
                    if not p.get('feasible_now',True):continue
                    key='LOCAL60' if p['key']=='local' else 'GLOBAL40' if p['key']=='defer' else 'GLOBAL'+str(p.get('activation_step',w['decision_step']))
                    if p['accepted']:
                        admitted.append((r['seed'],r['scenario'],w['decision_step'],p['key']))
                        if r['labels'][key]!='A':false.append([r['seed'],r['scenario'],w['decision_step'],p['key'],r['labels'][key]])
        reports[method]={'N60_states':len(rows),'recoverable_states':len(available),
            'safe_coverage':sum(r['actual_label']=='A' for r in available),
            'needed_upgrade_states':len(needed),'needed_upgrade_misses':sum(r['actual_label']!='A' for r in needed),
            'necessary_upgrades_safely_executed':sum(r['actual_label']=='A' and r['actual_branch'].startswith('GLOBAL') for r in needed),
            'unnecessary_upgrades':sum(r['actual_branch'].startswith('GLOBAL') and r['labels']['LOCAL60']=='A' for r in rows),
            'candidate_admissions':len(admitted),'false_admitted_candidates':len(false),'false_admission_cases':false,
            'executed_model_false_admissions':sum(r['actual_label']!='A' and bool(r['windows']) and
                r['windows'][-1]['decision'].get('model_admitted',False) and not r['windows'][-1]['baseline_fallback'] for r in rows),
            'unsafe_actual_continues':sum(r['actual_branch']=='LOCAL60' and r['actual_label']!='A' for r in rows),
            'no_safe_states':sum('A' not in r['labels'].values() for r in rows),
            'no_safe_rejections':sum('A' not in r['labels'].values() and r['actual_branch']=='reject' for r in rows)}
    return reports


def second_window_effect(cases):
    # Fix the actual two-window policy's prefix and its first model decision.
    # The alternative is physically executed with ALL recorded waits retained.
    rows=[]
    for r in cases:
        if r['method'] not in ('mid_two','selective') or len(r['windows'])!=2:continue
        action=r['windows'][0].get('selected_branch',r['windows'][0]['decision']['action'])
        first={'continue':'LOCAL60','defer':'GLOBAL40'}.get(action)
        if first is None:continue
        rows.append({'seed':r['seed'],'scenario':r['scenario'],'first_plan':first,
            'first_plan_label':r['labels'][first],'two_window_label':r['actual_label'],
            'first_plan_seconds':r['branch_seconds'][first],'two_window_seconds':r['actual_seconds']})
    return {'states_with_two_actual_queries':len(rows),
        'safe_first_plan':sum(r['first_plan_label']=='A' for r in rows),
        'safe_two_window':sum(r['two_window_label']=='A' for r in rows),
        'safely_rescued':sum(r['first_plan_label']!='A' and r['two_window_label']=='A' for r in rows),
        'safe_coverage_lost':sum(r['first_plan_label']=='A' and r['two_window_label']!='A' for r in rows),
        'cases':rows,'scope':'Same physical prefix and recorded waits, including second-query waits; compares executing the first forecasted plan against the actual revised decision. Native three-method costs remain separately reported.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.jsonl').exists():raise ValueError('counterfactual collection exists')
    sources=[r for r in read(a.source) if sum(f['stage']=='local_relevel' for f in r['trace']) in (30,40,60)
        and r.get('local_steps',{}).get('local_relevel_steps')==60]
    cases=[];episodes=0
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (out/'trials.jsonl').open('w') as f:
            for generated,case in pool.map(execute,sources):
                for row in generated:f.write(json.dumps(row)+'\n');episodes+=1
                f.flush();cases.append(case)
                print(json.dumps({k:case[k] for k in ('seed','scenario','method','labels','actual_branch')}),flush=True)
    (out/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
    report={'frozen_test_diagnosis':True,'used_for_training':False,'new_physical_suffixes':episodes,'states':len(cases),'methods':analyze(cases),
        'matched_second_window_effect':second_window_effect(cases)}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
