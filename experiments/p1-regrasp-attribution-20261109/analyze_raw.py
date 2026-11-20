"""Locate global-regrasp failures in the original, unchanged 960 executions."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'outputs/p1-load-transfer-20261109'
OUT=Path(__file__).resolve().parent
STAGES=('inference_wait','global_lower_to_table','global_table_loaded','global_release',
    'global_table_settle','global_clear','global_hover','global_descend','global_close',
    'global_lift','global_lift_settle','final_stable_hold')


def statistics(values):
    a=np.asarray(values,dtype=float)
    return {'minimum':float(a.min()),'median':float(np.median(a)),'maximum':float(a.max())}


def measure(row):
    stages=defaultdict(list)
    for f in row['trace']:
        if f['phase'] in ('recovery','final_hold'):stages[f['stage']].append(f)
    targets=np.asarray(row['bootstrap']['registered_table_robot_targets'])
    data=[]
    for stage,frames in stages.items():
        f=frames[-1];e=f['evaluator'];handles=np.asarray(e['handle_positions']);eef=np.asarray(f['robot']['positions'])
        data.append({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
            'load':row['load'],'window':row['window'],'stage':stage,'samples':len(frames),
            'end_tilt_deg':e['tilt_deg'],'end_clearance_m':e['clearance'],
            'end_table_contact':e['table_contact'],'end_left_grasp':e['holds'][0],'end_right_grasp':e['holds'][1],
            'any_both_grasps':any(all(f['evaluator']['holds']) for f in frames),
            'end_max_handle_eef_error_m':float(np.linalg.norm(handles-eef,axis=1).max()),
            'end_left_handle_eef_error_m':float(np.linalg.norm(handles-eef,axis=1)[0]),
            'end_right_handle_eef_error_m':float(np.linalg.norm(handles-eef,axis=1)[1]),
            'end_max_handle_registered_target_error_m':float(np.linalg.norm(handles-targets,axis=1).max()),
            'end_handle_height_difference_m':e['handle_height_difference'],
            'end_pot_center':e['pot_center']})
    first=lambda predicate:next((f for f in row['trace'] if f['phase']=='recovery' and predicate(f)),None)
    table=first(lambda f:f['evaluator']['table_contact'])
    keeper=1 if row['scenario']=='left_loss' else 0 if row['scenario']=='right_loss' else None
    lost=first(lambda f:keeper is not None and not f['evaluator']['holds'][keeper])
    after_release=[f for f in row['trace'] if f['stage'] in ('global_close','global_lift','global_lift_settle','final_stable_hold')]
    both=[f for f in after_release if all(f['evaluator']['holds'])]
    result={k:row[k] for k in ('seed','scenario','method','load','window','success','task_complete','failure_reason',
        'table_setdown_verified','table_regrasp_and_lift_verified')}
    result.update(label=row['audit']['label'],first_table_stage=table['stage'] if table else None,
        first_table_seconds=table['time']-row['prefix_state']['time'] if table else None,
        first_keeper_loss_stage=lost['stage'] if lost else None,
        first_keeper_loss_seconds=lost['time']-row['prefix_state']['time'] if lost else None,
        keeper_lost_before_first_table=bool(lost and table and lost['time']<table['time']),
        any_both_grasps_after_reclose=bool(both),both_grasp_samples_after_reclose=len(both),
        first_both_grasps_stage=both[0]['stage'] if both else None)
    for stage in ('global_table_settle','global_descend','global_close'):
        d=next(x for x in data if x['stage']==stage)
        result[stage+'_tilt_deg']=d['end_tilt_deg']
        result[stage+'_max_handle_eef_error_m']=d['end_max_handle_eef_error_m']
        result[stage+'_max_handle_target_error_m']=d['end_max_handle_registered_target_error_m']
    return result,data


def main():
    source=STUDY/'trials.jsonl';cases=[];stages=[];total=0
    # Global calls selected by different policies share the same registered skill.
    # Retain them all, and use fixed-global rows for the principal paired diagnosis.
    for line in source.open():
        row=json.loads(line);total+=1
        if row['candidate']=='R_global':
            c,s=measure(row);cases.append(c);stages.extend(s)
    fixed=[c for c in cases if c['method']=='global']
    groups={}
    for load in ('native','density150'):
        for window in ('actual','1000ms'):
            for cohort,predicate in [('unilateral',lambda x:x['scenario'] in ('left_loss','right_loss')),
                                     ('two_grasp_controls',lambda x:x['scenario'] in ('coupled_height','sham'))]:
                rows=[c for c in fixed if c['load']==load and c['window']==window and predicate(c)]
                groups[f'{load}/{window}/{cohort}']={'episodes':len(rows),'independent_seeds':len({c['seed'] for c in rows}),
                    'task_complete':sum(c['task_complete'] for c in rows),'setdown_verified':sum(c['table_setdown_verified'] for c in rows),
                    'regrasp_lift_verified':sum(c['table_regrasp_and_lift_verified'] for c in rows),
                    'ever_both_grasps_after_reclose':sum(c['any_both_grasps_after_reclose'] for c in rows),
                    'keeper_lost_before_table':sum(c['keeper_lost_before_first_table'] for c in rows),
                    'first_keeper_loss_stage':dict(Counter(c['first_keeper_loss_stage'] for c in rows)),
                    **{key:statistics([c[key] for c in rows]) for key in (
                        'global_table_settle_tilt_deg','global_descend_tilt_deg','global_close_tilt_deg',
                        'global_close_max_handle_eef_error_m','global_table_settle_max_handle_target_error_m')}}
    assert total==960 and len(fixed)==320
    result={'original_episodes_read':total,'executed_global_calls':len(cases),'fixed_global_episodes':len(fixed),
        'groups':groups,'all_original_models_and_skills_unchanged':True,
        'interpretation':'Stage/pose measurements describe already executed trajectories. No new scoring threshold, policy success, physical intervention or fresh-seed claim is introduced.'}
    (OUT/'raw-attribution.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('global-cases',cases),('stage-endpoints',stages)]:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
