"""NumPy-only first-close error identity and whole-cohort stage descriptions."""
from collections import Counter,defaultdict
import numpy as np


def interval(values):
    x=np.asarray(values,float)
    draws=np.random.default_rng(20261117).integers(0,len(x),(2000,len(x)))
    return {'mean':float(x.mean()),'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),'seeds':len(x)}


def score(row,previous):
    trace=row['trace'];log=row['global_execution'];attempts=log['pickup_attempts']
    closing=next((a['close_time'] for a in attempts if a.get('close_time') is not None),None)
    valid=[d for d in log['observations'] if d['estimate']['valid'] and closing is not None
           and d['observed_time']<closing+1e-8]
    frames=[f for f in trace if f['phase'] in ('recovery','final_hold')]
    placed=any(f['stage']=='global_table_settle' and f['evaluator']['table_contact']
               and not any(f['evaluator']['holds']) for f in frames)
    lifted=any(f['stage'] in ('global_lift','global_lift_settle') and all(f['evaluator']['holds'])
               and f['evaluator']['clearance']>=.10 for f in frames)
    assert placed==row['table_setdown_verified'] and (placed and lifted)==row['table_cycle_verified']
    closed=[f for f in frames if f['stage']=='global_close']
    ever_dual=any(all(f['evaluator']['holds']) for f in frames if f['stage'] in
        ('global_close','global_recenter_on_table','global_lift','global_lift_settle','final_stable_hold'))
    record={k:row[k] for k in ('seed','scenario','method','hold_mode','wait_steps')}
    record.update(C=bool(row['task_complete']),Y500=bool(previous['native_Y']),retry=len(attempts)>1,
        stages={'placed':placed,'preclose_target':bool(valid),
            'approach_attempted':any(f['stage']=='global_descend' for f in frames),
            'close_attempted':bool(closed),'first_dual':bool(previous['first_close_dual']),
            'ever_dual':ever_dual,'lifted':lifted,'terminal':bool(row['task_complete'])},arms=[])
    assert record['C']==previous['C']
    terminal=[f['evaluator'] for f in frames if f['stage']=='final_stable_hold']
    conditions={
        'grasp':bool(terminal and all(all(e['holds']) for e in terminal)),
        'relative_motion':bool(terminal and all(max(e['relative_speeds'])<=.20 for e in terminal)),
        'clearance':bool(terminal and all(e['clearance']>=.10 for e in terminal)),
        'balance':bool(terminal and all(e['tilt_deg']<=15. and e['handle_height_difference']<=.03 for e in terminal))}
    record['terminal_unmet']=[k for k,v in conditions.items() if not v]
    if record['C']:signature='complete'
    elif not placed:signature='placement_interface'
    elif not valid:signature='visual_target_or_attempt_unavailable'
    elif not closed:signature='no_close_attempt'
    elif not ever_dual:signature='no_dual_grasp'
    elif not lifted:signature='lift_interface'
    elif not conditions['grasp']:signature='final_grasp_retention'
    elif not conditions['relative_motion']:signature='final_relative_motion'
    elif not conditions['clearance']:signature='final_clearance'
    elif not conditions['balance']:signature='final_balance'
    else:signature='terminal_other'
    record['failure_signature']=signature
    if not valid:return record
    d=valid[-1];estimate=d['estimate'];port=d['input']
    if row['method']=='closed_loop':
        local=(np.asarray(port['reference_targets'])-port['reference_center'])@np.asarray(port['reference_rotation'])
        target=np.asarray(estimate['center'])+local@np.asarray(estimate['rotation']).T
        assert np.allclose(target,estimate['targets'],rtol=0,atol=1e-9)
    else:target=np.asarray(estimate['targets'])
    index=next(i for i,f in enumerate(trace) if abs(f['time']-closing)<1e-8)
    f=trace[index];before=trace[index-1]
    assert f['stage']=='global_close'
    pos=np.asarray(f['robot']['positions']);handle=np.asarray(f['evaluator']['handle_positions'])
    previous_pos=np.asarray(before['robot']['positions'])
    action=np.asarray(f['robot']['action']).reshape(2,7)
    commanded=np.clip(.8*(target-previous_pos)/.05,-1.,1.)
    assert np.allclose(commanded,action[:,:3],rtol=0,atol=1e-8),'Actual close target does not match logged RGB target'
    h0=np.asarray(d['observation_truth']['handle_positions'])
    target_error=target-h0;drift=handle-h0;tracking=pos-target;final=pos-handle
    residual=target_error-drift+tracking-final
    assert np.max(np.abs(residual))<1e-12
    keeper=1 if row['scenario']=='left_loss' else 0 if row['scenario']=='right_loss' else None
    for arm in range(2):
        norm=np.linalg.norm(final[arm]);direction=final[arm]/max(norm,1e-15)
        components=np.array([target_error[arm],-drift[arm],tracking[arm]])*1000
        projections=components@direction
        assert abs(projections.sum()-norm*1000)<1e-8
        desired=np.asarray(estimate['rotation_targets'])[arm]
        current=np.asarray(f['robot']['rotations'])[arm]
        angle=float(np.degrees(np.arccos(np.clip((np.trace(desired@current.T)-1)/2,-1,1))))
        segment=[a for a in trace if d['observed_time']<a['time']<=closing+1e-8]
        saturated=np.array([np.max(np.abs(np.asarray(a['robot']['action']).reshape(2,7)[arm,:3]))>=.999999 for a in segment])
        record['arms'].append({'arm':arm,'role':'healthy' if keeper is None else 'keeper' if arm==keeper else 'free',
            'target_vector_mm':components[0].tolist(),'negative_drift_vector_mm':components[1].tolist(),
            'tracking_vector_mm':components[2].tolist(),'closure_vector_mm':(final[arm]*1000).tolist(),
            'target_norm_mm':float(np.linalg.norm(components[0])),
            'drift_norm_mm':float(np.linalg.norm(components[1])),
            'tracking_norm_mm':float(np.linalg.norm(components[2])),
            'closure_norm_mm':float(norm*1000),'signed_contributions_mm':projections.tolist(),
            'wrist_error_deg':angle,'translation_saturation_fraction':float(saturated.mean()),
            'registered_grasp_at_close':bool(f['evaluator']['holds'][arm])})
    record.update(observation_to_close_seconds=float(closing-d['observed_time']),
        charged_preclose_service_seconds=float(d['service_wait_steps']*.05),
        identity_max_residual_m=float(np.max(abs(residual))),actual_target_verified=True)
    return record


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['method'],r['hold_mode'],r['wait_steps'],r['scenario']=='sham'].append(r)
    report={'records':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
        'physical_executions_added':0,'actual_close_targets_verified':sum(r.get('actual_target_verified',False) for r in rows),
        'maximum_vector_identity_residual_m':max(r.get('identity_max_residual_m',0) for r in rows),
        'conditions':[]}
    for (method,hold,wait,healthy),g in sorted(groups.items()):
        group={'method':method,'hold_mode':hold,'wait_steps':wait,'healthy':healthy,
            'records':len(g),'C':sum(r['C'] for r in g),'Y500':sum(r['Y500'] for r in g),
            'stages':{k:sum(r['stages'][k] for r in g) for k in g[0]['stages']},
            'failure_signatures':dict(Counter(r['failure_signature'] for r in g)),
            'terminal_unmet':dict(Counter(k for r in g for k in r['terminal_unmet'])),
            'roles':{}}
        for role in ('keeper','free','healthy'):
            pairs=[(r,a) for r in g for a in r['arms'] if a['role']==role]
            if not pairs:continue
            values={'arm_records':len(pairs),'registered_grasps':sum(a['registered_grasp_at_close'] for r,a in pairs)}
            for key in ('target_norm_mm','drift_norm_mm','tracking_norm_mm','closure_norm_mm',
                        'wrist_error_deg','translation_saturation_fraction'):
                seeds=sorted({r['seed'] for r,a in pairs})
                values[key]=interval([np.mean([a[key] for r,a in pairs if r['seed']==s]) for s in seeds])
            values['signed_contributions_mm']=[float(np.mean([a['signed_contributions_mm'][j] for r,a in pairs])) for j in range(3)]
            group['roles'][role]=values
        report['conditions'].append(group)
    return report
