"""NumPy-only paired holding contrasts and evaluator-side stage diagnostics."""
from collections import defaultdict
import numpy as np
import holding_components_audit as components

HOLDS=components.HOLDS
METHODS=('closed_loop','visual_servo')
FIELDS=('C','Y','native_Y','total','pickup_attempts','first_close_dual')


def check_pairing(rows):
    prefixes=defaultdict(list);ends=defaultdict(list)
    for r in rows:
        prefixes[r['seed'],r['scenario']].append({k:r[k] for k in
            ('initialization_state','preclose_state','initialization_end_state','prefix_state')})
        ends[r['seed'],r['scenario'],r['hold_mode'],r['wait_steps']].append(r['post_wait_state'])
    for key,group in prefixes.items():assert all(x==group[0] for x in group),key
    for key,group in ends.items():assert all(x==group[0] for x in group),key
    return {'matched_prehold_groups':len(prefixes),
            'matched_posthold_method_pairs':sum(len(g)==2 for g in ends.values())}


def interval(values):
    x=np.asarray(values,dtype=float)
    draw=np.random.default_rng(20261115).integers(0,len(x),(2000,len(x)))
    return {'estimate':float(x.mean()),'ci95':np.quantile(x[draw].mean(1),[.025,.975]).tolist(),'seeds':len(x)}


def diagnostics(row,native_path):
    """Truth is used after execution, outside every policy observation port."""
    log=row['global_execution'];observations=log['observations']
    frames={round(f['time'],8):f for f in row['trace']}
    closing=next((a.get('close_time') for a in log['pickup_attempts'] if a.get('close_time') is not None),None)
    available=[d for d in observations if d['estimate'].get('valid') and
               closing is not None and d['observed_time']<closing+1e-8]
    result={'preclose_rgb_target_miss_mm':None,'preclose_observation_to_close_s':None,
            'preclose_service_seconds':None}
    if available:
        d=available[-1]
        truth=d.get('observation_truth')
        if truth is None:
            f=frames[round(d['observed_time'],8)]
            truth={'handle_positions':f['evaluator']['handle_positions']}
        result.update(preclose_rgb_target_miss_mm=float(np.linalg.norm(
            np.asarray(d['estimate']['targets'])-truth['handle_positions'],axis=1).max()*1000),
            preclose_observation_to_close_s=float(closing-d['observed_time']),
            preclose_service_seconds=float(d['service_wait_steps']*.05))
    with np.load(native_path,allow_pickle=False) as a:
        indices=np.flatnonzero(a['stage']=='inference_wait');assert len(indices)>=6
        last=indices[-6:]
        xyz=np.stack([a[k][last] for k in ('x','y','z')],axis=1)
        velocities=np.diff(xyz,axis=0)/np.diff(a['time'][last])[:,None]
        contact=np.stack([a[k][indices] for k in ('left','right','table')],axis=1)
        result.update(wait_end_speed_m_s=float(np.linalg.norm(velocities,axis=1).mean()),
            wait_support_fraction=float((a['left'][indices]|a['right'][indices]|a['table'][indices]).mean()),
            wait_contact_changes=int(np.any(contact[1:]!=contact[:-1],axis=1).sum()))
    return result


def score(row,native_path,legacy,controller,fit=None):
    r,n=components.score(row,native_path,legacy,controller,fit)
    r.update(diagnostics(row,native_path))
    return r,n


def contrast(rows,terms,field):
    grouped=defaultdict(dict)
    for r in rows:
        if r['scenario']=='sham':continue
        key=(r['method'],r['hold_mode'],r['wait_steps'])
        grouped[r['seed']].setdefault(key,[]).append(float(r[field]))
    values=[]
    for seed,g in sorted(grouped.items()):
        assert all(len(g[k])==2 for k,w in terms),(seed,field)
        values.append(sum(w*np.mean(g[k]) for k,w in terms))
    return interval(values)


def effects(rows):
    result={}
    methods=sorted({r['method'] for r in rows})
    for method in methods:
        holds=[h for h in HOLDS if any(r['method']==method and r['hold_mode']==h for r in rows)]
        terms={}
        def slope(hold,weight):return [((method,hold,20),weight),((method,hold,1),-weight)]
        for hold in holds:
            terms[hold+'_wait']=slope(hold,1)
            if hold!='registered':
                terms[hold+'_hold_wait']=slope(hold,1)+slope('registered',-1)
                terms[hold+'_long_gain']=[((method,hold,20),1),((method,'registered',20),-1)]
        if len(holds)==4:
            terms['position_main_wait']=slope('measured_position',.5)+slope('registered',-.5)+slope('measured_pose',.5)+slope('measured_rotation',-.5)
            terms['rotation_main_wait']=slope('measured_rotation',.5)+slope('registered',-.5)+slope('measured_pose',.5)+slope('measured_position',-.5)
        result[method]={name:{f:contrast(rows,t,f) for f in FIELDS} for name,t in terms.items()}
    if all(m in methods for m in METHODS):
        t=[]
        for method,mw in [('closed_loop',1),('visual_servo',-1)]:
            for hold,hw in [('registered',-.5),('measured_position',.5),('measured_rotation',-.5),('measured_pose',.5)]:
                t.extend([((method,hold,20),mw*hw),((method,hold,1),-mw*hw)])
        result['executor_position_wait']={f:contrast(rows,t,f) for f in FIELDS}
    if 'fixed_touchdown' in methods and 'closed_loop' in methods:
        t=[]
        for method,mw in [('fixed_touchdown',1),('closed_loop',-1)]:
            for hold,hw in [('registered',-1),('measured_position',1)]:
                t.extend([((method,hold,20),mw*hw),((method,hold,1),-mw*hw)])
        result['placement_position_wait']={f:contrast(rows,t,f) for f in FIELDS}
    return result


def transition(rows,first,second):
    # Keys are method, holding target, and initial waiting steps.
    grouped=defaultdict(dict)
    for r in rows:
        if r['scenario']=='sham':continue
        key=(r['method'],r['hold_mode'],r['wait_steps'])
        if key in (first,second):grouped[r['seed'],r['scenario']][key]=r['native_Y']
    assert all(set(g)=={first,second} for g in grouped.values())
    answer={'episodes':len(grouped),'seeds':len({s for s,q in grouped}),
            'both_protected':0,'first_only':0,'second_only':0,'neither':0}
    for g in grouped.values():
        key='both_protected' if g[first] and g[second] else 'first_only' if g[first] else 'second_only' if g[second] else 'neither'
        answer[key]+=1
    return answer


def summarize(rows):
    result={'episodes':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
            'native_samples':sum(r['native_samples'] for r in rows),'conditions':{},'effects':effects(rows),'transitions':{}}
    for method in sorted({r['method'] for r in rows}):
        result['conditions'][method]={};result['transitions'][method]={}
        holds=[h for h in HOLDS if any(r['method']==method and r['hold_mode']==h for r in rows)]
        for hold in holds:
            result['conditions'][method][hold]={}
            for step in (1,20):
                result['conditions'][method][hold][str(step)]={}
                for healthy,name in [(False,'faults'),(True,'healthy')]:
                    group=[r for r in rows if r['method']==method and r['hold_mode']==hold and r['wait_steps']==step and (r['scenario']=='sham')==healthy]
                    m=components.base.metrics(group)
                    for field in ('hold_mean_position_error_mm','hold_mean_rotation_error_deg','hold_entry_position_offset_mm',
                                  'wait_end_speed_m_s','wait_contact_changes','wait_support_fraction',
                                  'preclose_rgb_target_miss_mm','preclose_observation_to_close_s','preclose_service_seconds'):
                        a=[r[field] for r in group if r.get(field) is not None]
                        m[field]=float(np.mean(a)) if a else None;m[field+'_n']=len(a)
                    result['conditions'][method][hold][str(step)][name]=m
            result['transitions'][method][hold+'_wait']=transition(rows,(method,hold,1),(method,hold,20))
            if hold!='registered':result['transitions'][method][hold+'_long_gain']=transition(rows,(method,'registered',20),(method,hold,20))
    return result


def table(report,methods=None):
    names={'closed_loop':'CAD feedback','visual_servo':'Landmark','fixed_touchdown':'Fixed touchdown'}
    lines=[r'Executor & Hold target & Wait & $C$ & $Y_{20}$ & $Y_{500}$ & Healthy & Retry & Total (s)\\',r'\midrule']
    for method,holds in report['conditions'].items():
        if methods is not None and method not in methods:continue
        for hold,windows in holds.items():
            for step,g in sorted(windows.items(),key=lambda x:int(x[0])):
                r=g['faults'];h=g['healthy'];n=r['episodes'];hn=h['episodes']
                lines.append(f"{names[method]} & {components.NAMES[hold]} & {int(step)*.05:.2f} & {r['C']}/{n} & {r['Y']}/{n} & {r['native_Y']}/{n} & {h['native_Y']}/{hn} & {r['repeated_pickups']} & {r['total']:.3f} "+r'\\')
    return '\n'.join([r'\begin{tabular}{lllrrrrrr}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
