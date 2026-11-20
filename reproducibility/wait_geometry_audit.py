"""NumPy-only scoring of prospective holding/geometry interventions."""
from collections import Counter,defaultdict
import itertools
import numpy as np

METHODS=('retarget','closed_loop')
HOLDS=('registered','measured_pose')


def score(row,native_path,legacy,controller,fit=None):
    result,observations=controller.score(row,legacy)
    result.update(controller.native_view(native_path,row,result))
    for key in ('geometry','hold_mode','wait_steps','registered_mass_kg'):
        result[key]=row[key]
    assert abs(row['registered_mass_kg']-1.818)<1e-9
    fits=0
    if fit is not None:
        for d in observations:
            estimate=fit(d['input']);fits+=1
            assert estimate['valid']==d['estimate']['valid']
            if estimate['valid']:
                for key in ('center','rotation','targets','optimization_vector'):
                    assert np.allclose(estimate[key],d['estimate'][key],atol=1e-5,rtol=0),key
    initial=row['post_fault'];end=row['post_wait_evaluator']
    result.update(wait_end_tilt_deg=float(end['tilt_deg']),
        wait_tilt_change_deg=float(end['tilt_deg']-initial['tilt_deg']),
        wait_translation_mm=float(np.linalg.norm(np.array(end['pot_center'])-initial['pot_center']))*1000,
        wait_end_registered_support=bool(any(end['holds']) or end['table_contact']))
    frames=[f for f in row['trace'] if f['phase']=='recovery']
    blocks=[(name,list(group)) for name,group in itertools.groupby(frames,key=lambda f:f['stage'])]
    closes=[group[-1] for name,group in blocks if name=='global_close']
    setdown=next((group[-1] for name,group in blocks if name=='global_table_settle'),None)
    result['setdown_tilt_deg']=setdown['evaluator']['tilt_deg'] if setdown else None
    for label,frame in (('first',closes[0] if closes else None),('last',closes[-1] if closes else None)):
        result[label+'_close_dual']=bool(frame and all(frame['evaluator']['holds']))
        result[label+'_close_miss_mm']=float(np.linalg.norm(np.array(frame['robot']['positions'])-
            frame['evaluator']['handle_positions'],axis=1).max())*1000 if frame else None
    result['first_guard_failure_stage']=None
    with np.load(native_path,allow_pickle=False) as a:
        missing=~(a['left']|a['right']|a['table']);run=0
        penetration=a['interarm_penetration_m']
        for i,value in enumerate(missing):
            run=run+1 if value else 0
            if run>50 or penetration[i]>.01:
                result['first_guard_failure_stage']=str(a['stage'][i]);break
    return result,fits


def interval(values):
    """One value per complete initialization seed, preserving paired faults."""
    x=np.asarray(values,dtype=float)
    draws=np.random.default_rng(20261113).integers(0,len(x),size=(2000,len(x)))
    return {'estimate':float(x.mean()),'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),
            'seeds':len(x)}


def paired(rows,terms,field='native_Y'):
    by_seed=defaultdict(dict)
    for r in rows:
        if r['scenario']=='sham':continue
        key=(r['hold_mode'],r['wait_steps'],r['method'])
        by_seed[r['seed']].setdefault(key,[]).append(float(r[field]))
    values=[]
    for seed,groups in sorted(by_seed.items()):
        assert all(len(groups[key])==2 for key,_ in terms),(seed,terms)
        values.append(sum(weight*np.mean(groups[key]) for key,weight in terms))
    return interval(values)


def metrics(rows):
    result={'episodes':len(rows),'seeds':len({r['seed'] for r in rows})}
    for key in ('C','Y','native_Y','I','S','native_disagreement','history_only','native_history_only',
                'first_close_dual','last_close_dual'):
        result[key]=sum(bool(r[key]) for r in rows)
    for key in ('total','wait','skill','pickup_attempts','wait_end_tilt_deg','wait_tilt_change_deg',
                'wait_translation_mm','setdown_tilt_deg','first_close_miss_mm','last_close_miss_mm'):
        values=[r[key] for r in rows if r[key] is not None]
        result[key]=float(np.mean(values)) if values else None
    result['repeated_pickups']=sum(r['pickup_attempts']>1 for r in rows)
    result['first_guard_failure_stages']=dict(Counter(r['first_guard_failure_stage'] for r in rows if r['first_guard_failure_stage']))
    return result


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['hold_mode'],r['wait_steps'],r['method'],r['scenario']=='sham')].append(r)
    report={'episodes':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
        'fault_results':{},'healthy_controls':{},'paired_effects':{},
        'native_samples':sum(r['native_samples'] for r in rows)}
    holds=sorted({r['hold_mode'] for r in rows});steps=sorted({r['wait_steps'] for r in rows})
    for hold in holds:
        report['fault_results'][hold]={};report['healthy_controls'][hold]={}
        for step in steps:
            report['fault_results'][hold][str(step)]={m:metrics(groups[hold,step,m,False]) for m in METHODS}
            health={m:metrics(groups[hold,step,m,True]) for m in METHODS if groups[hold,step,m,True]}
            if health:report['healthy_controls'][hold][str(step)]=health
        terms=[((hold,20,'closed_loop'),1),((hold,20,'retarget'),-1),
               ((hold,1,'closed_loop'),-1),((hold,1,'retarget'),1)]
        report['paired_effects'][hold+'_method_wait']={k:paired(rows,terms,k) for k in ('C','Y','native_Y','total')}
        for method in METHODS:
            terms=[((hold,20,method),1),((hold,1,method),-1)]
            report['paired_effects'][hold+'_'+method+'_wait']={k:paired(rows,terms,k)
                for k in ('C','Y','native_Y','total','wait','skill','pickup_attempts','wait_end_tilt_deg',
                          'setdown_tilt_deg','first_close_miss_mm','last_close_miss_mm')}
        for step in steps:
            terms=[((hold,step,'closed_loop'),1),((hold,step,'retarget'),-1)]
            report['paired_effects'][hold+f'_method_{step}']={k:paired(rows,terms,k) for k in ('C','Y','native_Y','total')}
    if set(holds)==set(HOLDS):
        for method in METHODS:
            terms=[(('measured_pose',20,method),1),(('registered',20,method),-1),
                   (('measured_pose',1,method),-1),(('registered',1,method),1)]
            report['paired_effects'][method+'_hold_wait']={k:paired(rows,terms,k) for k in ('C','native_Y','total')}
    return report


def tables(report,geometry=False):
    def table(columns,lines):
        return '\\begin{tabular}{'+columns+'}\n\\toprule\n'+'\n'.join(lines)+'\n\\bottomrule\n\\end{tabular}\n'
    lines=[r'Hold & Wait & Policy & $C$ & $Y_{20}$ & $Y_{500}$ & Retry & Total (s)\\',r'\midrule']
    health=report['healthy_controls']
    for hold in ('registered','measured_pose'):
        if hold not in report['fault_results']:continue
        for step,group in sorted(report['fault_results'][hold].items(),key=lambda x:int(x[0])):
            for method,name in zip(METHODS,('Target','RGB loop')):
                r=group[method];n=r['episodes'];title='Registered' if hold=='registered' else 'Measured'
                lines.append(f"{title} & {int(step)*.05:.2f} & {name} & {r['C']}/{n} & {r['Y']}/{n} & {r['native_Y']}/{n} & {r['repeated_pickups']} & {r['total']:.3f} "+r'\\')
    name='geometry-transfer' if geometry else 'wait-hold-full'
    result={name:table('lllrrrrr',lines)}
    return result
