"""NumPy-only reconstruction of the holding factorial and visual-servo cohorts."""
from collections import defaultdict
import numpy as np
import wait_geometry_audit as base

HOLDS=('registered','measured_position','measured_rotation','measured_pose')
NAMES={'registered':'Registered','measured_position':'Position only',
       'measured_rotation':'Rotation only','measured_pose':'Position + rotation'}


def interval(values):
    x=np.asarray(values,dtype=float)
    draw=np.random.default_rng(20261114).integers(0,len(x),(2000,len(x)))
    return {'estimate':float(x.mean()),'ci95':np.quantile(x[draw].mean(1),[.025,.975]).tolist(),'seeds':len(x)}


def score(row,native_path,legacy,controller,fit=None):
    result,count=base.score(row,native_path,legacy,controller,fit)
    tracking=row['hold_tracking'];ref=row['holding_input']
    result.update(hold_mean_position_error_mm=float(np.mean([max(x['position_error_mm']) for x in tracking])),
        hold_mean_rotation_error_deg=float(np.mean([max(x['rotation_error_deg']) for x in tracking])),
        hold_entry_position_offset_mm=float(np.linalg.norm(np.asarray(ref['measured_positions'])-ref['registered_positions'],axis=1).max()*1000),
        hold_entry_rotation_offset_deg=float(np.degrees(np.arccos(np.clip((np.einsum('aij,aij->a',np.asarray(ref['measured_rotations']),np.asarray(ref['registered_rotations']))-1)/2,-1,1))).max()))
    return result,count


def contrast(rows,terms,field):
    grouped=defaultdict(dict)
    for r in rows:
        if r['scenario']=='sham':continue
        grouped[r['seed']].setdefault((r['hold_mode'],r['wait_steps']),[]).append(float(r[field]))
    values=[]
    for seed,group in sorted(grouped.items()):
        assert all(len(group[key])==2 for key,_ in terms),seed
        values.append(sum(weight*np.mean(group[key]) for key,weight in terms))
    return interval(values)


def summarize(rows):
    holds=[h for h in HOLDS if any(r['hold_mode']==h for r in rows)]
    groups=defaultdict(list)
    for r in rows:groups[r['hold_mode'],r['wait_steps'],r['scenario']=='sham'].append(r)
    report={'episodes':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
        'method':rows[0]['method'],'native_samples':sum(r['native_samples'] for r in rows),
        'fault_results':{},'healthy_controls':{},'paired_effects':{}}
    for hold in holds:
        report['fault_results'][hold]={};report['healthy_controls'][hold]={}
        for step in (1,20):
            for key,healthy in [('fault_results',False),('healthy_controls',True)]:
                g=groups[hold,step,healthy];d=base.metrics(g)
                for field in ('hold_mean_position_error_mm','hold_mean_rotation_error_deg',
                              'hold_entry_position_offset_mm','hold_entry_rotation_offset_deg'):
                    d[field]=float(np.mean([r[field] for r in g]))
                report[key][hold][str(step)]=d
    fields=('C','Y','native_Y','total','pickup_attempts','first_close_dual',
            'setdown_tilt_deg','first_close_miss_mm','hold_mean_position_error_mm','hold_mean_rotation_error_deg')
    def put(name,terms):
        report['paired_effects'][name]={f:contrast(rows,terms,f) for f in fields
            if all(r[f] is not None for r in rows if r['scenario']!='sham')}
    for hold in holds:
        put(hold+'_wait',[((hold,20),1),((hold,1),-1)])
        if hold!='registered':
            put(hold+'_long_vs_registered',[((hold,20),1),(('registered',20),-1)])
            put(hold+'_hold_wait',[((hold,20),1),((hold,1),-1),
                (('registered',20),-1),(('registered',1),1)])
    if len(holds)==4:
        def slope(hold,w):return [((hold,20),w),((hold,1),-w)]
        put('position_main_wait',slope('measured_position',.5)+slope('registered',-.5)+slope('measured_pose',.5)+slope('measured_rotation',-.5))
        put('rotation_main_wait',slope('measured_rotation',.5)+slope('registered',-.5)+slope('measured_pose',.5)+slope('measured_position',-.5))
        put('position_rotation_wait',slope('measured_pose',1)+slope('measured_position',-1)+slope('measured_rotation',-1)+slope('registered',1))
    return report


def table(report):
    lines=[r'Hold target & Wait & $C$ & $Y_{20}$ & $Y_{500}$ & Healthy $Y_{500}$ & Retry & Total (s)\\',r'\midrule']
    for hold in HOLDS:
        if hold not in report['fault_results']:continue
        for step in (1,20):
            r=report['fault_results'][hold][str(step)];h=report['healthy_controls'][hold][str(step)]
            n=r['episodes'];hn=h['episodes']
            lines.append(f"{NAMES[hold]} & {step*.05:.2f} & {r['C']}/{n} & {r['Y']}/{n} & {r['native_Y']}/{n} & {h['native_Y']}/{hn} & {r['repeated_pickups']} & {r['total']:.3f} "+r'\\')
    return '\n'.join([r'\begin{tabular}{llrrrrrr}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'


def native_handover(path):
    with np.load(path,allow_pickle=False) as a:
        def longest(flags):
            run=maximum=0
            for value in flags:run=run+1 if value else 0;maximum=max(maximum,run)
            return maximum
        assert np.allclose(np.diff(a['time']),.002,rtol=0,atol=1e-10)
        missing=~(a['giver_holds']|a['receiver_holds']|a['table'])
        role=np.zeros(len(missing),dtype=bool)
        for name in ('giver_holds','receiver_holds','receiver_supports'):
            indices=a['role']==name;role[indices]=~a[name][indices]
        assert np.array_equal(role,a['role_bad'])
        return {'native_samples':len(missing),'native_registered_gap_ms':longest(missing)*2,
            'native_role_loss_samples':int(role.sum()),'native_longest_role_gap_ms':longest(role)*2,
            'sampled_role_loss_samples':int(role[24::25].sum()),
            'native_shared_bad':bool(longest(missing)>50 or np.any(a['penetration']>.01))}
