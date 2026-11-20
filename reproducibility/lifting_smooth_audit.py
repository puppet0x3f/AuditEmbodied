"""Frozen-metric scoring of registered, immediate and smooth position holding."""
from collections import defaultdict
import numpy as np
import paired_holding_audit as paired

HOLDS=('registered','measured_position','smooth_position')
NAMES={'registered':'Registered','measured_position':'Position only','smooth_position':'Smooth position'}

def score(row,native_path,legacy,controller,fit=None):
    r,n=paired.score(row,native_path,legacy,controller,fit)
    ref=row['holding_input'];start=np.asarray(ref['registered_positions']);end=np.asarray(ref['measured_positions'])
    for d in row['hold_tracking']:
        elapsed=d['time']-ref['entry_time']
        alpha=0. if row['hold_mode']=='registered' else 1. if row['hold_mode']=='measured_position' else min(1.,elapsed/.25)
        assert abs(alpha-d['alpha'])<1e-12
        assert np.allclose((1-alpha)*start+alpha*end,d['target_positions'],atol=1e-12,rtol=0)
    r['initial_wait_seconds']=row['post_wait_state']['time']-row['prefix_state']['time']
    assert abs(r['initial_wait_seconds']-.05*row['wait_steps'])<1e-8
    return r,n

def pairing(rows):
    groups=defaultdict(list);short=defaultdict(dict)
    for r in rows:
        groups[r['seed'],r['scenario']].append({k:r[k] for k in ('initialization_state','preclose_state','initialization_end_state','prefix_state')})
        if r['wait_steps']==1:short[r['seed'],r['scenario']][r['hold_mode']]=r['post_wait_state']
    for k,g in groups.items():assert all(x==g[0] for x in g),k
    for k,g in short.items():assert g['registered']==g['smooth_position'],k
    return {'matched_prehold_groups':len(groups),'exact_short_registered_smooth_pairs':len(short)}

def interval(values):
    x=np.asarray(values,float);d=np.random.default_rng(20261118).integers(0,len(x),(2000,len(x)))
    return {'estimate':float(x.mean()),'ci95':np.quantile(x[d].mean(1),[.025,.975]).tolist(),'seeds':len(x)}

def contrast(rows,terms,field):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['scenario']!='sham':groups[r['seed']][r['hold_mode'],r['wait_steps']].append(float(r[field]))
    return interval([sum(w*np.mean(g[k]) for k,w in terms) for s,g in sorted(groups.items())])

def summarize(rows):
    report={'episodes':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
        'native_samples':sum(r['native_samples'] for r in rows),'conditions':{},'effects':{},'transitions':{}}
    for hold in HOLDS:
        report['conditions'][hold]={}
        for step in (1,20):
            cells={}
            for healthy,label in ((False,'faults'),(True,'healthy')):
                g=[r for r in rows if r['hold_mode']==hold and r['wait_steps']==step and (r['scenario']=='sham')==healthy]
                cells[label]=paired.components.base.metrics(g)
                for f in ('hold_entry_position_offset_mm','hold_mean_position_error_mm','wait_translation_mm',
                          'wait_end_speed_m_s','preclose_rgb_target_miss_mm','first_close_miss_mm'):
                    v=[r[f] for r in g if r.get(f) is not None]
                    cells[label][f]=float(np.mean(v)) if v else None
            report['conditions'][hold][str(step)]=cells
        if hold=='registered':continue
        terms={'long_gain':[((hold,20),1),(('registered',20),-1)],
            'hold_wait':[((hold,20),1),((hold,1),-1),(('registered',20),-1),(('registered',1),1)]}
        if hold=='smooth_position':terms['long_vs_immediate']=[((hold,20),1),(('measured_position',20),-1)]
        for name,t in terms.items():
            report['effects'][hold+'_'+name]={f:contrast(rows,t,f) for f in ('C','Y','native_Y','total','pickup_attempts','first_close_dual')}
        report['transitions'][hold+'_long_gain']=paired.transition(rows,('closed_loop','registered',20),('closed_loop',hold,20))
    return report

def table(report):
    lines=[r'Hold target & Wait & $C$ & $Y_{20}$ & $Y_{500}$ & Healthy & Retry & Total (s)\\',r'\midrule']
    for hold in HOLDS:
        for step in (1,20):
            g=report['conditions'][hold][str(step)];r=g['faults'];h=g['healthy'];n=r['episodes'];hn=h['episodes']
            lines.append(f"{NAMES[hold]} & {step*.05:.2f} & {r['C']}/{n} & {r['Y']}/{n} & {r['native_Y']}/{n} & {h['native_Y']}/{hn} & {r['repeated_pickups']} & {r['total']:.3f} "+r'\\')
    return '\n'.join([r'\begin{tabular}{llrrrrrr}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
