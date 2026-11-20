"""Common-budget evaluation of the separately frozen arrival-adaptive CAD."""
from collections import defaultdict
import numpy as np
import paired_holding_audit as paired
from ranking_closure_audit import interval

BUDGET=45.


def score(row,native_path,legacy,controller,fit=None):
    r,n=paired.score(row,native_path,legacy,controller,fit)
    log=row['global_execution']
    assert len(log['observations'])<=8 and len(log['pickup_attempts'])<=2
    assert row['state_edit_count_after_initialization']==row['contact_anchor_count']==0
    r.update(C45=bool(r['C'] and r['total']<=BUDGET+1e-7),
             Y500_45=bool(r['native_Y'] and r['total']<=BUDGET+1e-7),
             budget_stop=log.get('budget_stop'),arrival_checks=log.get('arrival_checks',[]))
    return r,n


def pairing(new,old):
    by={(r['seed'],r['scenario'],r['hold_mode'],r['wait_steps']):r for r in old
        if r['method']=='closed_loop' and r['hold_mode'] in ('registered','measured_position')}
    for r in new:
        reference=by[r['seed'],r['scenario'],r['hold_mode'],r['wait_steps']]
        for k in ('initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state'):
            assert r[k]==reference[k],(r['seed'],r['scenario'],k)
    return {'exact_pre_and_post_wait_baseline_pairs':len(new)}


def summarize(new,old):
    allrows=[dict(r) for r in old if r['hold_mode'] in ('registered','measured_position')]+[dict(r) for r in new]
    groups=defaultdict(list)
    for r in allrows:
        r['C45']=bool(r['C'] and r['total']<=BUDGET+1e-7)
        r['Y500_45']=bool(r['native_Y'] and r['total']<=BUDGET+1e-7)
        groups[r['method'],r['hold_mode'],r['wait_steps'],r['scenario']=='sham'].append(r)
    cells=[]
    for (method,hold,wait,healthy),g in sorted(groups.items()):
        cells.append({'method':method,'hold_mode':hold,'wait_steps':wait,'healthy':healthy,'episodes':len(g),
            **{k:sum(bool(r[k]) for r in g) for k in ('C','Y','native_Y','C45','Y500_45')},
            'mean_total_seconds':float(np.mean([r['total'] for r in g])),
            'budget_overruns':sum(r['total']>BUDGET+1e-7 for r in g),
            'budget_stops':sum(bool(r.get('budget_stop')) for r in g),
            'queries':sum(r['observations'] for r in g),'retries':sum(r['pickup_attempts']>1 for r in g)})
    effects={}
    for method in ('closed_loop','visual_servo','robust_cad'):
        terms=[((method,'measured_position',20),1),((method,'measured_position',1),-1),
               ((method,'registered',20),-1),((method,'registered',1),1)]
        effects[method]={f:paired.contrast(allrows,terms,f) for f in ('C','native_Y','C45','Y500_45','total')}
    terms=[]
    for method,mw in (('robust_cad',1),('closed_loop',-1)):
        for hold,hw in (('registered',-1),('measured_position',1)):
            terms.extend([((method,hold,20),mw*hw),((method,hold,1),-mw*hw)])
    effects['adaptive_minus_original_interaction']={f:paired.contrast(allrows,terms,f) for f in ('C','native_Y','C45','Y500_45','total')}
    comparisons={}
    for hold in ('registered','measured_position'):
        for wait in (1,20):
            t=[(('robust_cad',hold,wait),1),(('closed_loop',hold,wait),-1)]
            comparisons[f'{hold}_{wait}']={f:paired.contrast(allrows,t,f) for f in ('C','native_Y','C45','Y500_45','total')}
    return {'new_formal_executions':len(new),'reused_original_executions':len(allrows)-len(new),
            'independent_seeds':len({r['seed'] for r in new}),'common_complete_budget_seconds':BUDGET,
            'common_maximum_queries':8,'common_maximum_pickup_attempts':2,
            'native_samples_new':sum(r['native_samples'] for r in new),'conditions':cells,
            'effects':effects,'adaptive_minus_original_cells':comparisons}


def table(report):
    lines=[r'Executor & Hold & Wait & $C$ & $Y_{500}$ & $Y_{500}^{45}$ & Healthy$^{45}$ & Retry & Total (s)\\',r'\midrule']
    for method,name in (('closed_loop','Original CAD'),('robust_cad','Arrival-adaptive CAD'),('visual_servo','Landmark')):
        for hold,title in (('registered','Registered'),('measured_position','Position only')):
            for wait in (1,20):
                pair=[c for c in report['conditions'] if (c['method'],c['hold_mode'],c['wait_steps'])==(method,hold,wait)]
                c=next(c for c in pair if not c['healthy']);h=next(c for c in pair if c['healthy'])
                lines.append(f"{name} & {title} & {wait*.05:.2f} & {c['C']}/40 & {c['native_Y']}/40 & {c['Y500_45']}/40 & {h['Y500_45']}/20 & {c['retries']} & {c['mean_total_seconds']:.3f}"+r'\\')
    return '\n'.join([r'\begin{tabular}{lllrrrrrr}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
