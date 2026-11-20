"""Pilot executor and selected historical replay diagnostics, NumPy only."""
from collections import Counter
from decimal import Decimal,ROUND_HALF_UP
import numpy as np

METHODS=('retarget','visual')

def longest(flags):
    n=best=0
    for value in flags:n=n+1 if value else 0;best=max(best,n)
    return best

def pilot(row,legacy,native_function,data):
    r=legacy(row,'controller_pilot')
    r['I']=r['cycle'];r['S']=not r['shared_bad']
    frames=[f for f in row['trace'] if f['phase'] in ('recovery','final_hold')]
    start=next((f['time']-.05 for f in frames if f['stage'] in ('global_hover','vision_hover')),None)
    r['pickup_suffix_start']=start
    suffix=[f for f in frames if start is not None and f['time']>start+1e-8]
    bad=longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in suffix])>2
    r['pickup_suffix']=bool(r['accepted'] and suffix and not bad and not row['protection']['deep_interarm_contact_steps'])
    r['history_only']=bool(r['pickup_suffix'] and not r['Y'])
    observations=row.get('controller_services',[row['regrasp_decision']] if 'regrasp_decision' in row else [])
    assert observations
    waits=sum(d['service_wait_steps'] for d in observations)
    initial=sum(f['stage']=='inference_wait' for f in frames)
    assert abs((waits+initial)*.05-r['wait'])<1e-7
    assert all(d['service_wall_seconds']<=d['service_wait_steps']*.05+1e-8 for d in observations)
    r.update(native_function(data/'pilot-native'/row['native_file'],row,r))
    with np.load(data/'pilot-native'/row['native_file'],allow_pickle=False) as a:
        miss=~(a['left']|a['right']|a['table']);phase=[]
        for j in range(25):
            phase.append(bool(r['accepted'] and longest(miss[j::25])<=2 and not np.any(a['interarm_penetration_m'][j::25]>.01)))
    assert phase[24]==r['Y']
    r.update(phase_Y=phase,observations=len(observations),pickup_attempts=len(row.get('controller_events',[])) if row['method']=='visual' else 1,
        calls=len(row['skill_events']))
    assert r['calls']==1
    return r,observations

def metrics(rows):
    return {'episodes':len(rows),**{k:sum(bool(r[k]) for r in rows) for k in ('C','Y','I','S','pickup_suffix','history_only','native_Y')},
        **{k:float(np.mean([r[k] for r in rows])) for k in ('total','wait','skill','hold','observations','pickup_attempts')},
        'labels':dict(Counter(r['label'] for r in rows)),
        'phase_Y_counts':[sum(r['phase_Y'][j] for r in rows) for j in range(25)]}

def paired(rows,axis,first,second):
    groups={}
    for r in rows:groups.setdefault((r['seed'],r['scenario']),{})[r[axis]]=r
    assert all(first in g and second in g for g in groups.values())
    seeds=sorted({s for s,q in groups});draw=np.random.default_rng(20261112).integers(0,len(seeds),(2000,len(seeds)))
    counts=np.array([sum(s==key[0] for key in groups) for s in seeds]);effects={}
    for measure in ('Y','C','total','wait'):
        delta=np.array([sum(float(g[first][measure])-float(g[second][measure]) for (s,q),g in groups.items() if s==seed) for seed in seeds])
        effects[measure]={'difference':float(delta.sum()/counts.sum()),'ci95':np.quantile(delta[draw].sum(1)/counts[draw].sum(1),[.025,.975]).tolist()}
    return effects

def summarize(rows):
    out={'episodes':len(rows),'initialization_seeds':len({r['seed'] for r in rows}),
        'fault_results':{},'healthy_controls':{},'executor_effects':{},'waiting_effects':{},'native_samples':sum(r['native_samples'] for r in rows),
        'interpretation':'62000-series independent pilot; legacy labels preserved. Suffix starts at first pickup approach. Separate from63000-series stronger executor.'}
    for load in ('native','density150'):
        for wait in ('actual','1000ms'):
            key=load+'/'+wait
            out['fault_results'][key]={m:metrics([r for r in rows if r['load']==load and r['window']==wait and r['method']==m and r['scenario']!='sham']) for m in METHODS}
            out['healthy_controls'][key]={m:metrics([r for r in rows if r['load']==load and r['window']==wait and r['method']==m and r['scenario']=='sham']) for m in METHODS}
            out['executor_effects'][key]=paired([r for r in rows if r['load']==load and r['window']==wait and r['scenario']!='sham'],'method','visual','retarget')
        for m in METHODS:
            out['waiting_effects'][load+'/'+m]=paired([r for r in rows if r['load']==load and r['method']==m and r['scenario']!='sham'],'window','1000ms','actual')
    return out

def tables(report,history):
    a=[r'Load & Wait & Pilot executor & Goal & $Y_{20}$ & $Y_{500}$ & Healthy & Total(s)\\',r'\midrule']
    for load,loadname in [('native','Native'),('density150','Heavy')]:
        for wait,wname in [('actual','Actual'),('1000ms','1s')]:
            key=load+'/'+wait
            for m,name in [('retarget','Position'),('visual','Pilot RGB')]:
                r=report['fault_results'][key][m];h=report['healthy_controls'][key][m]
                seconds=Decimal(str(round(r['total'],9))).quantize(Decimal('.001'),rounding=ROUND_HALF_UP)
                a.append(f"{loadname} & {wname} & {name} & {r['C']}/40 & {r['Y']}/40 & {r['native_Y']}/40 & {h['Y']}/20 & {seconds} "+r'\\')
    b=[r'State & Wait & Sampled gap(ms) & Native gap(ms) & Native pickup(ms) & Passing phases\\',r'\midrule']
    for r in history['rows']:
        side='L' if r['scenario']=='left_loss' else 'R'
        b.append(f"{r['seed']}{side} & {'Actual' if r['window']=='actual' else '1s'} & {r['primary_gap_ms']:.0f} & {r['native_gap_ms']:.0f} & {r['native_suffix_gap_ms']:.0f} & {r['phase_Y_count']}/25 "+r'\\')
    def wrap(columns,body):return '\n'.join([r'\begin{tabular}{'+columns+'}',r'\toprule',*body,r'\bottomrule',r'\end{tabular}'])+'\n'
    return {'controller-pilot':wrap('lllrrrrr',a),'history-native':wrap('llrrrr',b)}

def history_view(row,case,result,native_function,data):
    r=dict(result);r['I']=r['cycle']
    r['pickup_suffix_start']=next(f['time']-.05 for f in row['trace'] if f['stage']=='global_hover')
    n=native_function(data/'history-native'/case['native_file'],row,r)
    assert r['label']=='B' and r['C'] and r['accepted'] and not r['Y']
    with np.load(data/'history-native'/case['native_file'],allow_pickle=False) as a:
        miss=~(a['left']|a['right']|a['table'])
        phases=[bool(longest(miss[j::25])<=2 and not np.any(a['interarm_penetration_m'][j::25]>.01)) for j in range(25)]
    assert not phases[24]
    return {k:case[k] for k in ('seed','scenario','method','load','window','name')} | {
        'primary_label':'B','primary_gap_ms':row['protection']['max_unsupported_seconds']*1000,
        'native_gap_ms':n['native_gap_ms'],'native_suffix_gap_ms':n['native_suffix_gap_ms'],
        'native_Y':n['native_Y'],'native_history_only':n['native_history_only'],
        'phase_Y':phases,'phase_Y_count':sum(phases),'native_post_fault_samples':n['native_samples']}
