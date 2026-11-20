"""New-controller cohort: uniform global interface, cost and native-guard views."""
from collections import Counter,defaultdict
import numpy as np

METHODS=('retarget','closed_loop')
NAMES=('Target','RGB loop')


def longest(flags):
    run=maximum=0
    for value in flags:
        run=run+1 if value else 0;maximum=max(maximum,run)
    return maximum


def score(row,legacy):
    """Keep legacy labels as checks; put the required interface in new P."""
    result=legacy(row,'controller')
    result['legacy_label']=result['label']
    result['I']=bool(result['cycle'])
    result['S']=bool(not result['shared_bad'])
    result['P']=bool(result['P'] and result['I'])
    result['Y']=bool(result['V'] and result['C'] and result['P'])
    result['label']='E' if not result['V'] else 'A' if result['C'] and result['P'] else 'B' if result['C'] else 'C' if result['P'] else 'D'
    assert result['Y']==row['audit']['protected_completion']
    if 'uniform_audit' in row:
        assert all(result[k]==row['uniform_audit'][k] for k in ('C','P','Y','I','label'))
    log=row.get('global_execution')
    observations=log['observations'] if log else [row['regrasp_decision']]
    total_steps=sum(d['service_wait_steps'] for d in observations)
    initial_steps=sum(f['stage']=='inference_wait' for f in row['trace'] if f['phase']=='recovery')
    assert abs((total_steps+initial_steps)*.05-result['wait'])<1e-7
    assert all(d['service_wait_steps']*.05+1e-8>=d['service_wall_seconds'] for d in observations)
    result.update(observations=len(observations),inspection_seconds=total_steps*.05,
                  maximum_service_seconds=max(d['service_wall_seconds'] for d in observations),
                  pickup_attempts=len(log['pickup_attempts']) if log else 1)
    frames=[f for f in row['trace'] if f['phase'] in ('recovery','final_hold')]
    last_pickup=log['pickup_attempts'][-1]['start_time'] if log and log['pickup_attempts'] else row['prefix_state']['time']
    start=next((f['time']-.05 for f in frames if f['stage']=='global_hover' and f['time']>last_pickup+1e-8),None)
    suffix=[f for f in frames if start is not None and f['time']>start+1e-8]
    suffix_bad=longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in suffix])>2
    result['pickup_suffix']=bool(suffix and result['V'] and result['C'] and result['I'] and not suffix_bad
                                  and not row['protection']['deep_interarm_contact_steps'])
    result['history_only']=bool(result['pickup_suffix'] and not result['Y'])
    result['pickup_suffix_start']=start
    result['registered_gap_ms']=row['protection']['max_unsupported_seconds']*1000
    return result,observations


def native_view(path,row,result):
    with np.load(path,allow_pickle=False) as a:
        t=a['time'];missing=~(a['left']|a['right']|a['table'])
        count=len(t);frames=[f for f in row['trace'] if f['phase'] in ('recovery','final_hold')]
        assert count==25*len(frames)
        assert np.allclose(np.diff(t),.002,atol=1e-10,rtol=0)
        indices=np.arange(24,count,25)
        assert np.allclose(t[indices],[f['time'] for f in frames],atol=1e-8,rtol=0)
        assert np.array_equal(np.stack([a['left'][indices],a['right'][indices]],axis=1),[f['evaluator']['holds'] for f in frames])
        assert np.array_equal(a['table'][indices],[f['evaluator']['table_contact'] for f in frames])
        gap=longest(missing)*.002
        bad=gap>.100000001 or bool(np.any(a['interarm_penetration_m']>.01))
        native_y=bool(result['V'] and result['C'] and result['I'] and not bad)
        start=result.get('pickup_suffix_start')
        suffix=a['time']>start+1e-8 if start is not None else np.zeros(count,dtype=bool)
        suffix_gap=longest(missing[suffix])*.002 if suffix.any() else None
        suffix_good=bool(suffix.any() and suffix_gap<=.100000001 and not np.any(a['interarm_penetration_m'][suffix]>.01))
        phase_y=[]
        for phase in range(25):
            q=slice(phase,None,25)
            phase_bad=longest(missing[q])>2 or bool(np.any(a['interarm_penetration_m'][q]>.01))
            phase_y.append(bool(result['V'] and result['C'] and result['I'] and not phase_bad))
        forces=a['object_contact_force_norm_N'][missing]
        assert np.all(a['force_measured'][missing])
        return {'native_samples':count,'native_gap_ms':gap*1000,'native_Y':native_y,
                'native_suffix_gap_ms':suffix_gap*1000 if suffix_gap is not None else None,
                'native_history_only':bool(result['V'] and result['C'] and result['I'] and suffix_good and not native_y),
                'native_disagreement':bool(native_y!=result['Y']),
                'phase_sensitive_Y':min(phase_y)!=max(phase_y),
                'phase_Y_min':min(phase_y),'phase_Y_max':max(phase_y),
                'missing_native_samples':int(missing.sum()),
                'missing_with_force':int((forces>1e-6).sum()),
                'minimum_force_on_missing_N':float(forces.min()) if len(forces) else None,
                'maximum_tilt_deg':float(a['tilt_deg'].max()) if 'tilt_deg' in a else None,
                'vertical_range_m':float(np.ptp(a['z'])) if 'z' in a else None}


def metrics(rows):
    return {'episodes':len(rows),'seeds':len({r['seed'] for r in rows}),
            **{k:sum(bool(r[k]) for r in rows) for k in ('C','Y','I','S','pickup_suffix','history_only','native_Y','native_disagreement','phase_sensitive_Y')},
            **{k:float(np.mean([r[k] for r in rows])) for k in ('total','wait','skill','hold','observations','pickup_attempts')},
            'labels':dict(Counter(r['label'] for r in rows))}


def paired(rows,first='closed_loop',second='retarget'):
    groups=defaultdict(dict)
    for r in rows:groups[r['seed'],r['scenario']][r['method']]=r
    assert all(set(g)==set(METHODS) for g in groups.values())
    seeds=sorted({s for s,q in groups});n=len(groups)
    draw=np.random.default_rng(20261112).integers(0,len(seeds),(2000,len(seeds)))
    delta=np.array([sum(int(g[first]['Y'])-int(g[second]['Y']) for (s,q),g in groups.items() if s==seed) for seed in seeds])
    per_seed=np.array([[sum(g[first]['total'] for (s,q),g in groups.items() if s==seed),
                        sum(g[second]['total'] for (s,q),g in groups.items() if s==seed)] for seed in seeds])
    common=[(k,g) for k,g in groups.items() if g[first]['Y'] and g[second]['Y']]
    gain={'states':n,'seeds':len(seeds),'Y_difference':float(delta.sum()/n),
          'Y_difference_ci95':np.quantile(delta[draw].sum(1)/n,[.025,.975]).tolist(),
          'Y_gains':sum(g[first]['Y'] and not g[second]['Y'] for g in groups.values()),
          'Y_losses':sum(g[second]['Y'] and not g[first]['Y'] for g in groups.values()),
          'all_attempt_time_difference':float((per_seed[:,0]-per_seed[:,1]).sum()/n),
          'all_attempt_time_difference_ci95':np.quantile((per_seed[draw,0]-per_seed[draw,1]).sum(1)/n,[.025,.975]).tolist(),
          'common_protected_states':len(common)}
    for key in ('C','native_Y'):
        values=np.array([sum(int(g[first][key])-int(g[second][key]) for (s,q),g in groups.items() if s==seed) for seed in seeds])
        gain[key+'_difference']=float(values.sum()/n)
        gain[key+'_difference_ci95']=np.quantile(values[draw].sum(1)/n,[.025,.975]).tolist()
    if common:
        times=np.array([[sum(g[first]['total'] for (s,q),g in common if s==seed),
                         sum(g[second]['total'] for (s,q),g in common if s==seed)] for seed in seeds])
        boot=times[draw].sum(1);valid=boot[:,1]>0
        gain['common_protected_time_reduction']=float(1-times[:,0].sum()/times[:,1].sum())
        gain['common_protected_time_reduction_ci95']=np.quantile(1-boot[valid,0]/boot[valid,1],[.025,.975]).tolist()
    else:
        gain['common_protected_time_reduction']=None;gain['common_protected_time_reduction_ci95']=None
    return gain


def summarize(rows):
    report={'episodes':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
            'fault_results':{},'healthy_controls':{},'paired_methods':{},
            'native_samples':sum(r['native_samples'] for r in rows),
            'native_endpoint_mismatches':0,'native_disagreements':sum(r['native_disagreement'] for r in rows),
            'paired_waits':{},'waiting_method_interaction':{}}
    for load in ('native','density150'):
        for key in ('fault_results','healthy_controls','paired_methods'):report[key][load]={}
        for window in ('actual','1000ms'):
            faults=[r for r in rows if r['load']==load and r['window']==window and r['scenario']!='sham']
            healthy=[r for r in rows if r['load']==load and r['window']==window and r['scenario']=='sham']
            report['fault_results'][load][window]={m:metrics([r for r in faults if r['method']==m]) for m in METHODS}
            report['healthy_controls'][load][window]={m:metrics([r for r in healthy if r['method']==m]) for m in METHODS}
            report['paired_methods'][load][window]=paired(faults)
        report['paired_waits'][load]={}
        for method in METHODS:
            groups=defaultdict(dict)
            for row in rows:
                if row['load']==load and row['method']==method and row['scenario']!='sham':
                    groups[row['seed'],row['scenario']][row['window']]=row
            seeds=sorted({s for s,q in groups})
            draw=np.random.default_rng(20261112).integers(0,len(seeds),(2000,len(seeds)))
            out={'states':len(groups),'seeds':len(seeds)}
            for key in ('C','Y','native_Y','total','wait','skill','pickup_attempts'):
                differences=np.array([np.mean([float(g['1000ms'][key])-float(g['actual'][key])
                    for (s,q),g in groups.items() if s==seed]) for seed in seeds])
                out[key+'_difference']=float(differences.mean())
                out[key+'_difference_ci95']=np.quantile(differences[draw].mean(1),[.025,.975]).tolist()
            out['multiple_pickups']={w:sum(g[w]['pickup_attempts']>1 for g in groups.values()) for w in ('actual','1000ms')}
            report['paired_waits'][load][method]=out
        interaction=defaultdict(dict)
        for row in rows:
            if row['load']==load and row['scenario']!='sham':
                interaction[row['seed'],row['scenario']][row['window'],row['method']]=row
        seeds=sorted({s for s,q in interaction})
        draw=np.random.default_rng(20261112).integers(0,len(seeds),(2000,len(seeds)))
        effects={}
        for key in ('C','Y','native_Y'):
            values=np.array([np.mean([
                float(g['1000ms','closed_loop'][key])-float(g['1000ms','retarget'][key])
                -float(g['actual','closed_loop'][key])+float(g['actual','retarget'][key])
                for (s,q),g in interaction.items() if s==seed]) for seed in seeds])
            effects[key+'_difference_of_differences']=float(values.mean())
            effects[key+'_ci95']=np.quantile(values[draw].mean(1),[.025,.975]).tolist()
        report['waiting_method_interaction'][load]=effects
    return report


def tables(report):
    lines=[r'Load & Wait & Policy & $C$ & $Y_{20}$ & $Y_{500}$ & Healthy & Total (s)\\',r'\midrule']
    detail=[r'Load & Wait & Policy & $C$ & $Y$ & $I$ & $S$ & Pickup & Hist. & $Y_{500}$\\',r'\midrule']
    for load,loadname in [('native','Native'),('density150','Heavy')]:
        for window,wname in [('actual','Actual'),('1000ms','1 s')]:
            for method,name in zip(METHODS,NAMES):
                r=report['fault_results'][load][window][method];h=report['healthy_controls'][load][window][method]
                n=r['episodes'];hn=h['episodes']
                lines.append(f"{loadname} & {wname} & {name} & {r['C']}/{n} & {r['Y']}/{n} & {r['native_Y']}/{n} & {h['Y']}/{hn} & {r['total']:.3f} "+r'\\')
                cells=[str(r[k])+f'/{n}' for k in ('C','Y','I','S','pickup_suffix','history_only','native_Y')]
                detail.append(f'{loadname} & {wname} & {name} & '+' & '.join(cells)+r'\\')
    def table(cols,body):return '\n'.join([r'\begin{tabular}{'+cols+'}',r'\toprule',*body,r'\bottomrule',r'\end{tabular}'])+'\n'
    return {'controller-transfer':table('lllrrrrr',lines),'controller-detail':table('lllrrrrrrr',detail)}
