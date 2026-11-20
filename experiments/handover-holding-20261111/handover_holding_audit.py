"""NumPy-only scoring of initial handover holding, with existing sampled roles."""
from collections import Counter,defaultdict
import numpy as np

HOLDS=('registered','measured_position','smooth_position')
NAMES={'registered':'Registered','measured_position':'Position only','smooth_position':'Smooth position'}
METHODS=('rule','R3')
METHOD_NAMES={'rule':'Local + retry','R3':'Global'}


def interval(values):
    x=np.asarray(values,dtype=float)
    draws=np.random.default_rng(20261116).integers(0,len(x),(2000,len(x)))
    return {'estimate':float(x.mean()),'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),'seeds':len(x)}


def reference_summary(rows):
    """The existing CAD component cohort; descriptive, no new executions."""
    rows=[r for r in rows if r['method']=='closed_loop' and r['scenario']!='sham'
          and r['wait_steps']==20 and r['hold_mode'] in ('registered','measured_position')]
    assert len(rows)==80
    fields=('hold_entry_position_offset_mm','hold_mean_position_error_mm','wait_translation_mm',
            'wait_support_fraction','wait_contact_changes','wait_end_speed_m_s','first_close_dual','native_Y')
    report={'source_cohort':'70000--70019 CAD holding factorial','physical_executions_added':0,
        'faults_per_condition':40,'initialization_seeds':20,'conditions':{}}
    for hold in ('registered','measured_position'):
        g=[r for r in rows if r['hold_mode']==hold];seeds=sorted({r['seed'] for r in g})
        d={}
        for field in fields:
            result=interval([np.mean([r[field] for r in g if r['seed']==seed]) for seed in seeds])
            d[field]={'mean':result['estimate'],'ci95':result['ci95']}
        report['conditions'][hold]=d
    return report


def longest(flags):
    run=maximum=0
    for flag in flags:run=run+1 if flag else 0;maximum=max(maximum,run)
    return maximum


def score(row,native_path,offline):
    result=offline.score_handover(row)
    frames=[f for f in row['trace'] if f['phase'] in ('recovery','release','hold')]
    with np.load(native_path,allow_pickle=False) as a:
        n=len(a['time']);assert n==25*len(frames),(n,len(frames))
        assert np.allclose(np.diff(a['time']),.002,rtol=0,atol=1e-10)
        assert np.allclose(a['time'][24::25],[f['time'] for f in frames],rtol=0,atol=1e-9)
        for key in ('giver_holds','receiver_holds','receiver_supports'):
            assert np.array_equal(a[key][24::25],[f['evaluator'][key] for f in frames]),key
        assert np.array_equal(a['table'][24::25],[f['evaluator']['table_contact'] for f in frames])
        missing=~(a['giver_holds']|a['receiver_holds']|a['table'])
        gap_ms=longest(missing)*2
        shared_bad=bool(gap_ms>100 or np.any(a['penetration']>.01))
        hard=bool(row['blocked_calls'] or row.get('failure_reason')=='numerical_instability')
        # Shared guards change recording rate; entry, role and interface rules
        # retain their predeclared sampled definitions.
        native_y=bool(result['C'] and result['V'] and result['cycle'] and not hard
            and not result['role_loss_samples'] and not shared_bad)
        w=a['initial_wait'];assert w.any()
        contacts=np.stack([a[k][w] for k in ('giver_holds','receiver_holds','table')],axis=1)
        force=a['object_contact_force_norm_N'][w]
        result.update(native_Y=native_y,native_samples=n,native_gap_ms=gap_ms,
            native_shared_bad=shared_bad,wait_support_fraction=float((contacts.any(1)).mean()),
            wait_contact_changes=int(np.any(contacts[1:]!=contacts[:-1],axis=1).sum()),
            wait_contact_force_mean_N=float(force.mean()),wait_contact_force_max_N=float(force.max()))
    result.update({k:row[k] for k in ('hold_mode','wait_steps')})
    ref=row['holding_input'];tracking=row['hold_tracking']
    measured=np.asarray(ref['measured_positions']);registered=np.asarray(ref['registered_positions'])
    assert np.isclose(ref['smooth_transition_seconds'],.25)
    for f in tracking:
        alpha=0. if row['hold_mode']=='registered' else 1. if row['hold_mode']=='measured_position' else min(1.,f['elapsed']/.25)
        assert abs(f['alpha']-alpha)<1e-9
        target=(1-alpha)*registered+alpha*measured
        assert np.allclose(target,f['target_positions'],rtol=0,atol=1e-12)
        error=np.linalg.norm(target-f['measured_positions'],axis=1)*1000
        assert np.allclose(error,f['position_error_mm'],rtol=0,atol=1e-9)
    first=row['decision_events'][0]
    assert len(tracking)==first['physical_wait_steps']+first['controller_wait_steps']
    result.update(hold_entry_position_offset_mm=float(np.linalg.norm(registered-measured,axis=1).max()*1000),
        hold_mean_position_error_mm=float(np.mean([max(f['position_error_mm']) for f in tracking])),
        hold_mean_arm_torque_norm_Nm=float(np.mean([np.linalg.norm(f['arm_actuator_torques_Nm'],axis=1).max() for f in tracking])),
        hold_mean_wrist_force_norm_N=float(np.mean([np.linalg.norm(f['wrist_force_sensor_N'],axis=1).max() for f in tracking])),
        control_overrun_steps=row['controller_overrun_steps'],
        initial_wait_steps=first['physical_wait_steps'],
        initial_wait_seconds=first['end_time']-first['start_time'])
    return result


def check_pairing(rows):
    groups=defaultdict(list);ends=defaultdict(list);short=defaultdict(dict)
    for r in rows:
        groups[r['seed'],r['scenario']].append(r)
        ends[r['seed'],r['scenario'],r['hold_mode'],r['wait_steps']].append(r['decision_events'][0]['post_wait_state'])
        if r['wait_steps']==1:short[r['seed'],r['scenario'],r['method']][r['hold_mode']]=r['decision_events'][0]['post_wait_state']
    for group in groups.values():
        assert len(group)==12,len(group)
        for key in ('initialization_state','initialization_end_state','prefix_state','registered_robot_poses'):
            assert all(r[key]==group[0][key] for r in group),key
        ref=group[0]['holding_input']
        assert all(r['holding_input']==ref for r in group)
    for group in ends.values():assert len(group)==2 and group[0]==group[1]
    # At t=0 the smooth policy starts exactly at the previous target. A one-step
    # wait must therefore match registered holding, unless a charged overrun occurs.
    equal=sum(g['registered']==g['smooth_position'] for g in short.values())
    return {'matched_prehold_groups':len(groups),'matched_posthold_path_pairs':len(ends),
            'registered_smooth_short_pairs':len(short),'registered_smooth_short_equal':equal}


def metrics(rows):
    out={'episodes':len(rows),'C':sum(r['C'] for r in rows),'Y20':sum(r['Y'] for r in rows),
        'Y500':sum(r['native_Y'] for r in rows),'retry':sum(r['local_then_global'] for r in rows),
        'invalid_entries':sum(not r['V'] for r in rows),'labels20':dict(Counter(r['label'] for r in rows))}
    for field in ('total','wait','skill','release','hold','hold_entry_position_offset_mm',
        'hold_mean_position_error_mm','hold_mean_arm_torque_norm_Nm','hold_mean_wrist_force_norm_N',
        'wait_contact_force_mean_N','wait_support_fraction','wait_contact_changes'):
        out[field]=float(np.mean([r[field] for r in rows]))
    return out


def contrast(rows,terms,field):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['scenario']=='nominal':continue
        groups[r['seed']][r['method'],r['hold_mode'],r['wait_steps']].append(float(r[field]))
    values=[]
    for seed,group in sorted(groups.items()):
        assert all(len(group[key])==2 for key,w in terms),seed
        values.append(sum(w*np.mean(group[key]) for key,w in terms))
    return interval(values)


def summarize(rows):
    report={'episodes':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
        'native_samples':sum(r['native_samples'] for r in rows),'conditions':{},'effects':{},
        'control_overrun_steps':sum(r['control_overrun_steps'] for r in rows)}
    fields=('C','Y','native_Y','total','local_then_global')
    for method in METHODS:
        report['conditions'][method]={};report['effects'][method]={}
        for hold in HOLDS:
            report['conditions'][method][hold]={}
            for step in (1,20):
                condition={}
                for name,healthy in [('faults',False),('healthy',True)]:
                    g=[r for r in rows if r['method']==method and r['hold_mode']==hold and r['wait_steps']==step and (r['scenario']=='nominal')==healthy]
                    condition[name]=metrics(g)
                report['conditions'][method][hold][str(step)]=condition
            terms=[((method,hold,20),1),((method,hold,1),-1)]
            report['effects'][method][hold+'_wait']={f:contrast(rows,terms,f) for f in fields}
            if hold!='registered':
                interaction=terms+[((method,'registered',20),-1),((method,'registered',1),1)]
                report['effects'][method][hold+'_hold_wait']={f:contrast(rows,interaction,f) for f in fields}
                long_terms=[((method,hold,20),1),((method,'registered',20),-1)]
                report['effects'][method][hold+'_long_gain']={f:contrast(rows,long_terms,f) for f in fields}
    terms=[]
    for method,sign in [('rule',1),('R3',-1)]:
        terms += [((method,'measured_position',20),sign),((method,'measured_position',1),-sign),
                  ((method,'registered',20),-sign),((method,'registered',1),sign)]
    report['effects']['path_position_wait']={f:contrast(rows,terms,f) for f in fields}
    return report


def table(report,compact=False):
    if compact:
        lines=[r'Path & Initial hold & $Y_{500}$ S/L & Total S/L (s) & Retry S/L & Healthy\\',r'\midrule']
        for method in METHODS:
            for hold in HOLDS:
                g=report['conditions'][method][hold];s=g['1']['faults'];l=g['20']['faults']
                hn=sum(g[str(w)]['healthy']['Y500'] for w in (1,20))
                den=sum(g[str(w)]['healthy']['episodes'] for w in (1,20))
                lines.append(f"{METHOD_NAMES[method]} & {NAMES[hold]} & {s['Y500']}/{l['Y500']} & {s['total']:.2f}/{l['total']:.2f} & {s['retry']}/{l['retry']} & {hn}/{den} "+r'\\')
        spec='llrrrr'
    else:
        lines=[r'Path & Initial hold & Wait & $C$ & $Y_{20}$ & $Y_{500}$ & Healthy $Y_{500}$ & Retry & Total (s)\\',r'\midrule']
        for method in METHODS:
            for hold in HOLDS:
                for w in (1,20):
                    c=report['conditions'][method][hold][str(w)];r=c['faults'];h=c['healthy'];n=r['episodes']
                    lines.append(f"{METHOD_NAMES[method]} & {NAMES[hold]} & {w*.05:.2f} & {r['C']}/{n} & {r['Y20']}/{n} & {r['Y500']}/{n} & {h['Y500']}/{h['episodes']} & {r['retry']} & {r['total']:.3f} "+r'\\')
        spec='lllrrrrrr'
    return '\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
