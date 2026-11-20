"""NumPy-only native Transport scoring and paired waiting-reference contrasts."""
from collections import defaultdict
import hashlib
import numpy as np

METHODS=('diffusion_policy_cnn','lstm_gmm')
NAMES={'diffusion_policy_cnn':'Diffusion Policy','lstm_gmm':'LSTM--GMM'}
CONTEXTS={32:'Early interaction',272:'Payload transport'}

def interval(values):
    x=np.asarray(values,float);d=np.random.default_rng(20261120).integers(0,len(x),(2000,len(x)))
    return {'estimate':float(x.mean()),'ci95':np.quantile(x[d].mean(1),[.025,.975]).tolist(),'seeds':len(x)}

def score(row):
    policy=row['records'];waiting=row['wait_records'];events=sorted(policy+waiting,key=lambda r:r['time'])
    assert len(policy)==row['native_policy_steps']==700
    assert [r['policy_step'] for r in policy]==list(range(1,701))
    assert np.allclose(np.diff([row['initial_state']['time']]+[r['time'] for r in events]),.05,atol=1e-8,rtol=0)
    assert row['state_edits_after_reset']==0
    assert all(r['C']==bool(r['payload_in_target_bin'] and r['trash_in_trash_bin']) for r in events)
    assert all(float(r['C'])==r['reward'] for r in policy)
    complete=any(r['C'] for r in events);assert complete==row['C']
    queries=row['policy_query_seconds'];chunk=8 if row['method']=='diffusion_policy_cnn' else 1
    assert len(queries)==int(np.ceil(700/chunk))
    physical=events[-1]['time']-row['initial_state']['time']
    observation=sum(r['observation_service_seconds'] for r in events)
    assert abs(physical-row['simulated_seconds'])<1e-8
    assert abs(observation-row['observation_service_seconds'])<1e-7
    assert abs(sum(queries)-row['query_seconds'])<1e-8
    charged=physical+observation+sum(queries);assert abs(charged-row['charged_seconds'])<1e-7
    first=None;observed=0.
    for event in events:
        observed+=event['observation_service_seconds']
        if event['C'] and first is None:
            index=(event['policy_step'] if event['phase']=='wait' else event['policy_step']-1)//chunk
            first=event['time']-row['initial_state']['time']+sum(queries[:index+1])
    ref=row['holding_input'];offset=tracking=drift=None
    if row['hold_mode']=='unpaused':assert len(waiting)==0 and ref is None
    else:
        assert len(waiting)==row['wait_steps']==row['physical_wait_steps']
        measured=np.asarray(ref['measured_positions']);registered=np.asarray(ref['registered_positions'])
        target=registered if row['hold_mode']=='registered' else measured
        assert np.allclose(target,ref['command_positions'],atol=1e-12,rtol=0)
        assert np.allclose(registered,ref['entry']['controller_position_goals'],atol=1e-12,rtol=0)
        assert np.allclose(ref['registered_rotations'],ref['entry']['controller_rotation_goals'],atol=1e-12,rtol=0)
        assert np.allclose(ref['grippers'],np.asarray(row['actions'][row['pause_boundary']-1])[[6,13]],atol=0,rtol=0)
        assert all(np.allclose(d['command_positions'],target,atol=1e-12,rtol=0) for d in row['hold_tracking'])
        assert abs(ref['end']['time']-ref['entry']['time']-.05*len(waiting))<1e-8
        offset=float(np.linalg.norm(registered-measured,axis=1).max()*1000)
        tracking=float(np.mean([np.linalg.norm(np.asarray(d['measured_positions'])-target,axis=1).max()*1000 for d in row['hold_tracking']]))
        entry=policy[row['pause_boundary']-1]
        drift=float(np.linalg.norm(np.asarray(waiting[-1]['payload_position'])-entry['payload_position'])*1000)
    return {'seed':row['seed'],'method':row['method'],'scenario':row['scenario'],'hold_mode':row['hold_mode'],
        'wait_steps':row['wait_steps'],'pause_boundary':row['pause_boundary'],'C':complete,
        'native_policy_steps':len(policy),'physical_wait_steps':len(waiting),'simulated_seconds':physical,
        'query_seconds':sum(queries),'port_assembly_seconds':observation,'control_query_seconds':physical+sum(queries),
        'first_completion_control_query_seconds':first,'entry_reference_error_mm':offset,'holding_tracking_error_mm':tracking,
        'wait_payload_translation_mm':drift,'prefix_already_completed':any(r['C'] and r['policy_step']<=row['pause_boundary'] for r in policy),
        'prefix_action_max_residual':row['prefix_action_max_residual']}

def pairing_record(row):
    r={k:row[k] for k in ('seed','method','hold_mode','wait_steps','pause_boundary','initial_state',
        'boundary_states','holding_input','prefix_observation_sha256','policy_boundary_state','prefix_action_max_residual')}
    chunk=8 if row['method']=='diffusion_policy_cnn' else 1
    r['rollin_actions']=row['actions'] if row['hold_mode']=='unpaused' else row['actions'][:row['pause_boundary']+chunk]
    r['unused_prediction_residual']=row.get('numerical_prefix_prediction_max_residual',row['prefix_action_max_residual'])
    return r

def check_pairing(rows):
    baseline={(r['method'],r['seed']):r for r in rows if r['hold_mode']=='unpaused'}
    pairs=0;groups=set();maximum=0.;unused=0.
    for r in rows:
        if r['hold_mode']=='unpaused':continue
        b=baseline[r['method'],r['seed']];boundary=str(r['pause_boundary'])
        assert r['initial_state']==b['initial_state']
        assert r['holding_input']['entry']==b['boundary_states'][boundary]
        assert r['prefix_observation_sha256'][boundary]==b['prefix_observation_sha256'][boundary]
        assert r['policy_boundary_state'][boundary]==b['policy_boundary_state'][boundary]
        length=len(r['rollin_actions']);delta=float(np.max(abs(np.asarray(r['rollin_actions'])-b['rollin_actions'][:length])))
        assert delta==0.;maximum=max(maximum,delta);unused=max(unused,r['unused_prediction_residual'])
        pairs+=1;groups.add((r['method'],r['seed'],boundary))
    return {'unique_unpaused_executions':len(baseline),'matched_boundary_groups':len(groups),
        'matched_physics_rgb_policy_pairs':pairs,'maximum_replayed_action_residual':maximum,
        'maximum_unused_reprediction_residual':unused}

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['method'],r['pause_boundary'],r['hold_mode'],r['wait_steps']].append(r)
    report={'episodes':len(rows),'independent_initialization_seeds':len({r['seed'] for r in rows}),
        'policy_action_steps':sum(r['native_policy_steps'] for r in rows),
        'physical_wait_steps':sum(r['physical_wait_steps'] for r in rows),'conditions':[],
        'effects':{},'budget_curves':{},
        'cost_view':'Native physical control time plus measured policy-query time. Port assembly is measured separately; camera rendering is not inferred from that timer. These synchronous benchmark components are not the complete physically replayed service costs used in the lifting studies.'}
    for key,g in sorted(groups.items()):
        method,boundary,hold,wait=key
        condition={'method':method,'pause_boundary':boundary,'hold_mode':hold,'wait_steps':wait,
            'episodes':len(g),'C':sum(r['C'] for r in g),'prefix_already_completed':sum(r['prefix_already_completed'] for r in g)}
        for f in ('simulated_seconds','query_seconds','port_assembly_seconds','control_query_seconds',
                  'entry_reference_error_mm','holding_tracking_error_mm','wait_payload_translation_mm'):
            v=[r[f] for r in g if r[f] is not None];condition[f]=float(np.mean(v)) if v else None
        report['conditions'].append(condition)
        times=sorted({0.,*[r['control_query_seconds'] for r in g],*[r['first_completion_control_query_seconds'] for r in g if r['C']]})
        report['budget_curves']['/'.join(map(str,key))]={'denominator':len(g),'seconds':times,
            'native_completed':[sum(r['C'] and r['first_completion_control_query_seconds']<=t for r in g)/len(g) for t in times]}
    for method in sorted({r['method'] for r in rows}):
        for boundary in (32,272):
            lookup={(r['seed'],r['hold_mode'],r['wait_steps']):r for r in rows if r['method']==method and r['pause_boundary']==boundary}
            seeds=sorted({s for s,h,w in lookup});assert len(seeds)==20
            def terms(t):return interval([sum(weight*float(lookup[s,h,w]['C']) for h,w,weight in t) for s in seeds])
            effect={'holding_by_wait':terms([('measured_position',20,1),('registered',20,-1),('measured_position',1,-1),('registered',1,1)]),
                'long_position_gain':terms([('measured_position',20,1),('registered',20,-1)])}
            transition={'both':0,'registered_only':0,'position_only':0,'neither':0}
            for s in seeds:
                a=lookup[s,'registered',20]['C'];b=lookup[s,'measured_position',20]['C']
                transition['both' if a and b else 'registered_only' if a else 'position_only' if b else 'neither']+=1
            effect['long_paired_transitions']=transition;report['effects'][f'{method}/{boundary}']=effect
    return report

def tables(report):
    compact=[r'Policy & Boundary & Registered $S/L$ & Position $S/L$ & Interaction [95\% CI]\\',r'\midrule']
    full=[r'Policy & Boundary & Hold & Wait & $C$ & Control (s) & Query (s) & Control + query (s)\\',r'\midrule']
    by={(c['method'],c['pause_boundary'],c['hold_mode'],c['wait_steps']):c for c in report['conditions']}
    for method in METHODS:
        if not any(c['method']==method for c in report['conditions']):continue
        base=by[method,0,'unpaused',0]
        full.append(f"{NAMES[method]} & Native & Unpaused & 0 & {base['C']}/{base['episodes']} & {base['simulated_seconds']:.2f} & {base['query_seconds']:.2f} & {base['control_query_seconds']:.2f} "+r'\\')
        for boundary in (32,272):
            cells=[]
            for hold in ('registered','measured_position'):
                a=by[method,boundary,hold,1];b=by[method,boundary,hold,20]
                cells.append(f"{a['C']}/{a['episodes']} / {b['C']}/{b['episodes']}")
                for c in (a,b):
                    name='Registered' if hold=='registered' else 'Position'
                    full.append(f"{NAMES[method]} & {boundary} & {name} & {c['wait_steps']*.05:.2f} & {c['C']}/{c['episodes']} & {c['simulated_seconds']:.2f} & {c['query_seconds']:.2f} & {c['control_query_seconds']:.2f} "+r'\\')
            e=report['effects'][f'{method}/{boundary}']['holding_by_wait'];lo,hi=np.asarray(e['ci95'])*100
            compact.append(f"{NAMES[method]} & {boundary} & {cells[0]} & {cells[1]} & {e['estimate']*100:+.1f} [{lo:.1f}, {hi:.1f}] "+r'\\')
    return {name:'\n'.join([r'\begin{tabular}{'+fmt+'}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
        for name,fmt,lines in [('public-policy-main','llrrr',compact),('public-policy-full','llllrrrr',full)]}
