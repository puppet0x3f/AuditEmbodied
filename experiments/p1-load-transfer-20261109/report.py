"""Separate load/wait denominators and paired initialization-seed statistics."""
from collections import Counter,defaultdict
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import campaign as c
from auditembodied.p0 import same_prefix

ROOT=Path(__file__).resolve().parent


def safe(r):return bool(r['audit']['protected_completion'])


def mean(rows,key):
    values=[r[key] for r in rows if key in r]
    return float(np.mean(values)) if values else None


def metrics(rows):
    guards=lambda r:r.get('protection',{})
    return {'episodes':len(rows),'protected_completions':sum(map(safe,rows)),
        'terminal_goals':sum(r.get('task_complete',False) for r in rows),
        'accepted_completions':sum(r.get('success',False) for r in rows),
        'labels':dict(Counter(r['audit']['label'] for r in rows)),
        'shared_guard_violations':sum(guards(r).get('max_unsupported_seconds',0)>.1 or bool(guards(r).get('deep_interarm_contact_steps',0)) for r in rows),
        'keeper_role_violations':sum(guards(r).get('keeper_grasp_lost_steps',0)>0 for r in rows),
        'role_only_violations':sum(guards(r).get('keeper_grasp_lost_steps',0)>0 and guards(r).get('max_unsupported_seconds',0)<=.1 and not guards(r).get('deep_interarm_contact_steps',0) for r in rows),
        'invalid_entry':sum(r['audit'].get('V_entry') is False for r in rows),
        'observation_valid_entry_invalid':sum(r['audit'].get('observation_valid_entry_invalid',False) for r in rows),
        'mean_total_seconds':mean(rows,'total_post_fault_seconds'),
        'complete_cost_observations':sum('total_post_fault_seconds' in r for r in rows),
        'mean_skill_seconds':mean(rows,'skill_seconds'),'mean_wait_seconds':mean(rows,'inference_wait_seconds'),
        'mean_hold_seconds':mean(rows,'hold_seconds'),'mean_recovery_path_m':mean(rows,'recovery_path_m'),
        'max_selector_wall_seconds':max(r['selector_wall_seconds'] for r in rows),
        'choices':dict(Counter(r['candidate'] for r in rows)),
        'failure_reasons':dict(Counter(r.get('failure_reason') or r['audit']['label'] for r in rows if not safe(r)))}


def paired(a,b,seeds):
    ga={(r['seed'],r['scenario']):r for r in a};gb={(r['seed'],r['scenario']):r for r in b}
    assert set(ga)==set(gb)
    keys=sorted(ga);rng=np.random.default_rng(20261109)
    indices=rng.integers(0,len(seeds),(2000,len(seeds)))
    delta=np.asarray([np.mean([int(safe(ga[k]))-int(safe(gb[k])) for k in keys if k[0]==seed]) for seed in seeds])
    result={'paired_states':len(keys),'independent_seeds':len(seeds),
        'protected_completion_difference':float(delta.mean()),
        'protected_completion_difference_ci95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist()}
    for name,valid in [('all_attempt_cost',lambda k:'total_post_fault_seconds' in ga[k] and 'total_post_fault_seconds' in gb[k]),
                       ('common_protected_cost',lambda k:safe(ga[k]) and safe(gb[k]))]:
        use=[k for k in keys if valid(k)]
        cost={'states':len(use),'missing_cost_pairs':len(keys)-sum('total_post_fault_seconds' in ga[k] and 'total_post_fault_seconds' in gb[k] for k in keys)}
        if use:
            counts=np.array([sum(k[0]==seed for k in use) for seed in seeds])
            asum=np.array([sum(ga[k]['total_post_fault_seconds'] for k in use if k[0]==seed) for seed in seeds])
            bsum=np.array([sum(gb[k]['total_post_fault_seconds'] for k in use if k[0]==seed) for seed in seeds])
            ns=counts[indices].sum(1);valid_draw=ns>0
            av=asum[indices].sum(1)[valid_draw]/ns[valid_draw]
            bv=bsum[indices].sum(1)[valid_draw]/ns[valid_draw]
            cost.update(first_mean_seconds=float(asum.sum()/counts.sum()),second_mean_seconds=float(bsum.sum()/counts.sum()),
                reduction_fraction=float(1-asum.sum()/bsum.sum()),
                reduction_ci95=np.quantile(1-av/bv,[.025,.975]).tolist(),
                difference_seconds=float((asum.sum()-bsum.sum())/counts.sum()),
                difference_ci95=np.quantile(av-bv,[.025,.975]).tolist())
        result[name]=cost
    return result


def main():
    p=json.loads((ROOT/'protocol.json').read_text());c.verify(p)
    freeze=json.loads((ROOT/'formal-freeze.json').read_text())
    assert c.sha(ROOT/'protocol.json')==freeze['protocol_sha256']
    rows=[json.loads(l) for l in (ROOT/'trials.jsonl').open()]
    assert len(rows)==p['expected_episodes']
    expected=set(itertools.product(p['test_seeds'],c.SCENARIOS,c.METHODS,c.LOADS,c.WINDOWS))
    assert {(r['seed'],r['scenario'],r['method'],r['load'],r['window']) for r in rows}==expected
    assert all(r['audit']==c.admission(r) for r in rows)
    group=defaultdict(list)
    for r in rows:group[(r['seed'],r['scenario'],r['load'])].append(r)
    prefix_equal=all(same_prefix(g) for g in group.values())
    fixed_equal=all(all(r.get('post_wait_state')==g[0].get('post_wait_state') for r in g)
        for group_rows in group.values() for g in [[r for r in group_rows if r['window']=='1000ms']])
    inputs_ok=all(set(r.get('selection',{}).get('input',{}))=={'positions','fingers','reference_positions'}
                  for r in rows if 'selection' in r)
    model=c.StartGatedMLP(json.loads((ROOT/'direct-selected.json').read_text())['mlp'])
    replay=all(model(r['selection']['input'])==r['selection']['decision'] for r in rows
               if r['method']=='gated_direct' and 'selection' in r)
    params={load:next(r['physical_parameters'] for r in rows if r['load']==load) for load in c.LOADS}
    assert all(r['physical_parameters']==params[r['load']] for r in rows)
    native,heavy=params['native'],params['density150']
    for key in ('body_mass_kg','body_inertia_kg_m2','geometry_density_kg_per_m3'):
        np.testing.assert_allclose(heavy[key],np.array(native[key])*1.5,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(heavy['body_ipos'],native['body_ipos'],rtol=0,atol=1e-14)
    assert all(native[k]==heavy[k] for k in native if k not in
        ('body_mass_kg','body_inertia_kg_m2','geometry_density_kg_per_m3','total_object_mass_kg','body_ipos'))
    checks={'episodes':len(rows),'independent_initialization_seeds':20,'complete_prespecified_cohort':True,
        'paired_prefixes_equal_within_load':prefix_equal,'fixed_wait_poststates_match':fixed_equal,
        'all_inputs_permitted':inputs_ok,'frozen_mlp_decisions_reproduced':replay,
        'frozen_weights_thresholds_and_sources_unchanged':True,'mass_and_inertia_scale_verified':1.5,
        'native_mass_kg':native['total_object_mass_kg'],'shifted_mass_kg':heavy['total_object_mass_kg'],
        'friction_geometry_and_actuator_parameters_unchanged':True,
        'initialization_failures':sum(not r.get('nominal_ready',False) for r in rows),
        'invalid_fault_prefixes':sum(not r.get('fault_class_valid',False) for r in rows),
        'execution_errors':sum(bool(r.get('error')) for r in rows),
        'numerical_instabilities':sum(r.get('failure_reason')=='numerical_instability' for r in rows),
        'contact_anchors':sum(r['contact_anchor_count'] for r in rows),
        'post_initialization_state_edits':sum(r['state_edit_count_after_initialization'] for r in rows),
        'forbidden_calls':sum(len(r['blocked_calls']) for r in rows),
        'complete_cost_observations':sum('total_post_fault_seconds' in r for r in rows),
        'cost_identity':all(r['audit'].get('cost_identity',False) for r in rows if r['audit']['complete']),
        'wait_steps_match_trace':all(sum(f['stage'] in ('inference_wait','monitor_overrun_wait') for f in r['trace'])==r['inference_wait_steps'] for r in rows),
        'initial_choices_equal_across_windows':all(len({r['candidate'] for r in rows if
            (r['seed'],r['scenario'],r['load'],r['method'])==(seed,q,load,m)})==1
            for seed,q,load,m in itertools.product(p['test_seeds'],c.SCENARIOS,c.LOADS,c.METHODS))}
    assert all((prefix_equal,fixed_equal,inputs_ok,replay,checks['cost_identity'],checks['wait_steps_match_trace']))
    faults=[r for r in rows if r['scenario']!='sham'];healthy=[r for r in rows if r['scenario']=='sham']
    take=lambda source,load,window,m:[r for r in source if (r['load'],r['window'],r['method'])==(load,window,m)]
    result={'scope':p['scope'],'validity':checks,'fault_results':{},'healthy_controls':{},
            'paired_methods':{},'paired_waits':{},'paired_loads':{},'by_scenario':{}}
    for load in c.LOADS:
        result['fault_results'][load]={};result['healthy_controls'][load]={};result['paired_methods'][load]={}
        result['paired_waits'][load]={};result['by_scenario'][load]={}
        for w in c.WINDOWS:
            result['fault_results'][load][w]={m:metrics(take(faults,load,w,m)) for m in c.METHODS}
            result['healthy_controls'][load][w]={m:metrics(take(healthy,load,w,m)) for m in c.METHODS}
            result['paired_methods'][load][w]={f'{a}_vs_{b}':paired(take(faults,load,w,a),take(faults,load,w,b),p['test_seeds'])
                for a,b in itertools.combinations(c.METHODS,2)}
            result['by_scenario'][load][w]={q:{m:metrics(take([r for r in faults if r['scenario']==q],load,w,m))
                for m in c.METHODS} for q in c.SCENARIOS[:-1]}
        for m in c.METHODS:
            result['paired_waits'][load][m]=paired(take(faults,load,'1000ms',m),take(faults,load,'actual',m),p['test_seeds'])
    for w in c.WINDOWS:
        result['paired_loads'][w]={m:paired(take(faults,'density150',w,m),take(faults,'native',w,m),p['test_seeds']) for m in c.METHODS}
    transitions=[];waiting_changes=[];case_rows=[]
    by_key={(r['seed'],r['scenario'],r['method'],r['load'],r['window']):r for r in rows}
    for seed,q,m in itertools.product(p['test_seeds'],c.SCENARIOS,c.METHODS):
        for w in c.WINDOWS:
            a=by_key[(seed,q,m,'native',w)];b=by_key[(seed,q,m,'density150',w)]
            if a['candidate']!=b['candidate'] or safe(a)!=safe(b):
                transitions.append({'seed':seed,'scenario':q,'method':m,'window':w,'native_choice':a['candidate'],
                    'new_load_choice':b['candidate'],'native_label':a['audit']['label'],'new_load_label':b['audit']['label']})
        for load in c.LOADS:
            a=by_key[(seed,q,m,load,'actual')];b=by_key[(seed,q,m,load,'1000ms')]
            if a['audit']['label']!=b['audit']['label']:
                waiting_changes.append({'seed':seed,'scenario':q,'method':m,'load':load,
                    'actual_label':a['audit']['label'],'wait_label':b['audit']['label'],'choice':a['candidate']})
    for r in rows:
        case_rows.append({**{k:r.get(k) for k in ('seed','scenario','method','load','window','candidate','task_complete','success',
            'protection_violation','total_post_fault_seconds','skill_seconds','inference_wait_seconds','hold_seconds','selector_wall_seconds','failure_reason','error')},
            **{k:r['audit'].get(k) for k in ('label','V_obs','V_entry','protected_completion')},
            'keeper_loss_samples':r.get('protection',{}).get('keeper_grasp_lost_steps'),
            'maximum_registered_support_gap_ms':r.get('protection',{}).get('max_unsupported_seconds',0)*1000})
    result['load_choice_or_verdict_transitions']=transitions;result['wait_verdict_transitions']=waiting_changes
    result['learning_participation']={load:{'fault_executions':len([r for r in faults if r['load']==load and r['method']=='gated_direct']),
        'selected_non_global':sum(r['candidate'] in ('continue','R_left','R_right') for r in faults if r['load']==load and r['method']=='gated_direct'),
        'definition':'The original frozen MLP scores and gate choose the scope. Global is a directly scored candidate, not a constant fallback.'} for load in c.LOADS}
    c.save(ROOT/'report.json',result);c.save(ROOT/'physical-parameters.json',params)
    fields=['load','window','method','split','episodes','terminal_goals','accepted_completions','protected_completions',
        'mean_total_seconds','mean_wait_seconds','mean_skill_seconds','keeper_role_violations','shared_guard_violations','invalid_entry']
    with (ROOT/'main-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for load,w,m,split in itertools.product(c.LOADS,c.WINDOWS,c.METHODS,('fault','healthy')):
            source=result['fault_results'] if split=='fault' else result['healthy_controls'];values=source[load][w][m]
            writer.writerow({'load':load,'window':w,'method':m,'split':split,**{k:values[k] for k in fields[4:]}})
    for name,items in [('cases',case_rows),('load-transitions',transitions),('wait-transitions',waiting_changes)]:
        if items:
            with (ROOT/(name+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    print(json.dumps({'validity':checks,'fault_results':result['fault_results'],
        'healthy_protected':{load:{w:{m:v['protected_completions'] for m,v in d.items()} for w,d in ws.items()} for load,ws in result['healthy_controls'].items()},
        'wait_transitions':waiting_changes,'learning_participation':result['learning_participation']},indent=2))


if __name__=='__main__':main()
