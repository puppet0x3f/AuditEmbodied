"""Independent reconstruction of full handover audit and paired statistics."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
METHODS=['visual','proprio','rule','R3']
WINDOWS=['actual','1000ms','2000ms']


def longest(flags):
    n=m=0
    for x in flags:n=n+1 if x else 0;m=max(m,n)
    return m


def safe(row):return bool(row['audit']['protected_completion'])


def mean(rows,key):
    values=[r['audit'].get(key) for r in rows]
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def metrics(rows):
    service=[e['service_wall_seconds'] for r in rows for e in r['decision_events']]
    return {'episodes':len(rows),'terminal_completions':sum(r['audit'].get('terminal_completion',False) for r in rows),
        'protected_completions':sum(safe(r) for r in rows),
        'legacy_protected_completions':sum(r['audit'].get('legacy_protected_completion',False) for r in rows),
        'shared_guard_violations':sum(r['audit'].get('shared_guard_violation',False) for r in rows),
        'role_violation_episodes':sum(r['audit'].get('role_violation',False) for r in rows),
        'interface_violation_episodes':sum(r['audit'].get('interface_violation',False) for r in rows),
        'invalid_entry_episodes':sum(not r['audit'].get('all_entry_prerequisites',False) for r in rows),
        'labels':dict(Counter(r['audit']['label'] for r in rows)),
        'complete_cost_observations':sum(r['audit']['total_seconds'] is not None for r in rows),
        'mean_total_seconds':mean(rows,'total_seconds'),'mean_wait_seconds':mean(rows,'wait_seconds'),
        'mean_recovery_motion_seconds':mean(rows,'recovery_motion_seconds'),
        'mean_release_seconds':mean(rows,'release_seconds'),'mean_verification_seconds':mean(rows,'verification_seconds'),
        'local_then_global_episodes':sum(r['audit'].get('local_then_global',False) for r in rows),
        'top_level_sequences':dict(Counter(' -> '.join(r['audit'].get('top_level_sequence',[])) for r in rows)),
        'global_calls':sum(c['skill']=='R3' for r in rows for c in r['skill_calls']),
        'global_pickup_attempts':sum(r['audit'].get('global_pickup_attempts',0) for r in rows),
        'global_search_retry_episodes':sum(any(len(c.get('pickup_attempts',[]))>1 for c in r['skill_calls']) for r in rows),
        'decision_calls':sum(r['audit'].get('decision_calls',0) for r in rows),
        'mean_decision_service_seconds':float(np.mean(service)) if service else None,
        'median_decision_service_seconds':float(np.median(service)) if service else None,
        'p95_decision_service_seconds':float(np.quantile(service,.95)) if service else None,
        'fixed_window_exceeded_events':sum(r['audit'].get('fixed_window_exceeded',0) for r in rows),
        'controller_overrun_steps':sum(r['controller_overrun_steps'] for r in rows),
        'valid_at_observation_invalid_at_entry':sum(r['audit'].get('observation_valid_entry_invalid_decisions',0) for r in rows),
        'max_registered_support_gap_ms':max((r['audit'].get('maximum_registered_support_gap_ms',0) for r in rows),default=0),
        'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in rows])) if rows else None}


def paired(rows,first,second,by='method'):
    groups=defaultdict(dict)
    for r in rows:groups[(r['seed'],r['scenario'],r['window'] if by=='method' else r['method'])][r[by]]=r
    pairs=[(key,g[first],g[second]) for key,g in groups.items() if first in g and second in g]
    seeds=sorted({k[0] for k,_,_ in pairs})
    draw=np.random.default_rng(20261109).integers(0,len(seeds),(2000,len(seeds)))
    difference=np.array([np.mean([int(safe(a))-int(safe(b)) for k,a,b in pairs if k[0]==s]) for s in seeds])
    common=[(k,a,b) for k,a,b in pairs if safe(a) and safe(b)]
    result={'first':first,'second':second,'paired_states':len(pairs),'both_protected_states':len(common),
        'protected_rate_difference':float(difference.mean()),
        'protected_rate_difference_ci95':np.quantile(difference[draw].mean(1),[.025,.975]).tolist(),
        'protected_gains':sum(safe(a) and not safe(b) for _,a,b in pairs),
        'protected_losses':sum(safe(b) and not safe(a) for _,a,b in pairs)}
    if common:
        a,b=[float(np.mean([r[i]['audit']['total_seconds'] for r in common])) for i in (1,2)]
        totals=np.array([[sum(a['audit']['total_seconds'] for k,a,b in common if k[0]==s),
                          sum(b['audit']['total_seconds'] for k,a,b in common if k[0]==s)] for s in seeds])
        sample=totals[draw].sum(1);use=sample[:,1]>0
        ratios=1-sample[use,0]/sample[use,1]
        result['total_seconds']={'first_mean':a,'second_mean':b,'reduction_fraction':1-a/b,
                                'reduction_ci95':np.quantile(ratios,[.025,.975]).tolist()}
    return result


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=json.loads((ROOT/'protocol.json').read_text());frozen=json.loads((ROOT/'formal-freeze.json').read_text())
    assert (ROOT/'complete.json').exists()
    assert hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()==frozen['protocol_sha256']
    for path,digest in frozen['sources'].items():assert hashlib.sha256((ROOT/'frozen-source'/path).read_bytes()).hexdigest()==digest
    for path,digest in p['model_hashes'].items():assert hashlib.sha256((ROOT/'models'/path).read_bytes()).hexdigest()==digest
    for path,digest in p['preserved_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    rows=[];role_reasons=[]
    with (ROOT/'trials.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);trace=row['trace'];a=row['audit']
            if a['complete']:
                frames=[f for f in trace if f['phase'] in ('recovery','release','hold')]
                missing=[not (f['evaluator']['giver_holds'] or f['evaluator']['receiver_holds'] or f['evaluator']['table_contact']) for f in frames]
                assert longest(missing)*50==a['maximum_registered_support_gap_ms']
                role_bad=[]
                for frame in frames:
                    role=frame['audit']['role_obligation'];truth=frame['evaluator']
                    ok=bool(truth[role]) if role else True
                    assert ok==frame['audit']['role_obligation_satisfied']
                    if not ok:
                        reason=('giver_contact_missing' if role=='giver_holds' else 'receiver_contact_missing' if not truth['receiver_holds']
                            else 'receiver_clearance_below_threshold' if truth['clearance']<=truth['height_threshold'] else 'receiver_relative_motion')
                        role_bad.append(reason)
                assert len(role_bad)==a['role_loss_samples'];row['role_failure_reasons']=dict(Counter(role_bad))
                by_time={f['time']:f for f in trace}
                for event in row['decision_events']:
                    assert event['observed_time']==event['start_time']
                    needed=event['required_entry_predicate']
                    before=by_time[event['start_time']]['evaluator'];after=by_time[event['end_time']]['evaluator']
                    assert event['prerequisite_at_observation']==(bool(before[needed]) if needed else True)
                    assert event['prerequisite_at_entry']==(bool(after[needed]) if needed else True)
                    timed=[f for f in frames if event['start_time']<f['time']<=event['end_time']]
                    assert sum(f['stage']=='decision_wait' for f in timed)==event['physical_wait_steps']
                assert sum(f['stage']=='decision_wait' for f in frames)==sum(e['physical_wait_steps'] for e in row['decision_events'])
                assert sum(f['stage']=='controller_compute_wait' for f in frames)==row['controller_overrun_steps']
                assert a['cost_identity'] and abs(a['total_seconds']-sum(a[k] for k in ['wait_seconds','recovery_motion_seconds','release_seconds','verification_seconds']))<1e-7
                valid=all(c['valid_at_entry'] for c in row['skill_calls']) and all(d['prerequisite_at_entry'] for d in row['decision_events'])
                assert valid==a['all_entry_prerequisites']
                for call in row['skill_calls']:
                    assert call['valid_at_entry']==(bool(call['start_evaluator'][call['prerequisite']]) if call['prerequisite'] else True)
                if row['method'] in ('visual','proprio'):
                    with np.load(row['policy_input_file']) as inputs:
                        assert inputs['robot'].shape==(len(row['decision_events']),46)
                        if row['method']=='visual':assert inputs['rgb'].shape==(len(row['decision_events']),4,64,64,3)
                        else:assert inputs['rgb'].size==0
            row.pop('trace');rows.append(row)
    assert len(rows)==720
    groups=defaultdict(list)
    for row in rows:groups[(row['seed'],row['scenario'])].append(row)
    assert len(groups)==60 and all(len(v)==12 for v in groups.values())
    all_reach_prefix=all('prefix_state' in r for r in rows)
    prefix_equal=all_reach_prefix and all(all(r[k]==g[0][k] for r in g for k in ['initialization_state','initialization_end_state','prefix_state']) for g in groups.values())
    fixed_matches={}
    for window in ('1000ms','2000ms'):
        matching=overrun=0
        for group in groups.values():
            subset=[r for r in group if r['window']==window]
            if not all(r['decision_events'] for r in subset):continue
            if any(r['decision_events'][0]['service_exceeds_fixed_window'] or r['decision_events'][0]['controller_wait_steps'] for r in subset):overrun+=1;continue
            assert all(r['decision_events'][0]['post_wait_state']==subset[0]['decision_events'][0]['post_wait_state'] for r in subset)
            matching+=1
        fixed_matches[window]={'exact_initial_postwait_groups':matching,'groups_with_wait_overruns':overrun}
    initial_choice_equal=all(len({r['decision_events'][0]['plan'] for r in g if r['method']==m and r['decision_events']})<=1 for g in groups.values() for m in METHODS)
    data={window:{split:{m:metrics([r for r in rows if r['window']==window and r['method']==m and (r['scenario']=='nominal')==(split=='nominal')]) for m in METHODS}
           for split in ['fault','nominal']} for window in WINDOWS}
    comparisons={w:{m:paired([r for r in rows if r['window']==w and r['scenario']!='nominal'],'visual',m) for m in ['proprio','rule','R3']} for w in WINDOWS}
    baseline_comparisons={w:{f'{a}_vs_{b}':paired([r for r in rows if r['window']==w and r['scenario']!='nominal'],a,b)
                               for a,b in [('rule','R3'),('proprio','R3'),('rule','proprio')]} for w in WINDOWS}
    scenario_results={w:{s:{m:metrics([r for r in rows if r['window']==w and r['scenario']==s and r['method']==m])
                            for m in METHODS} for s in p['scenarios']} for w in WINDOWS}
    retries={w:{m:metrics([r for r in rows if r['window']==w and r['scenario']!='nominal' and r['method']==m
                           and r['audit'].get('local_then_global')]) for m in METHODS} for w in WINDOWS}
    effects={m:paired([r for r in rows if r['method']==m and r['scenario']!='nominal'],'2000ms','1000ms',by='window') for m in METHODS}
    measured_effects={w:{m:paired([r for r in rows if r['method']==m and r['scenario']!='nominal'],w,'actual',by='window')
                         for m in METHODS} for w in ['1000ms','2000ms']}
    transitions=[]
    for group in groups.values():
        for method in METHODS:
            pair={r['window']:r for r in group if r['method']==method}
            if pair['1000ms']['audit']['label']!=pair['2000ms']['audit']['label']:
                transitions.append({'seed':pair['1000ms']['seed'],'scenario':pair['1000ms']['scenario'],'method':method,
                    '1s':pair['1000ms']['audit']['label'],'2s':pair['2000ms']['audit']['label'],
                    '1s_total':pair['1000ms']['audit']['total_seconds'],'2s_total':pair['2000ms']['audit']['total_seconds']})
    validity={'episodes':len(rows),'seeds':len({r['seed'] for r in rows}),'paired_prefix_groups':len(groups),
        'all_initial_and_fault_prefixes_identical':prefix_equal,'all_initial_choices_equal_across_wait_windows':initial_choice_equal,
        'all_episodes_reach_fault_prefix':all_reach_prefix,
        'fixed_window_state_pairing':fixed_matches,'complete_costs':sum(r['audit']['complete'] for r in rows),
        'initialization_failures':sum(not r.get('initialization_supported',False) for r in rows),
        'execution_errors':sum(bool(r.get('error')) for r in rows),'numerical_instabilities':sum(r.get('failure_reason')=='numerical_instability' for r in rows),
        'state_edits':sum(r['state_edit_count_after_initialization'] for r in rows),'contact_anchors':sum(r['contact_anchor_count'] for r in rows),
        'forbidden_calls':sum(len(r['blocked_calls']) for r in rows),'external_wrench_leak_steps':sum(r['external_wrench_leak_steps'] for r in rows),
        'first_position_counts':{w:{q:dict(Counter(r['method'] for r in rows if r['window']==w and r['scenario']==q and r['execution_position']==0)) for q in p['scenarios']} for w in WINDOWS},
        'models_sources_protocol_unchanged':True,'observations_prerequisites_waits_and_costs_reconstructed':True}
    report={'scope':p['scope'],'validity':validity,'methods':data,'scenario_results':scenario_results,
        'local_then_global_fault_paths':retries,'visual_paired_fault_comparisons':comparisons,
        'baseline_paired_fault_comparisons':baseline_comparisons,'controlled_vs_measured_fault_effects':measured_effects,
        'controlled_2s_vs1s_fault_effects':effects,'controlled_outcome_transitions':transitions,
        'failures':[{'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],'window':r['window'],'audit':r['audit'],
            'role_failure_reasons':r.get('role_failure_reasons',{}),'failure_reason':r['failure_reason']} for r in rows if not safe(r)],
        'statistical_notes':['Bootstrap resamples20initialization seeds, preserving correlated conditions.','Equal observed protected sets and a degenerate empirical interval do not establish population equality or formal noninferiority.',
            'Controlled windows are minima per high-level decision; cumulative wait depends on the number of decisions. Measured service or control overruns are physically executed and charged.',
            'Full path audit adds local and carrier obligations to the unchanged legacy release/registered-support criteria. Historical control-time results retain their original protocol.','Registered-support absence is not literal contact-free flight.']}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['window','split','method','episodes','terminal_completions','protected_completions','legacy_protected_completions','shared_guard_violations','role_violation_episodes','invalid_entry_episodes','mean_total_seconds','mean_wait_seconds','mean_recovery_motion_seconds','mean_release_seconds','mean_verification_seconds','local_then_global_episodes','global_search_retry_episodes']
    csv_write(ROOT/'main-results.csv',[{'window':w,'split':s,'method':m,**{k:v[k] for k in columns[3:]}} for w,ss in data.items() for s,mm in ss.items() for m,v in mm.items()])
    csv_write(ROOT/'cases.csv',[{'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],'window':r['window'],'label':r['audit']['label'],
        'terminal':r['audit'].get('terminal_completion',False),'protected':safe(r),'total_seconds':r['audit']['total_seconds'],
        'wait_seconds':r['audit'].get('wait_seconds'),'sequence':' -> '.join(r['audit'].get('top_level_sequence',[])),
        'role_failure_reasons':json.dumps(r.get('role_failure_reasons',{}))} for r in rows])
    (ROOT/'episodes-without-traces.json').write_text(json.dumps(rows)+'\n')
    print(json.dumps({'validity':validity,'methods':data,'visual_comparisons':comparisons,'waiting':effects}),flush=True)


if __name__=='__main__':main()
