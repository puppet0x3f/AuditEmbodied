"""Recompute full-path, suffix and native-rate views of the480executions."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
METHODS=('retarget','visual');LOADS=('native','density150');WINDOWS=('actual','1000ms')

def longest(flags):
    n=best=0
    for f in flags:n=n+1 if f else 0;best=max(best,n)
    return best

def mean(rows,k):return float(np.mean([r[k] for r in rows])) if rows else None
def ready(e):
    support=[bool(h and v<=.2) for h,v in zip(e['holds'],e['relative_speeds'])]
    result=bool(all(support) and e['clearance']>=.10 and e['tilt_deg']<=15. and e['handle_height_difference']<=.03)
    assert support==e['supports'] and result==e['ready']
    return result
def score(row):
    frames=[f for f in row['trace'] if f['phase'] in ('recovery','final_hold')]
    assert not row.get('error'),(row.get('seed'),row.get('error'))
    assert row['nominal_ready'] and row['fault_class_valid']
    assert not row['blocked_calls'] and row['contact_anchor_count']==row['state_edit_count_after_initialization']==0
    assert row['external_wrench_at_end']==0
    dt=np.diff([row['prefix_state']['time']]+[f['time'] for f in frames]);assert np.allclose(dt,.05,rtol=0,atol=1e-8)
    held=[f for f in frames if f['phase']=='final_hold']
    complete=len(held)>=20 and all(ready(f['evaluator']) for f in held)
    table=any(f['stage']=='global_table_settle' and f['evaluator']['table_contact'] and not any(f['evaluator']['holds']) for f in frames)
    lifted=any(f['stage'] in ('global_lift','global_lift_settle') and all(f['evaluator']['holds']) and f['evaluator']['clearance']>=.10 for f in frames)
    cycle=table and lifted
    accepted=complete and cycle and not row.get('abstained',False)
    unsupported=[not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames]
    run=longest(unsupported);bad=run>2 or bool(row['protection']['deep_interarm_contact_steps'])
    assert bool(complete)==row['task_complete'] and bool(accepted)==row['success']
    assert cycle==row['table_cycle_verified'] and table==row['table_setdown_verified']
    assert bad==row['protection_violation'] and abs(run/20-row['protection']['max_unsupported_seconds'])<1e-8
    total=len(frames)/20;wait=sum(f['stage'].startswith('vision_wait_') or f['stage'] in ('inference_wait','regrasp_observation_wait','controller_compute_wait') for f in frames)/20
    hold=len(held)/20;skill=total-wait-hold
    for k,v in [('total_post_fault_seconds',total),('inference_wait_seconds',wait),('hold_seconds',hold),('skill_seconds',skill)]:assert abs(row[k]-v)<1e-7,(k,row[k],v)
    pickup=[f['time'] for f in frames if f['stage'] in ('global_hover','vision_hover')]
    suffix=[f for f in frames if f['time']>=pickup[0]-1e-8] if pickup else []
    tail_bad=longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in suffix])>2 if suffix else True
    tail_ok=bool(complete and suffix and not tail_bad)
    prefix_hashes={}
    import hashlib
    for name in ('initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state'):
        prefix_hashes[name]=hashlib.sha256(json.dumps(row[name],sort_keys=True).encode()).hexdigest()
    path=Path(row['native_recording']['path']);path=path if path.is_absolute() else ROOT.parents[1]/path
    with np.load(path) as record:
        a=record['values'];fields=list(record['fields']);end=a[24::25]
        assert len(a)==25*len(row['trace']) and len(end)==len(row['trace'])
        assert np.allclose(np.diff(a[:,0]),.002,atol=1e-8,rtol=0)
        for i,f in enumerate(row['trace']):
            assert abs(end[i,0]-f['time'])<1e-8
            assert np.array_equal(end[i,3:6],[*f['evaluator']['holds'],f['evaluator']['table_contact']])
            assert np.allclose(end[i,6:11],[*f['evaluator']['pot_center'],f['evaluator']['tilt_deg'],f['evaluator']['clearance']],atol=1e-8,rtol=0)
        assert int(np.sum(end[end[:,1]>=2,11]>.01))==row['protection']['deep_interarm_contact_steps']
        b=a[a[:,1]>=2];flags=~(b[:,3].astype(bool)|b[:,4].astype(bool)|b[:,5].astype(bool))
        native_gap=longest(flags)*.002
        native_deep=float(np.max(b[:,11]));native_bad=native_gap>.10000001 or native_deep>.01
        unforced=flags&(b[:,12]<1e-6)
        phase_guard=[]
        for offset in range(25):
            sampled=b[offset::25]
            missing=~(sampled[:,3].astype(bool)|sampled[:,4].astype(bool)|sampled[:,5].astype(bool))
            phase_guard.append(bool(longest(missing)<=2 and np.max(sampled[:,11])<=.01))
        assert phase_guard[24]==(not bad)
        high={'native500_samples':len(a),'native500_gap_ms':native_gap*1000,
            'native500_deep_m':native_deep,'native500_bad':bool(native_bad),
            'native500_Y':bool(accepted and not native_bad),'native500_zero_force_gap_ms':longest(unforced)*2,
            'maximum_tilt_deg':float(np.max(b[:,9])),
            'maximum_downward_displacement_m':max(0.,row['post_fault']['pot_center'][2]-float(np.min(b[:,8]))),
            'maximum_center_speed_m_s':float(np.max(np.linalg.norm(np.diff(b[:,6:9],axis=0),axis=1)/.002)),
            'minimum_object_force_norm_sum_during_gap_N':float(np.min(b[flags,12])) if flags.any() else None,
            'phase20_guard_pass':phase_guard,'phase20_Y':[bool(accepted and v) for v in phase_guard]}
    result={k:row[k] for k in ('seed','scenario','load','window','method')}
    result.update(C=bool(complete),accepted=bool(accepted),Y=bool(accepted and not bad),label=row['label'],
        setdown=bool(table),cycle=bool(cycle),shared_bad=bool(bad),pickup_tail_ok=tail_ok,
        history_only_rejection=bool(tail_ok and accepted and bad),interface_rejection=bool(complete and not cycle),
        unsupported_ms=run*50,total=total,wait=wait,skill=skill,hold=hold,
        process_wall=row.get('post_fault_process_wall_seconds',row['wall_seconds']),full_process_wall=row['full_process_wall_seconds'],
        audit_wall=row['native_recording']['wall_seconds'],archive_wall=row['archive_wall_seconds'],
        path_m=row['recovery_path_m'],calls=len(row['skill_events']),
        grasp_attempts=len(row.get('controller_events',[])) if row['method']=='visual' else 1,
        visual_services=len(row.get('controller_services',[])) if row['method']=='visual' else int('regrasp_decision' in row),
        failure_reason=row['failure_reason'],prefix_hashes=prefix_hashes,**high)
    return result

def metrics(a):
    return {'episodes':len(a),**{k:sum(r[k] for r in a) for k in ('C','accepted','Y','setdown','cycle','shared_bad','pickup_tail_ok','history_only_rejection','interface_rejection','native500_Y')},
        'labels':dict(Counter(r['label'] for r in a)),
        **{k:mean(a,k) for k in ('total','wait','skill','hold','process_wall','full_process_wall','audit_wall','path_m','visual_services','grasp_attempts')},
        'native_verdict_changed':sum(r['Y']!=r['native500_Y'] for r in a),
        'native_guard_changed':sum(r['shared_bad']!=r['native500_bad'] for r in a),
        'phase20_Y_counts':[sum(r['phase20_Y'][j] for r in a) for j in range(25)],
        'phase20_Y_range':[min(sum(r['phase20_Y'][j] for r in a) for j in range(25)),
            max(sum(r['phase20_Y'][j] for r in a) for j in range(25))],
        'failure_reasons':dict(Counter(r['failure_reason'] for r in a if not r['Y']))}

def paired(a,axis,first,second):
    groups=defaultdict(dict)
    for r in a:groups[(r['seed'],r['scenario'])][r[axis]]=r
    assert all(first in g and second in g for g in groups.values())
    seeds=sorted({s for s,q in groups});draw=np.random.default_rng(20261112).integers(0,len(seeds),(2000,len(seeds)))
    reports={}
    counts=np.array([sum(s==key[0] for key in groups) for s in seeds])
    for key in ('Y','C','total','wait','process_wall','history_only_rejection'):
        differences=np.array([sum(float(g[first][key])-float(g[second][key]) for (s,q),g in groups.items() if s==seed) for seed in seeds])
        boot=differences[draw].sum(1)/counts[draw].sum(1)
        reports[key]={'difference':float(differences.sum()/counts.sum()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    common=[g for g in groups.values() if g[first]['Y'] and g[second]['Y']]
    return {'states':len(groups),'seeds':len(seeds),'first':first,'second':second,'axis':axis,
        'differences':reports,'Y_gains':sum(g[first]['Y'] and not g[second]['Y'] for g in groups.values()),
        'Y_losses':sum(g[second]['Y'] and not g[first]['Y'] for g in groups.values()),
        'common_protected_states':len(common),
        'common_protected_total':{n:float(np.mean([g[n]['total'] for g in common])) if common else None for n in (first,second)}}

def main():
    from campaign import verify
    p=json.loads((ROOT/'protocol.json').read_text());verify(p)
    scored=[];services=[]
    with (ROOT/'scored.jsonl').open('w') as out:
        for line in (ROOT/'trials.jsonl').open():
            row=json.loads(line);r=score(row);scored.append(r);out.write(json.dumps(r)+'\n')
            for d in row.get('controller_services',[row['regrasp_decision']] if 'regrasp_decision' in row else []):
                services.append({'method':row['method'],'seconds':d['service_wall_seconds'],'steps':d['service_wait_steps']})
    assert len(scored)==480 and {r['seed'] for r in scored}==set(p['seeds'])
    groups=defaultdict(list)
    for r in scored:groups[(r['seed'],r['scenario'],r['load'],r['window'])].append(r)
    assert len(groups)==240
    for key,group in groups.items():
        assert {r['method'] for r in group}==set(METHODS)
        assert group[0]['prefix_hashes']==group[1]['prefix_hashes'],key
    before_wait=defaultdict(list)
    for r in scored:before_wait[(r['seed'],r['scenario'],r['load'],r['method'])].append(r)
    for key,group in before_wait.items():
        assert len(group)==2
        assert all(group[0]['prefix_hashes'][k]==group[1]['prefix_hashes'][k]
            for k in group[0]['prefix_hashes'] if k!='post_wait_state'),key
    report={'episodes':480,'initialization_seeds':20,'paired_conditions':240,
        'checks':{'frozen_files_unchanged':True,'all_costs_recomputed':True,'matched_initial_fault_and_wait_states':True,
            'matched_fault_prefixes_across_wait_windows':True,
            'all_native500_endpoints_match20hz':True,'no_state_edits_or_anchors':True,'no_execution_errors':True},
        'fault_results':{},'healthy_controls':{},'executor_effects':{},'waiting_effects':{},'services':{},
        'history_cases':[r for r in scored if r['history_only_rejection']],
        'native_verdict_changes':[r for r in scored if r['Y']!=r['native500_Y']],
        'interpretation':'500Hz sensitivity retains20Hz terminal and interface predicates; only shared path guards are recomputed at native rate. Every episode has one registered global call; grasp attempts are internal stages, not separate skill calls.'}
    for l in LOADS:
        for w in WINDOWS:
            key=l+'/'+w;fault=[r for r in scored if r['load']==l and r['window']==w and r['scenario']!='sham']
            report['fault_results'][key]={m:metrics([r for r in fault if r['method']==m]) for m in METHODS}
            report['healthy_controls'][key]={m:metrics([r for r in scored if r['load']==l and r['window']==w and r['scenario']=='sham' and r['method']==m]) for m in METHODS}
            report['executor_effects'][key]=paired(fault,'method','visual','retarget')
        for m in METHODS:
            a=[r for r in scored if r['load']==l and r['method']==m and r['scenario']!='sham']
            report['waiting_effects'][l+'/'+m]=paired(a,'window','1000ms','actual')
    for m in METHODS:
        a=[x for x in services if x['method']==m]
        report['services'][m]={'calls':len(a),'median_seconds':float(np.median([x['seconds'] for x in a])),
            'max_seconds':max(x['seconds'] for x in a),'total_measured_seconds':sum(x['seconds'] for x in a),
            'physical_wait_seconds':sum(x['steps']/20 for x in a)}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'main-results.csv').open('w') as out:
        names=['load','window','method','episodes','C','accepted','Y','pickup_tail_ok','history_only_rejection','total','wait','process_wall','native500_Y','native_verdict_changed']
        writer=csv.DictWriter(out,fieldnames=names);writer.writeheader()
        for key,methods in report['fault_results'].items():
            for m,r in methods.items():writer.writerow({n:{'load':key.split('/')[0],'window':key.split('/')[1],'method':m}.get(n,r.get(n)) for n in names})
    print(json.dumps({k:report[k] for k in ('episodes','fault_results','healthy_controls','executor_effects','waiting_effects','services')},indent=2),flush=True)

if __name__=='__main__':main()
