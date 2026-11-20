"""From executed frames to protected outcomes and paired component effects."""
from collections import Counter,defaultdict
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT))
from geometry import fit,mask_features,CAMERAS
spec=importlib.util.spec_from_file_location('offline',REPO/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
offline.LIFT_WAIT=(*offline.LIFT_WAIT,'regrasp_observation_wait')
METHODS=('fixed','refresh','retarget')
LABELS=('Fixed','Refresh only','Retarget')
LOADS=('native','density150')
WINDOWS=('actual','1000ms')


def metrics(rows):
    return {'episodes':len(rows),'C':sum(r['C'] for r in rows),'Y':sum(r['Y'] for r in rows),
        'labels':dict(Counter(r['label'] for r in rows)),
        'setdown':sum(r['table'] for r in rows),'ever_dual_regrasp':sum(r['ever_dual_regrasp'] for r in rows),
        'close_dual_grasp':sum(all(r['close_holds']) for r in rows),
        'pretarget_support_violations':sum(r['pretarget_support_violation'] for r in rows),
        'completed_with_prior_violation':sum(r['C'] and r['pretarget_support_violation'] for r in rows),
        **{k:float(np.mean([r[k] for r in rows])) for k in ('total','wait','skill','hold','path_m')},
        'close_worst_miss_cm':float(np.mean([r['close_miss_m'] for r in rows]))*100,
        'close_grasp_histogram':dict(Counter(sum(r['close_holds']) for r in rows))}


def paired(rows,first,second):
    group=defaultdict(dict)
    for r in rows:group[(r['seed'],r['scenario'])][r['method']]=r
    seeds=sorted({k[0] for k in group});n=len(group)
    differences=np.array([sum(int(g[first]['Y'])-int(g[second]['Y']) for k,g in group.items() if k[0]==s) for s in seeds])
    draw=np.random.default_rng(20261111).integers(0,len(seeds),(2000,len(seeds)))
    sample=differences[draw].sum(1)/(2*len(seeds))
    common=[g for g in group.values() if g[first]['Y'] and g[second]['Y']]
    return {'states':n,'seeds':len(seeds),'first':first,'second':second,
        'Y_difference':float(differences.sum()/n),'Y_difference_ci95':np.quantile(sample,[.025,.975]).tolist(),
        'Y_gains':sum(g[first]['Y'] and not g[second]['Y'] for g in group.values()),
        'Y_losses':sum(g[second]['Y'] and not g[first]['Y'] for g in group.values()),
        'all_attempt_time_difference':float(np.mean([g[first]['total']-g[second]['total'] for g in group.values()])),
        'common_protected_states':len(common),
        'common_protected_time_difference':float(np.mean([g[first]['total']-g[second]['total'] for g in common])) if common else None}


def score(row):
    result=offline.score_lift(row,'interface')
    assert result['label']==row['audit']['label']
    fs=[f for f in row['trace'] if f['phase']=='recovery']
    prefix=[f for f in row['trace'] if f['time']<=row['prefix_state']['time']+1e-8][-1]
    positions=np.array([prefix['robot']['positions']]+[f['robot']['positions'] for f in fs])
    path=float(np.linalg.norm(np.diff(positions,axis=0),axis=2).sum())
    assert abs(path-row['recovery_path_m'])<1e-8
    closed=[f for f in fs if f['stage']=='global_close'][-1]
    hover=next(f['time'] for f in fs if f['stage']=='global_hover')
    before=[f for f in fs if f['time']<hover-1e-8]
    prior=offline.longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in before])>2
    result.update(ever_dual_regrasp=any(all(f['evaluator']['holds']) for f in row['trace'] if f['stage'] in
        ('global_close','global_lift','global_lift_settle','final_stable_hold')),
        close_holds=closed['evaluator']['holds'],
        close_miss_m=float(np.linalg.norm(np.asarray(closed['robot']['positions'])-closed['evaluator']['handle_positions'],axis=1).max()),
        close_tilt_deg=closed['evaluator']['tilt_deg'],path_m=path,pretarget_support_violation=prior)
    return result


def tables(report):
    main=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Wait & Policy & Native $Y$ & Heavy $Y$ & Healthy $Y$ & Total (s) \\',r'\midrule']
    full=[r'\begin{tabular}{lllrrrrrrr}',r'\toprule',r'Load & Wait & Policy & $C$ & $Y$ & Dual & Healthy $Y$ & Wait (s) & Total (s) & Miss (cm) \\',r'\midrule']
    for load in LOADS:
        for window,wname in zip(WINDOWS,('Actual','1 s')):
            for method,label in zip(METHODS,LABELS):
                m=report['fault_results'][load][window][method];h=report['healthy_controls'][load][window][method]
                full.append(f"{'Native' if load=='native' else 'Heavy'} & {wname} & {label} & {m['C']}/40 & {m['Y']}/40 & {m['ever_dual_regrasp']}/40 & {h['Y']}/20 & {m['wait']:.3f} & {m['total']:.3f} & {m['close_worst_miss_cm']:.2f} "+r'\\')
    for window,wname in zip(WINDOWS,('Actual','1 s')):
        for method,label in zip(METHODS,LABELS):
            native=report['fault_results']['native'][window][method];heavy=report['fault_results']['density150'][window][method]
            healthy=report['healthy_controls']['density150'][window][method]
            main.append(f"{wname if method=='fixed' else ''} & {label} & {native['Y']}/40 & {heavy['Y']}/40 & {healthy['Y']}/20 & {heavy['total']:.3f} "+r'\\')
    return {n:'\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n' for n,lines in [('interface-retarget',main),('interface-full',full)]}


def analyze(raw,check_images=True):
    groups=defaultdict(dict);scored=[];poses=[];services=[];changed=[]
    for row in raw:
        assert not row.get('error') and row['nominal_ready'] and row['fault_class_valid']
        assert not row['blocked_calls'] and row['contact_anchor_count']==row['state_edit_count_after_initialization']==0
        r=score(row);scored.append(r)
        key=(row['seed'],row['scenario'],row['load'],row['window']);groups[key][row['method']]=row
        if 'regrasp_decision' in row:
            d=row['regrasp_decision'];est=fit(d['input'])
            assert est['valid']==d['estimate']['valid']
            for k in ('center','rotation','targets','optimization_vector'):
                assert np.allclose(est[k],d['estimate'][k],atol=1e-10,rtol=0),k
            assert d['service_wait_steps']==40 and d['service_wall_seconds']<=2
            if check_images:
                import imageio.v3 as iio
                import hashlib
                for i,camera in enumerate(CAMERAS):
                    rgb=iio.imread(ROOT/d['image_directory']/f'{camera}.png')
                    assert hashlib.sha256(rgb.tobytes()).hexdigest()==d['rgb_sha256'][i]
                    assert mask_features(rgb)==d['input']['views'][i]['colors']
            services.append(d['service_wall_seconds'])
            if row['method']=='retarget':
                e=d['estimate'];h=np.asarray(e['center'])+np.asarray([[0.,.155,.06],[0.,-.155,.06]])@np.asarray(e['rotation']).T
                entry=next(f for f in row['trace'][::-1] if f['stage']=='regrasp_observation_wait')['evaluator']
                poses.append({'seed':row['seed'],'scenario':row['scenario'],'load':row['load'],'window':row['window'],
                    'handle_error_mm':np.asarray(d['estimated_handle_error_m'])*1000,
                    'entry_handle_error_mm':np.linalg.norm(h-np.asarray(entry['handle_positions']),axis=1)*1000,
                    'fixed_miss_cm':max(d['fixed_grip_to_handle_error_m'])*100,
                    'retarget_miss_cm':max(d['retarget_grip_to_handle_error_m'])*100,
                    'observed_tilt':d['observation_truth']['tilt_deg'],'label':r['label']})
    for group in groups.values():
        assert set(group)==set(METHODS)
        for name in ('initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state'):
            assert all(group[m][name]==group['fixed'][name] for m in METHODS)
        prefixes=[]
        for m in METHODS:
            r=group[m];end=next(f['time'] for f in r['trace'][::-1] if f['stage']=='global_clear')
            prefixes.append([f for f in r['trace'] if f['time']<=end+1e-8])
        assert all(prefix==prefixes[0] for prefix in prefixes)
        a,b=group['refresh'],group['retarget']
        assert a['regrasp_decision']['input']==b['regrasp_decision']['input']
        assert a['regrasp_decision']['rgb_sha256']==b['regrasp_decision']['rgb_sha256']
        af=[f for f in a['trace'] if f['stage']=='regrasp_observation_wait']
        bf=[f for f in b['trace'] if f['stage']=='regrasp_observation_wait']
        assert af==bf
        if a['audit']['label']!=b['audit']['label']:
            changed.append({'seed':a['seed'],'scenario':a['scenario'],'load':a['load'],'window':a['window'],
                'fixed':group['fixed']['audit']['label'],'refresh':a['audit']['label'],'retarget':b['audit']['label']})
    report={'episodes':len(raw),'independent_seeds':len({r['seed'] for r in raw}),
        'validity':{'matched_fault_and_clear_prefixes':len(groups),'matched_rgb_and_inspection_hold_groups':len(groups),
            'pose_decisions_recomputed':len(services),'maximum_callback_seconds':max(services),'all_costs_complete':len(raw),
            'errors':0,'forbidden_calls':0,'post_initialization_state_edits':0},
        'fault_results':{},'healthy_controls':{},'paired_effects':{},'pose_quality':{},'outcome_changes':changed}
    for load in LOADS:
        report['fault_results'][load]={};report['healthy_controls'][load]={};report['paired_effects'][load]={}
        for w in WINDOWS:
            faults=[r for r in scored if r['load']==load and r['window']==w and r['scenario']!='sham']
            report['fault_results'][load][w]={m:metrics([r for r in faults if r['method']==m]) for m in METHODS}
            report['healthy_controls'][load][w]={m:metrics([r for r in scored if r['load']==load and r['window']==w and r['scenario']=='sham' and r['method']==m]) for m in METHODS}
            report['paired_effects'][load][w]={f'{a}_vs_{b}':paired(faults,a,b) for a,b in [('retarget','refresh'),('refresh','fixed'),('retarget','fixed')]}
        p=[r for r in poses if r['load']==load and r['scenario']!='sham']
        errors=np.concatenate([r['handle_error_mm'] for r in p])
        entry_errors=np.concatenate([r['entry_handle_error_mm'] for r in p])
        report['pose_quality'][load]={'observations':len(p),'handle_error_mm_median':float(np.median(errors)),
            'handle_error_mm_p95':float(np.quantile(errors,.95)),
            'entry_handle_error_mm_median':float(np.median(entry_errors)),
            'entry_handle_error_mm_p95':float(np.quantile(entry_errors,.95)),
            'fixed_worst_target_error_cm_mean':float(np.mean([r['fixed_miss_cm'] for r in p])),
            'retarget_worst_target_error_cm_mean':float(np.mean([r['retarget_miss_cm'] for r in p]))}
    return report,scored,poses


def main():
    raw=[json.loads(l) for l in (ROOT/'trials.jsonl').open()]
    assert len(raw)==720
    report,scored,poses=analyze(raw)
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,content in tables(report).items():(ROOT/f'{name}.tex').write_text(content)
    with (ROOT/'main-results.csv').open('w') as f:
        keys=['load','window','method','episodes','C','Y','total','wait','path_m','ever_dual_regrasp','close_worst_miss_cm']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for l in LOADS:
            for w in WINDOWS:
                for m in METHODS:
                    r=report['fault_results'][l][w][m];writer.writerow({k:({'load':l,'window':w,'method':m}.get(k,r.get(k))) for k in keys})
    with (ROOT/'scored.jsonl').open('w') as f:
        for r in scored:f.write(json.dumps(r)+'\n')
    with (ROOT/'pose-cases.jsonl').open('w') as f:
        for r in poses:f.write(json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()})+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
