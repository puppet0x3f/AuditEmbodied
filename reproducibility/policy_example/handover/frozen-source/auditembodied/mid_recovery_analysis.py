"""Paired physical suffixes of the actual independent closed-loop prefixes."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
import json
import multiprocessing
from pathlib import Path

from .duration_continuation import boundary_prefix,run_continuation
from .mid_recovery_data import outcome,read


def collect(live_dir,output,workers):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'trials.jsonl').exists():raise ValueError('counterfactual file exists')
    sources=[r for r in read(live_dir/'trials.jsonl') if r['method']=='mid_selective' and 'mid_selection' in r]
    jobs=[]
    for source in sources:
        chosen=source['mid_selection']['selected_branch']
        for branch in ('continue','R_global'):
            if branch!=chosen:jobs.append((source,branch))
    protocol={'source':str(live_dir.resolve()),'eligible_states':len(sources),'episodes':len(jobs),
        'scope':'New independent-test diagnostic continuations of the learned policy actual prefix. Replay every original wait and actuator command. Outcomes not available to deployed policy and never used for training.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    run=partial(run_continuation,transition='defer_rotation',account_control_latency=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (output/'trials.jsonl').open('w') as stream:
            for row in pool.map(run,jobs):
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
                    'outcome':outcome(row),'errors':row['diagnostic_errors']}),flush=True)
                if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
    (output/'report.json').write_text(json.dumps({**protocol,'all_prefixes_exact':True,'all_waits_audited':True},indent=2)+'\n')


def analyze(live_dir,counterfactual_dir):
    import numpy as np
    live=read(live_dir/'trials.jsonl');cf=read(counterfactual_dir/'trials.jsonl')
    lookup={}
    for r in cf:lookup.setdefault((r['seed'],r['scenario']),{})['local' if r['method']=='continue60' else 'global']=r
    cases=[]
    for source in live:
        if source['method']!='mid_selective' or 'mid_selection' not in source:continue
        mid=source['mid_selection'];branch=mid['selected_branch'];key=(source['seed'],source['scenario'])
        pair=dict(lookup[key]);selected={'continue':'local','R_global':'global'}.get(branch)
        if selected is not None:pair[selected]=source
        assert set(pair)=={'local','global'}
        labels={k:outcome(v) for k,v in pair.items()}
        decision=mid['decision'];records=decision.get('records',[])
        cases.append({'seed':source['seed'],'scenario':source['scenario'],'local_label':labels['local'],
            'global_label':labels['global'],'selected':branch,'model_admitted':bool(decision.get('model_admitted')) and not mid['baseline_fallback'],
            'fallback':mid['baseline_fallback'],'late':not mid['forecast_ready'],
            'stale':not mid['continuation_matches_forecast'],
            'prefix_already_violated':any(v>0 for k,v in source['continuation_diagnostic']['protection_before'].items()
                if k!='max_unsupported_seconds') or source['continuation_diagnostic']['protection_before']['max_unsupported_seconds']>.1+1e-8,
            'false_admissions':sum(r['accepted'] and labels[r['key']]!='A' for r in records) if not mid['baseline_fallback'] else 0,
            'model_safe_coverage':any(r['accepted'] and labels[r['key']]=='A' for r in records) if not mid['baseline_fallback'] else False,
            'needed':labels['local']!='A' and labels['global']=='A',
            'unnecessary_upgrade':branch=='R_global' and labels['local']=='A',
            'unsafe_continue':branch=='continue' and labels['local']!='A',
            'safe_success':source['success'] and not source['protection_violation'],
            'total_seconds':source['total_post_fault_seconds'],
            'same_prefix_local_seconds':pair['local']['total_post_fault_seconds'],
            'same_prefix_global_seconds':pair['global']['total_post_fault_seconds']})
    safe=[c for c in cases if c['local_label']=='A' or c['global_label']=='A']
    needed=[c for c in cases if c['needed']];no_safe=[c for c in cases if c not in safe]
    mid_report={'eligible_states':len(cases),'actual_label_pairs':dict(Counter(c['local_label']+c['global_label'] for c in cases)),
        'needed_upgrades':len(needed),'needed_upgrades_taken':sum(c['selected']=='R_global' for c in needed),
        'unnecessary_upgrades':sum(c['unnecessary_upgrade'] for c in cases),
        'unsafe_continues':sum(c['unsafe_continue'] for c in cases),
        'false_admissions':sum(c['false_admissions'] for c in cases),
        'safe_states':len(safe),'model_safe_coverage':sum(c['model_safe_coverage'] for c in safe),
        'safe_choices':sum(c['safe_success'] for c in cases),
        'no_safe_states':len(no_safe),'no_safe_rejected':sum(c['selected']=='reject' for c in no_safe),
        'safe_states_rejected':sum(c['selected']=='reject' for c in safe),
        'baseline_fallbacks':sum(c['fallback'] for c in cases),'late':sum(c['late'] for c in cases),
        'stale':sum(c['stale'] for c in cases)}
    mid_report['prefix_already_violated']=sum(c['prefix_already_violated'] for c in cases)
    mid_report['model_admitted_unsafe_choices']=sum(c['model_admitted'] and not c['safe_success'] for c in cases)
    prediction_rows=[]
    for r in live:
        if r['method']!='mid_selective' or 'mid_selection' not in r:continue
        pair=next(c for c in cases if (c['seed'],c['scenario'])==(r['seed'],r['scenario']))
        for record in r['mid_selection']['decision'].get('records',[]):
            prediction='ABCD'[int(np.argmax(record['conditional_probabilities']))]
            prediction_rows.append({'seed':r['seed'],'scenario':r['scenario'],'candidate':record['key'],
                'label':pair[record['key']+'_label'],'prediction':prediction,'accepted':record['accepted'],
                'joint_A_probability':record['class_probabilities'][0]})
    mid_report.update(candidate_count=len(prediction_rows),
        candidate_class_correct=sum(r['label']==r['prediction'] for r in prediction_rows),
        unsafe_candidate_count=sum(r['label']!='A' for r in prediction_rows),
        candidate_admissions=sum(r['accepted'] for r in prediction_rows),
        safe_candidate_admissions=sum(r['accepted'] and r['label']=='A' for r in prediction_rows))
    with (live_dir/'paired-candidate-predictions.csv').open('w') as f:
        if prediction_rows:
            writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0]));writer.writeheader();writer.writerows(prediction_rows)
    mid_report['same_prefix_continue_only_safe_successes']=sum(c['local_label']=='A' for c in cases)
    if cases:
        mid_report['same_prefix_continue_only_mean_seconds']=float(np.mean([c['same_prefix_local_seconds'] for c in cases]))
        mid_report['selective_eligible_mean_seconds']=float(np.mean([c['total_seconds'] for c in cases]))
    groups={}
    for r in live:groups.setdefault((r['seed'],r['scenario']),{})[r['method']]=r
    comparisons={}
    rng=np.random.default_rng(20261109)
    for reference in ('family_set','mid_always'):
        paired=[p for p in groups.values() if p['mid_selective']['success'] and not p['mid_selective']['protection_violation'] and p[reference]['success'] and not p[reference]['protection_violation']]
        if not paired:
            comparisons[reference]={'common_safe_states':0};continue
        a=np.array([p['mid_selective']['total_post_fault_seconds'] for p in paired]);b=np.array([p[reference]['total_post_fault_seconds'] for p in paired])
        seeds=sorted({p['mid_selective']['seed'] for p in paired});d={s:[] for s in seeds}
        for p in paired:d[p['mid_selective']['seed']].append((p['mid_selective']['total_post_fault_seconds'],p[reference]['total_post_fault_seconds']))
        draws=[]
        for _ in range(2000):
            x=np.array([v for s in rng.choice(seeds,len(seeds),replace=True) for v in d[int(s)]])
            draws.append(1-x[:,0].mean()/x[:,1].mean())
        comparisons[reference]={'common_safe_states':len(paired),'selective_mean_seconds':float(a.mean()),
            'reference_mean_seconds':float(b.mean()),'time_reduction':float(1-a.mean()/b.mean()),
            'seed_bootstrap_95_interval':np.quantile(draws,[.025,.975]).tolist()}
    with (live_dir/'paired-suffixes.csv').open('w') as f:
        if cases:
            writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    residual={}
    for method in ('family_set','mid_always','mid_selective'):
        a=[r for r in live if r['method']==method and not (r['success'] and not r['protection_violation'])]
        residual[method]=[{'seed':r['seed'],'scenario':r['scenario'],
            'initial_relevel_steps':r.get('local_steps',{}).get('local_relevel_steps'),
            'mid_branch':r.get('mid_selection',{}).get('selected_branch'),
            'success':r['success'],'protection':r['protection']} for r in a]
    result={'paired_suffix_diagnostics':mid_report,'common_safe_cost_comparisons':comparisons,
        'residual_unsafe_episodes':residual,
        'cost_note':'Full post-fault physical cost including every inference/observation/controller wait and final hold. Different policies can have different actual initial inference holds; necessity labels use the selective policy EXACT prefix.'}
    (live_dir/'paired-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live-dir',required=True)
    parser.add_argument('--counterfactual-dir',required=True);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--analyze-only',action='store_true');args=parser.parse_args()
    live=Path(args.live_dir);cf=Path(args.counterfactual_dir)
    if not args.analyze_only:collect(live,cf,args.workers)
    analyze(live,cf)


if __name__=='__main__':main()
