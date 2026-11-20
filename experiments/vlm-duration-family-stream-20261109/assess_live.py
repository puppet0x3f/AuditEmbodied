"""Paired, all-wait live costs; learned admissions separate from baseline fallback."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def main():
    report=json.loads((ROOT/'closed-loop/report.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'closed-loop/trials.jsonl').read_text().splitlines()]
    assert len(rows)==report['episodes']==72
    assert report['frozen_unchanged'] and report['all_waits_audited']
    paired=defaultdict(dict)
    for row in rows:paired[row['seed'],row['scenario']][row['method']]=row
    assert len(paired)==24 and all(set(pair)=={'fixed60','family_set','family_control'} for pair in paired.values())
    safe=lambda r:r['success'] and not r['protection_violation']
    used=lambda r:bool(r.get('duration_selection',{}).get('decision',{}).get('model_admitted')) and not r.get('duration_selection',{}).get('baseline_fallback',False)
    comparisons={};cases=[]
    for name in ('family_set','family_control'):
        pairs=[(p['fixed60'],p[name]) for p in paired.values()]
        common_safe=[(a,b) for a,b in pairs if safe(a) and safe(b)]
        admitted=[b for a,b in pairs if used(b)]
        fallback=[b for a,b in pairs if b.get('duration_selection',{}).get('baseline_fallback',False)]
        timings=[t['pipeline_wall_seconds'] for a,b in pairs for t in b['async_timing'] if t['kind']=='tail']
        comparisons[name]={
            'paired_states':len(pairs),'both_safe_states':len(common_safe),
            'safe_gained_vs_fixed60':sum(safe(b) and not safe(a) for a,b in pairs),
            'safe_lost_vs_fixed60':sum(safe(a) and not safe(b) for a,b in pairs),
            'model_admitted_executions':len(admitted),'model_admitted_A':sum(safe(b) for b in admitted),
            'model_admitted_non_A':sum(not safe(b) for b in admitted),
            'model_admitted_process_violations':sum(b['protection_violation'] for b in admitted),
            'fallback_executions':len(fallback),'fallback_safe_successes':sum(safe(b) for b in fallback),
            'same_initial_wait_budget_states':sum(a['inference_wait_steps']-a.get('controller_compute_wait_steps',0)==
                b['inference_wait_steps']-b.get('controller_compute_wait_steps',0) for a,b in pairs),
            'all_state_mean_total_delta_seconds':mean(b['total_post_fault_seconds']-a['total_post_fault_seconds'] for a,b in pairs),
            'all_state_mean_wait_delta_seconds':mean(b['inference_wait_seconds']-a['inference_wait_seconds'] for a,b in pairs),
            'all_state_mean_skill_delta_seconds':mean(b['skill_seconds']-a['skill_seconds'] for a,b in pairs),
            'both_safe_fixed60_mean_seconds':mean(a['total_post_fault_seconds'] for a,b in common_safe),
            'both_safe_model_mean_seconds':mean(b['total_post_fault_seconds'] for a,b in common_safe),
            'both_safe_mean_delta_seconds':mean(b['total_post_fault_seconds']-a['total_post_fault_seconds'] for a,b in common_safe),
            'tail_calls':len(timings),'tail_mean_seconds':mean(timings),'tail_max_seconds':max(timings) if timings else None}
        seeds=sorted({seed for seed,scenario in paired})
        seed_deltas=np.asarray([[mean(p[name]['total_post_fault_seconds']-p['fixed60']['total_post_fault_seconds']
                for (s,q),p in paired.items() if s==seed),
            mean(int(safe(p[name]))-int(safe(p['fixed60'])) for (s,q),p in paired.items() if s==seed)] for seed in seeds])
        draws=np.random.default_rng(20261109).integers(0,len(seeds),size=(10000,len(seeds)))
        bootstrap=seed_deltas[draws].mean(axis=1)
        comparisons[name]['seed_cluster_bootstrap95']={
            'independent_seeds':len(seeds),'resamples':10000,
            'all_state_total_delta_seconds':np.percentile(bootstrap[:,0],[2.5,97.5]).tolist(),
            'safe_success_rate_delta':np.percentile(bootstrap[:,1],[2.5,97.5]).tolist()}
        for a,b in pairs:
            faults=[f for f in b['trace'] if f['phase']=='recovery' and b['candidate'] in ('R_left','R_right')
                and not f['evaluator']['holds'][1 if b['candidate']=='R_left' else 0]]
            cases.append({'seed':b['seed'],'scenario':b['scenario'],'method':name,
                'baseline_safe':safe(a),'model_safe':safe(b),'model_admitted':used(b),
                'baseline_fallback':b.get('duration_selection',{}).get('baseline_fallback',False),
                'duration_steps':b.get('local_steps',{}).get('local_relevel_steps'),
                'fixed60_total_seconds':a['total_post_fault_seconds'],'model_total_seconds':b['total_post_fault_seconds'],
                'first_keeper_loss_stage':faults[0]['stage'] if faults else None,
                'first_keeper_loss_seconds':faults[0]['time']-b['prefix_state']['time'] if faults else None,
                'failure_reason':b['failure_reason']})
    out={'independent_fault_states':24,'physical_episodes':72,'methods':report['methods'],
        'paired_comparisons':comparisons,'cases':cases,'frozen_unchanged':True,'all_waits_audited':True,
        'actual_method_first_counts':dict(Counter(next(iter(p)) for p in paired.values())),
        'execution_order_note':'Current72 uses fixed60 first on left and family_set first on right; family_control never first. Actual source is in closed-loop/source. Future CLI rotation was corrected after campaign; current results unchanged.',
        'scope':'All-state and both-safe paired costs reported separately. Every method uses the same frozen scope policy; no role oracle. Outcome model admissions, explicit rejection and fixed60 fallback are separate. These72 executions are test only; no counterfactual coverage is inferred from unexecuted candidates.'}
    (ROOT/'live-assessment.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'}))


if __name__=='__main__':main()
