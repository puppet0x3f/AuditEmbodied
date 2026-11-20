"""Independent physical results with paired safety/cost and fallback attribution."""
from collections import Counter,defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from auditembodied.l4_selector import safe
from auditembodied.admissible_mlp import local_precondition
from auditembodied.p0 import method_metrics,paired_statistics,same_prefix,outcome

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def used(row):
    return bool(row.get('duration_selection',{}).get('decision',{}).get('model_admitted')) and not row.get('duration_selection',{}).get('baseline_fallback',False)


def joint_label(row):
    start=local_precondition(row)
    return 'E' if start and not start['keeper_present_at_start'] else outcome(row)


def main():
    output=ROOT/'closed-loop'
    p=json.loads((output/'protocol.json').read_text())
    rows=[json.loads(line) for line in (output/'trials.jsonl').read_text().splitlines()]
    assert len(rows)==p['episodes']==160
    groups=defaultdict(dict)
    for row in rows:
        key=row['seed'],row['scenario'];assert row['method'] not in groups[key]
        groups[key][row['method']]=row
    assert len(groups)==40 and all(set(g)==set(p['methods']) for g in groups.values())
    assert all(sha(path)==digest for path,digest in p['frozen_inputs'].items())
    assert all(sha(ROOT/'frozen-source'/path)==digest for path,digest in p['runtime_source_sha256'].items())
    pixels_equal=True
    for group in groups.values():
        arrays=[]
        for row in group.values():
            with np.load(row['acquisition']['h0']['image']) as file:arrays.append(file['rgb'].copy())
        pixels_equal &= all(np.array_equal(arrays[0],a) for a in arrays[1:])
    validity={'episodes':len(rows),'paired_states':len(groups),
        'all_paired_prefixes_equal':all(same_prefix(list(g.values())) for g in groups.values()),
        'all_h0_pixels_identical':bool(pixels_equal),
        'scope_choices_equal':all(len({r['candidate'] for r in g.values()})==1 for g in groups.values()),
        'initialization_failures':sum(not r.get('nominal_ready',False) for r in rows),
        'invalid_faults':sum(not r.get('fault_class_valid',False) for r in rows),
        'execution_errors':sum(bool(r.get('error')) for r in rows),
        'numerical_instabilities':sum(r.get('failure_reason')=='numerical_instability' for r in rows),
        'contact_anchors':sum(r['contact_anchor_count'] for r in rows),
        'state_edits':sum(r['state_edit_count_after_initialization'] for r in rows),
        'forbidden_calls':sum(len(r['blocked_calls']) for r in rows),
        'external_wrench_zero':all(r['external_wrench_at_end']==0 for r in rows),
        'complete_costs':sum('total_post_fault_seconds' in r for r in rows),
        'cost_identity':all(abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-7 for r in rows),
        'all_waits_audited':all(sum(f['phase']=='recovery' and f['stage'] in ('inference_wait','controller_compute_wait','prefix_observation_wait') for f in r['trace'])==r['inference_wait_steps'] for r in rows),
        'no_post_prefix_budget_wait':all(not any(f['stage'] in ('duration_budget_wait','duration_deadline_wait') for f in r['trace']) for r in rows),
        'frozen_models_thresholds_and_sources_unchanged':True,
        'execution_order_first_counts':dict(Counter(next(iter(g)) for g in groups.values()))}
    metrics={}
    for method in p['methods']:
        group=[r for r in rows if r['method']==method]
        values=method_metrics(group)
        starts=[local_precondition(r) for r in group]
        starts=[s for s in starts if s is not None]
        values.update(model_admissions=sum(used(r) for r in group),
            admitted_safe=sum(used(r) and safe(r) for r in group),
            admitted_unsafe=sum(used(r) and not safe(r) for r in group),
            admitted_process_violations=sum(used(r) and r['protection_violation'] for r in group),
            baseline_fallbacks=sum(r.get('duration_selection',{}).get('baseline_fallback',False) for r in group),
            fallback_safe=sum(r.get('duration_selection',{}).get('baseline_fallback',False) and safe(r) for r in group),
            explicit_rejections=sum(bool(r.get('abstained')) for r in group),
            deadline_misses=sum(r.get('duration_selection',{}).get('deadline_missed',False) for r in group),
            duration_decisions=sum('duration_selection' in r for r in group),
            timely_forecasts=sum(r.get('duration_selection',{}).get('forecast_ready',False) for r in group),
            controller_overrun_steps=sum(r.get('controller_compute_wait_steps',0) for r in group),
            duration_counts=dict(Counter(str(r.get('local_steps',{}).get('local_relevel_steps')) for r in group)),
            local_starts=len(starts),missing_keeper_at_h0=sum(not s['keeper_present_at_start'] for s in starts),
            keeper_interrupted_after_valid_h0=sum(s['keeper_interrupted_after_valid_start'] for s in starts),
            first_keeper_loss_stages=dict(Counter(s['first_loss_stage'] for s in starts if s['keeper_interrupted_after_valid_start'])))
        metrics[method]=values
    pairs=[(m,'fixed60') for m in ('ce','terminal','joint')]+[('joint','ce'),('joint','terminal'),('ce','terminal')]
    common=[r for r in rows if all(safe(x) for x in groups[r['seed'],r['scenario']].values())]
    decompositions={}
    for method in ('ce','terminal','joint'):
        pairs_for_method=[(g['fixed60'],g[method]) for g in groups.values()]
        decompositions[method]={
            'mean_total_delta_seconds':float(np.mean([b['total_post_fault_seconds']-a['total_post_fault_seconds'] for a,b in pairs_for_method])),
            'mean_wait_delta_seconds':float(np.mean([b['inference_wait_seconds']-a['inference_wait_seconds'] for a,b in pairs_for_method])),
            'mean_skill_delta_seconds':float(np.mean([b['skill_seconds']-a['skill_seconds'] for a,b in pairs_for_method])),
            'same_initial_wait_budget_states':sum(a['inference_wait_steps']-a.get('controller_compute_wait_steps',0)==
                b['inference_wait_steps']-b.get('controller_compute_wait_steps',0) for a,b in pairs_for_method)}
    report={'scope':p['scope'],'validity':validity,'methods':metrics,
        'outcomes':{m:dict(Counter(joint_label(r) for r in rows if r['method']==m)) for m in p['methods']},
        'by_scenario':{q:{m:method_metrics([r for r in rows if r['method']==m and r['scenario']==q]) for m in p['methods']} for q in ('left_loss','right_loss')},
        'paired':{f'{a}_vs_{b}':paired_statistics(rows,a,b) for a,b in pairs},
        'paired_cost_decomposition_vs_fixed60':decompositions,
        'all_four_common_safe':{'states':len(common)//4,
            'methods':{m:method_metrics([r for r in common if r['method']==m]) for m in p['methods']},
            'paired':{f'{a}_vs_{b}':paired_statistics(common,a,b) for a,b in pairs} if common else {}},
        'failures':[{k:r.get(k) for k in ('seed','scenario','method','candidate','success','task_complete','protection_violation','failure_reason','protection','duration_selection')} for r in rows if not safe(r)]}
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['method','episodes','safe_successes','terminal_successes','protection_violations','mean_total_seconds',
        'mean_wait_seconds','mean_skill_seconds','mean_recovery_path_m','model_admissions','admitted_safe',
        'baseline_fallbacks','fallback_safe','explicit_rejections','deadline_misses']
    with (ROOT/'main-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({'method':m,**{k:v[k] for k in columns[1:]}} for m,v in metrics.items())
    columns=['seed','scenario','method','candidate','success','task_complete','protection_violation',
        'total_post_fault_seconds','inference_wait_seconds','skill_seconds','recovery_path_m','failure_reason']
    with (ROOT/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r.get(k) for k in columns} for r in rows)
    print(json.dumps({'validity':validity,'methods':metrics,'common_safe_states':len(common)//4}),flush=True)


if __name__=='__main__':main()
