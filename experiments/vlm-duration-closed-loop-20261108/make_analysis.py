"""Report the frozen online pilot and its separate delay-removal diagnostic."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]))
from auditembodied.duration_pilot import file_hash

report=json.loads((root/'report.json').read_text())
diagnostic=json.loads((root/'no-wait-diagnostic.json').read_text())
protocol=json.loads((root/'protocol.json').read_text())
rows=[json.loads(s) for s in (root/'trials.jsonl').read_text().splitlines()]
assert report['valid_execution'] and report['frozen_artifacts_unchanged']
assert diagnostic['all_prefixes_identical'] and diagnostic['all_no_wait_pre_relevel_match_fixed60']
policy=[r for r in rows if r['method']=='vlm_with_wait']
baseline=[r for r in rows if r['method']=='oracle_role_fixed60']
policy_summary=report['methods']['vlm_with_wait'];base_summary=report['methods']['oracle_role_fixed60']
diag_summary=diagnostic['summary']
safe_pairs=[c for c in report['cases'] if c['policy_success'] and not c['policy_process_violation']
            and c['baseline_success'] and not c['baseline_process_violation']]
mean=lambda xs:sum(xs)/len(xs) if xs else None
cases=[]
for c,d in zip(report['cases'],diagnostic['cases']):
    assert (c['seed'],c['scenario'])==(d['seed'],d['scenario'])
    cases.append({**c,**{k:d[k] for k in ['without_wait_success','without_wait_violation','without_wait_total_seconds',
        'with_wait_searches','without_wait_searches','extra_skill_seconds']}})
with (root/'cases.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(cases[0]));w.writeheader();w.writerows(cases)

duration_saving=base_summary['mean_total_post_fault_seconds']-diag_summary['mean_total_post_fault_seconds']
direct_wait=policy_summary['mean_executed_wait_seconds']
extra_skill=policy_summary['mean_skill_seconds']-diag_summary['mean_skill_seconds']
net_extra=policy_summary['mean_total_post_fault_seconds']-base_summary['mean_total_post_fault_seconds']
assert abs(net_extra-(-duration_saving+direct_wait+extra_skill))<1e-8
waiting_loss=[]
for r in policy:
    initial=[i for i,v in enumerate(r['post_fault']['holds']) if v]
    waiting=[f for f in r['trace'] if f['stage']=='inference_wait']
    waiting_loss.append(sum(any(not f['evaluator']['holds'][i] for i in initial) for f in waiting))
summary={
    'primary_result':'FROZEN_INDEPENDENT_PILOT_DID_NOT_DEMONSTRATE_SAFETY_OR_TIME_ADVANTAGE',
    'independent_initialization_seeds':report['seeds'],'independent_fault_states':8,'primary_physical_episodes':16,
    'additional_posthoc_diagnostic_episodes':8,'frozen_model_and_threshold_unchanged':report['frozen_artifacts_unchanged'],
    'physics_source_unchanged_during_test':all(file_hash(p)==h for p,h in protocol['physics_source_sha256'].items()),
    'all_paired_prefixes_match':not report['prefix_mismatches'],'all_wait_accounting_checks_pass':not report['wait_accounting_errors'],
    'primary_policy':policy_summary,'oracle_role_fixed60_control':base_summary,
    'posthoc_selected_actions_without_wait':diag_summary,
    'model_setup_seconds':report['model_setup']['setup_seconds'],
    'campaign_wall_seconds':report['campaign_wall_seconds'],
    'time_difference_seconds':net_extra,
    'time_increase_fraction':net_extra/base_summary['mean_total_post_fault_seconds'],
    'common_safe_pairs':len(safe_pairs),
    'common_safe_mean_policy_total_seconds':mean([c['policy_total_seconds'] for c in safe_pairs]),
    'common_safe_mean_baseline_total_seconds':mean([c['baseline_total_seconds'] for c in safe_pairs]),
    'delay_decomposition_seconds':{'nominal_duration_saving_vs_fixed60':duration_saving,
        'direct_measured_delay_cost':direct_wait,'extra_feedback_skill_time_after_wait':extra_skill,
        'net_total_extra_time_vs_fixed60':net_extra},
    'waiting_initial_keeper_loss_steps':waiting_loss,
    'mean_search_attempts_without_wait':mean([c['without_wait_searches'] for c in cases]),
    'mean_search_attempts_with_wait':mean([c['with_wait_searches'] for c in cases]),
    'additional_violations_from_wait':sum(c['with_wait_violation'] and not c['without_wait_violation'] for c in diagnostic['cases']),
    'additional_terminal_failures_from_wait':sum(not c['with_wait_success'] and c['without_wait_success'] for c in diagnostic['cases']),
    'observed_repair_role_selection_correct':sum(r['candidate']==('R_left' if r['scenario']=='left_loss' else 'R_right') for r in policy),
    'interpretation':'Waiting itself did not lose the initial keeper grasp in these eight trials, but changed the state before recovery, adding search cycles and two later protection violations plus one terminal failure. One unsafe state persists without waiting. No-wait replay is an attribution diagnostic, not online performance.',
    'scope':'Four new initialization seeds, eight fault states; a small fixed-policy simulation pilot. Baseline has oracle repair role and zero model delay. No training or threshold retuning.'}
(root/'analysis-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
names=['Fixed60\n(oracle role)','Selected actions\n(no-wait diagnostic)','Frozen VLM\n(with measured wait)']
ms=[base_summary,diag_summary,policy_summary]
x=np.arange(3)
for offset,key,color,label in [(-.18,'successes','#8c949e','Terminal success'),(.18,'safe_successes','#009e73','Safe completion')]:
    bars=axes[0].bar(x+offset,[m[key] for m in ms],.34,color=color,label=label)
    axes[0].bar_label(bars,labels=[f"{m[key]}/8" for m in ms],padding=3,fontsize=8)
axes[0].set(xticks=x,xticklabels=names,ylim=(0,9.5),ylabel='Trials',title='Frozen independent pilot')
bottom=np.zeros(3)
for key,color,label in [('mean_skill_seconds','#0072b2','Recovery skill'),('mean_executed_wait_seconds','#e69f00','Audited inference wait'),('mean_final_hold_seconds','#b8b8b8','Final hold')]:
    values=[m[key] for m in ms]
    axes[1].bar(x,values,bottom=bottom,color=color,label=label)
    bottom+=values
for i,value in enumerate(bottom):axes[1].text(i,value+.12,f'{value:.2f}s',ha='center',fontsize=9)
axes[1].set(xticks=x,xticklabels=names,ylim=(0,10.5),ylabel='Mean post-fault seconds',title='All waiting contributes to robot time')
labels=[f"{c['seed']}\n{'L' if c['scenario']=='left_loss' else 'R'}" for c in cases]
axes[2].plot(range(8),[c['without_wait_searches'] for c in cases],'o-',label='Same actions, no wait',color='#8c949e')
axes[2].plot(range(8),[c['with_wait_searches'] for c in cases],'o-',label='With measured wait',color='#d55e00')
axes[2].set(xticks=range(8),xticklabels=labels,yticks=[1,2,3],ylim=(.7,3.6),ylabel='Approach/close attempts',title='Wait changes later feedback execution')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8)
fig.savefig(root/'closed-loop-results.png',dpi=200)
fig.savefig(root/'closed-loop-results.pdf')
