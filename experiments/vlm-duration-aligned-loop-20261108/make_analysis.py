"""Summarize the frozen, fresh-seed three-way timing experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent
report=json.loads((out/'report.json').read_text())
rows=[json.loads(s) for s in (out/'trials.jsonl').read_text().splitlines()]
names=['oracle_role_fixed60','vlm_with_wait','prefix_overlap']
labels=['Oracle role\nfixed 60','VLM +\nserial wait','VLM +\nshared prefix']
groups={name:[r for r in rows if r['method']==name] for name in names}
paired={}
for r in rows:paired.setdefault((r['seed'],r['scenario']),{})[r['method']]=r

comparisons={}
for reference in names[:2]:
    both=[p for p in paired.values() if all(p[m]['success'] and not p[m]['protection_violation'] for m in (reference,'prefix_overlap'))]
    comparisons[reference]={'both_safe_states':len(both),
        'reference_mean_seconds':float(np.mean([p[reference]['total_post_fault_seconds'] for p in both])) if both else None,
        'aligned_mean_seconds':float(np.mean([p['prefix_overlap']['total_post_fault_seconds'] for p in both])) if both else None,
        'safe_gains':sum(p['prefix_overlap']['success'] and not p['prefix_overlap']['protection_violation']
            and not (p[reference]['success'] and not p[reference]['protection_violation']) for p in paired.values()),
        'safe_losses':sum(p[reference]['success'] and not p[reference]['protection_violation']
            and not (p['prefix_overlap']['success'] and not p['prefix_overlap']['protection_violation']) for p in paired.values())}

summary={'valid_execution':report['valid_execution'],'physical_states':report['physical_states'],
    'physical_episodes':report['physical_episodes'],'methods':report['methods'],
    'both_safe_paired_comparisons':comparisons,
    'overlap_pipeline_seconds':[r['duration_selection']['pipeline_wall_seconds'] for r in groups['prefix_overlap'] if 'duration_selection' in r],
    'prefix_budget_seconds':[r['duration_selection']['prefix_seconds'] for r in groups['prefix_overlap'] if 'duration_selection' in r],
    'all_outcomes_retained':True,'no_weight_or_threshold_updates':report['frozen_artifacts_unchanged'],
    'latency_emulation':report['time_definition'],
    'interpretation_scope':'Small independent simulation pilot. Both-safe comparisons avoid treating fast abstention/failure as efficiency improvement. Scope gating and refreshed forecasts are a controller change; weights, native-token score formula and threshold are unchanged.'}
(out/'analysis-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.2),gridspec_kw={'width_ratios':[1,1.1,1.25]})
x=np.arange(3)
safe=np.array([sum(r['success'] and not r['protection_violation'] for r in groups[m]) for m in names])
unsafe=np.array([sum(r['success'] and r['protection_violation'] for r in groups[m]) for m in names])
failed=8-safe-unsafe
axes[0].bar(x,safe,color='#2a8c85',label='Safe completion')
axes[0].bar(x,unsafe,bottom=safe,color='#e8a441',label='Completed with violation')
axes[0].bar(x,failed,bottom=safe+unsafe,color='#b65555',label='Unresolved / failed')
for i,v in enumerate(safe):axes[0].text(i,v/2 if v else .12,f'{v}/8',ha='center',va='center',color='white' if v else 'black',fontweight='bold')
axes[0].set(xticks=x,xticklabels=labels,ylim=(0,8.8),ylabel='Fault states',title='Completion and process protection')
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.24),fontsize=8,frameon=False)

means=report['methods']
wait=np.array([means[m]['mean_executed_wait_seconds'] for m in names])
skill=np.array([means[m]['mean_skill_seconds'] for m in names])
hold=np.array([means[m]['mean_final_hold_seconds'] for m in names])
axes[1].bar(x,skill,color='#4c78a8',label='Executed skill (incl. overlap)')
axes[1].bar(x,wait,bottom=skill,color='#db805f',label='Executed inference holds')
axes[1].bar(x,hold,bottom=skill+wait,color='#bcbcbc',label='Final hold')
for i,total in enumerate(skill+wait+hold):axes[1].text(i,total+.12,f'{total:.2f}s',ha='center')
axes[1].set(xticks=x,xticklabels=labels,ylabel='Mean post-fault seconds',title='All states; all waits charged')
axes[1].set_ylim(0,max(skill+wait+hold)*1.16)
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.24),fontsize=8,frameon=False)

for i,(key,pair) in enumerate(sorted(paired.items())):
    a=pair['vlm_with_wait'];b=pair['prefix_overlap']
    axes[2].plot([a['total_post_fault_seconds'],b['total_post_fault_seconds']],[i,i],color='#c4c4c4',zorder=1)
    for j,r in enumerate((a,b)):
        ok=r['success'] and not r['protection_violation']
        axes[2].scatter(r['total_post_fault_seconds'],i,marker='o' if ok else 'x',color='#2a8c85' if ok else '#b65555',s=40,zorder=2)
keys=sorted(paired)
axes[2].set(yticks=np.arange(len(keys)),yticklabels=[str(s)+('L' if q=='left_loss' else 'R') for s,q in keys],
    xlim=(7.5,8.15),xlabel='Post-fault seconds (shared prefix at left)',title='Paired states (cross = unsafe / failed)')
axes[2].invert_yaxis()
fig.suptitle('Frozen Qwen3-VL + fixed threshold | fresh seeds 2000-2003 | 24 rollouts',fontsize=12)
fig.subplots_adjust(bottom=.29,top=.83,wspace=.35)
for extension in ('png','pdf'):fig.savefig(out/f'alignment-results.{extension}',dpi=180,bbox_inches='tight')
print(json.dumps({'valid_execution':report['valid_execution'],'both_safe_paired_comparisons':comparisons},indent=2))
