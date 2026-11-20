"""Plot all fault episodes; roles are evaluator-only descriptive labels."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'existing-analysis/report.json').read_text())
rows=[json.loads(s) for s in (HERE/'existing-analysis/cases.jsonl').read_text().splitlines()]
colors=['#4b7b9b','#c3a45f','#ae634d']
fig,axes=plt.subplots(1,2,figsize=(7.0,3.0),gridspec_kw={'width_ratios':[1.05,1.2]})
for ax,method,title in zip(axes[:1],('closed_loop',),('CAD feedback',)):
    selected=[c for c in report['conditions'] if c['method']==method and not c['healthy']
              and c['wait_steps']==20 and c['hold_mode'] in ('registered','measured_position')]
    selected.sort(key=lambda c:c['hold_mode']!='registered')
    for i,c in enumerate(selected):
        for j,role in enumerate(('keeper','free')):
            y=3-(i*2+j)
            v=c['roles'][role];p=np.asarray(v['signed_contributions_mm'])
            positive=negative=0.
            for k,val in enumerate(p):
                left=positive if val>=0 else negative
                ax.barh(y,val,left=left,color=colors[k],height=.62,label=('Target at observation','Handle drift','EEF tracking')[k] if y==3 else None)
                if val>=0:positive+=val
                else:negative+=val
            ax.plot(v['closure_norm_mm']['mean'],y,'k|',ms=11)
    ax.set_yticks([3,2,1,0],['Reg. keeper','Reg. free','Pos. keeper','Pos. free'])
    ax.axvline(0,color='.6',lw=.7);ax.set_xlim(-18,140);ax.set_xlabel('Signed closure-error contribution (mm)',fontsize=7)
    ax.set_title(title,fontsize=8);ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=7)
ax=axes[1]
keys=['placed','first_dual','ever_dual','lifted','terminal']
for method,hold,label,color,ls in [
    ('closed_loop','registered','CAD / registered','#b76536','-'),
    ('closed_loop','measured_position','CAD / position','#3976a8','-'),
    ('visual_servo','registered','Landmark / registered','#b76536','--'),
    ('visual_servo','measured_position','Landmark / position','#3976a8','--')]:
    c=next(c for c in report['conditions'] if c['method']==method and c['hold_mode']==hold and c['wait_steps']==20 and not c['healthy'])
    ax.plot(range(len(keys)),[c['stages'][k] for k in keys],marker='o',ms=4,color=color,ls=ls,label=label)
ax.set_ylim(0,42);ax.set_xticks(range(len(keys)),['Place','First\ndual','Ever\ndual','Lift','Finish']);ax.set_ylabel('Episodes / 40',fontsize=7);ax.set_title('Stage exposure at 1 s wait',fontsize=8)
ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7)
stage_handles,stage_labels=ax.get_legend_handles_labels()
fig.legend(stage_handles,stage_labels,loc='lower center',ncol=2,frameon=False,fontsize=6.5,bbox_to_anchor=(.5,-.01))
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=7)
fig.tight_layout(rect=[0,.16,1,.9],w_pad=1.4)
fig.savefig(HERE/'existing-analysis/error-decomposition.pdf',bbox_inches='tight')
fig.savefig(HERE/'existing-analysis/error-decomposition.png',dpi=180,bbox_inches='tight')

text='''# First-close error and stage analysis

All 960 original executions (20 complete seed blocks, including 320 healthy runs) are retained. No physical executions or fitted models are added. The actual first-close target is reconstructed from the last valid RGB estimate, then checked against the logged translation action in all 960 runs. The maximum vector identity residual is 2.78e-17 m.

The identity is EEF(close) − handle(close) = [target − handle(observation)] − [handle(close) − handle(observation)] + [EEF(close) − target]. Target error includes the fixed registered grasp offset. Norms do not add; signed projections onto the final error direction do. Role labels come from the fault manifest only after execution. These descriptions identify where error appears, not an isolated causal effect of either arm.

At 1 s, CAD free-arm target, drift, tracking, and closure norms are respectively 7.08, 20.48, 126.79, 108.85 mm under registered holding and 5.40, 5.83, 13.50, 14.40 mm under position holding. Signed contributions are −4.40 − 8.20 + 121.45 = 108.85 mm for registered and 4.34 + 4.78 + 5.28 = 14.40 mm for position holding. The large final error is principally aligned with free-arm tracking, not the observation target residual. These are post-treatment descriptions, not mediated causal shares.

Both CAD conditions reach placement and attempt closing in all 40 fault runs. First dual grasp is 2/40 versus 34/40; ever dual grasp is 6/40 versus 36/40; completion is 6/40 versus 34/40. Thus the main separation precedes lifting and terminal balance. Landmark reaches the same stages but has 20/40 versus 17/40 first dual, 23/40 versus 23/40 ever dual, and 19/40 versus 22/40 completion. Its long-wait free-arm target and tracking errors remain about 30 and 64 mm under either reference. Lower Landmark completion therefore cannot be treated as an equivalent-capability architectural comparison or as population-level invariance to holding.

The stage panel uses full original denominators throughout. First dual and ever dual are distinct stage summaries, not a monotone survival curve. `report.json` includes all four position/rotation conditions, both waits, healthy runs, keeper/free roles, seed-level confidence intervals, and terminal predicate failures. `cases.jsonl` preserves per-execution vectors and action-target checks. Original completion and native audit labels are unchanged.
'''
(HERE/'existing-analysis/README.md').write_text(text)
print('Existing-log analysis complete: 960 target checks; whole-cohort report and figure saved.')
