"""Publication figures from all completed, independently audited executions."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'))
from wait_geometry_audit import interval

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'retarget':'#3976a8','closed_loop':'#c66b32'}
HOLDS=('registered','measured_pose')
NAMES={'registered':'Registered-target holding','measured_pose':'Measured-pose holding'}


def load(phase):
    with (HERE/(phase+'-analysis/scored.jsonl')).open() as f:return [json.loads(line) for line in f]


def curve(ax,rows,field,method,hold,label,color=None,scale=1.,style='-'):
    steps=sorted({r['wait_steps'] for r in rows})
    means=[];lows=[];highs=[]
    for step in steps:
        group=[r for r in rows if r['scenario']!='sham' and r['method']==method and
               r['hold_mode']==hold and r['wait_steps']==step]
        seeds=sorted({r['seed'] for r in group})
        values=[np.mean([r[field] for r in group if r['seed']==seed]) for seed in seeds]
        estimate=interval(values);mean=estimate['estimate']*scale
        lo,hi=np.asarray(estimate['ci95'])*scale
        means.append(mean);lows.append(mean-lo);highs.append(hi-mean)
    ax.errorbar(np.array(steps)*.05,means,yerr=[lows,highs],color=color or COLORS[method],
        marker='o',markersize=3.5,capsize=2,lw=1.3,label=label,ls=style)
    ax.set_xticks(np.array(steps)*.05);ax.set_xlabel('Total initial holding (s)')
    ax.grid(alpha=.18,axis='y')


def main():
    rows=load('wait')
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.18),sharey=True)
    for ax,hold in zip(axes,HOLDS):
        for method,label in [('retarget','Position target'),('closed_loop','RGB feedback')]:
            curve(ax,rows,'native_Y',method,hold,label,scale=100.)
        ax.set_title(NAMES[hold]);ax.set_ylim(-3,103)
    axes[0].set_ylabel('Native-guard protected completion (%)')
    axes[1].legend(frameon=False,loc='upper right',fontsize=7)
    fig.tight_layout(pad=.6,w_pad=1.1)
    for suffix in ('pdf','png'):fig.savefig(HERE/f'wait-hold.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(10.1,5.1))
    panels=[('wait_end_tilt_deg','Wait-end tilt (deg)',1.),
            ('setdown_tilt_deg','First settled tilt (deg)',1.),
            ('first_close_miss_mm','First close maximum arm error (mm)',1.),
            ('first_close_dual','First close dual grasp (%)',100.),
            ('pickup_attempts','Pickup attempts',1.),('total','Complete physical time (s)',1.)]
    for ax,(field,title,scale) in zip(axes.flat,panels):
        for hold,color in [('registered','#c66b32'),('measured_pose','#5d7c54')]:
            curve(ax,rows,field,'closed_loop',hold,NAMES[hold],color,scale)
        ax.set_title(title)
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.tight_layout(pad=.8,w_pad=1.0,h_pad=1.2)
    for suffix in ('pdf','png'):fig.savefig(HERE/f'wait-mechanism.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)

    if not (HERE/'geometry-analysis/scored.jsonl').exists():return
    rows=load('geometry')
    fig,axes=plt.subplots(1,3,figsize=(9.4,2.55))
    for ax,shape,phase in zip(axes[:2],('pot','tray'),('development-wait','development-geometry')):
        path=HERE/phase/f'70-sham-{shape}-01-registered-retarget-agentview.png'
        if path.exists():ax.imshow(plt.imread(path))
        ax.set_title('Original pot: 14 x 14 x 14 cm' if shape=='pot' else 'Shallow container: 18 x 11 x 8 cm')
        ax.axis('off')
    for method,label in [('retarget','Position target'),('closed_loop','RGB feedback')]:
        curve(axes[2],rows,'native_Y',method,'registered',label,scale=100.)
    axes[2].set_ylabel('Native protection (%)');axes[2].set_ylim(-3,103)
    axes[2].set_title('Mass matched: 1.818 kg');axes[2].legend(frameon=False,fontsize=7)
    fig.tight_layout(pad=.6,w_pad=.8)
    for suffix in ('pdf','png'):fig.savefig(HERE/f'geometry-transfer.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
