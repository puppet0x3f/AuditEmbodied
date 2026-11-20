"""Publication figures from frozen physical outcomes and common-prefix diagnostics."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'));sys.path.insert(0,str(HERE))
from paired_holding_audit import interval

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'registered':'#b76536','measured_position':'#3976a8','measured_rotation':'#8b6ca7','measured_pose':'#407d63'}


def save(fig,name):
    fig.tight_layout(pad=.7)
    for suffix in ('pdf','png'):fig.savefig(HERE/(name+'.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)


def prior():
    report=json.loads((HERE/'prior-mechanism.json').read_text())
    a=report['cohorts']['mechanism']['long_wait_diagnostics']
    holds=('registered','measured_position');labels=('Registered','Measured position')
    fig=plt.figure(figsize=(7.05,3.0));grid=fig.add_gridspec(2,3,height_ratios=(.55,1.5),hspace=.7)
    flow=fig.add_subplot(grid[0,:]);flow.axis('off')
    boxes=['Matched\ngrasp fault','Initial holding\nreference','Placement and\nfresh RGB','Approach, grasp,\nretry and verify']
    for i,label in enumerate(boxes):
        x=.11+i*.26
        flow.text(x,.52,label,ha='center',va='center',fontsize=8,
            bbox={'boxstyle':'round,pad=.45','fc':'#edf1f4','ec':'#8d9ba6','lw':.7})
        if i<3:flow.annotate('',xy=(x+.155,.52),xytext=(x+.105,.52),arrowprops={'arrowstyle':'->','lw':.9,'color':'#60717e'})
    axes=[fig.add_subplot(grid[1,j]) for j in range(3)]
    values=[a[h]['wait_translation_mm'] for h in holds]
    bars=axes[0].bar(range(2),values,color=[COLORS[h] for h in holds],width=.58)
    for b,v in zip(bars,values):axes[0].text(b.get_x()+b.get_width()/2,v+.6,f'{v:.1f}',ha='center',fontsize=7)
    axes[0].set(ylim=(0,25),title='(a) Wait-end translation\n(mm)')
    x=np.arange(2);width=.34
    for h,offset,label in zip(holds,(-width/2,width/2),labels):
        values=[a[h]['preclose_rgb_target_miss_mm'],a[h]['first_close_miss_mm']]
        bars=axes[1].bar(x+offset,values,width,color=COLORS[h],label=label)
        for b,v in zip(bars,values):axes[1].text(b.get_x()+b.get_width()/2,v+3,f'{v:.1f}',ha='center',fontsize=6.7)
    axes[1].set_xticks(x,['RGB target\nat observation','Actual first\nclose']);axes[1].set(ylim=(0,150),title='(b) Target / handle mismatch\n(mm)')
    values=[a[h]['first_close_dual']*100 for h in holds]
    bars=axes[2].bar(range(2),values,color=[COLORS[h] for h in holds],width=.58)
    for b,v in zip(bars,values):axes[2].text(b.get_x()+b.get_width()/2,v+3,f'{round(v*.4)}/40',ha='center',fontsize=7)
    axes[2].set(ylim=(0,100),title='(c) First dual grasp (%)')
    for ax in (axes[0],axes[2]):ax.set_xticks(range(2),['Registered','Position only'],fontsize=7)
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,frameon=False,fontsize=7,ncol=2,
        loc='upper center',bbox_to_anchor=(.51,.73))
    for ax in axes:ax.grid(alpha=.14,axis='y')
    save(fig,'holding-mechanism-chain')


def paired():
    path=HERE/'paired-analysis/report.json'
    if not path.exists():return
    report=json.loads(path.read_text())
    holds=('registered','measured_position','measured_rotation','measured_pose')
    labels=('Registered','Position only','Rotation only','Position + rotation')
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.5))
    for ax,method,title in zip(axes,('closed_loop','visual_servo'),('Frozen CAD feedback','Frozen landmark recovery')):
        for step,offset,alpha in [(1,-.17,.45),(20,.17,1.)]:
            group=report['conditions'][method];values=[group[h][str(step)]['faults']['native_Y']/40*100 for h in holds]
            ax.bar(np.arange(4)+offset,values,.34,color=[COLORS[h] for h in holds],alpha=alpha,label=f'{step*.05:.2f} s')
            for i,v in enumerate(values):ax.text(i+offset,v+1.5,str(round(v*.4)),ha='center',fontsize=6.5)
        ax.set_xticks(range(4),['Registered','Position\nonly','Rotation\nonly','Position +\nrotation'],fontsize=7)
        ax.set(ylim=(0,105),title=title);ax.grid(alpha=.15,axis='y')
    axes[0].set_ylabel('Protected completion (%)');axes[1].legend(frameon=False,fontsize=7,loc='upper right')
    save(fig,'paired-holding')


if __name__=='__main__':prior();paired()
