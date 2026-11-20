"""Publication figures using actual replay pixels and original trace measurements."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FRAMES=OUT/'keyframes'
PAPER=ROOT/'paper/figures'
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})


def show(ax,case,index):
    rgb=plt.imread(FRAMES/f'{case}-{index}-agentview.png')
    crop=rgb[235:512,70:440] if case=='handover' else rgb[170:512,15:512]
    ax.imshow(crop);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_color('#aab2b8');spine.set_linewidth(.5)
    meta=json.loads((FRAMES/f'{case}.json').read_text())['frames'][index]
    ax.set_title(meta['caption'],fontsize=7.5,pad=3)
    ax.set_xlabel(f"t = {meta['post_fault_seconds']:.2f} s",fontsize=7,labelpad=2)


def save(fig,name):
    for suffix in ('pdf','png'):
        fig.savefig(OUT/f'{name}.{suffix}',dpi=200,bbox_inches='tight')
        (PAPER/f'{name}.{suffix}').write_bytes((OUT/f'{name}.{suffix}').read_bytes())
    plt.close(fig)


def main():
    fig,axes=plt.subplots(3,4,figsize=(7.1,4.95))
    for i,case in enumerate(('lift_graph','handover','lift_heavy')):
        for j in range(4):show(axes[i,j],case,j)
    fig.subplots_adjust(wspace=.07,hspace=.47,left=.01,right=.99,top=.96,bottom=.06)
    save(fig,'execution-keyframes')
    fig,axes=plt.subplots(1,2,figsize=(3.4,1.55))
    for ax,index in zip(axes,(1,2)):show(ax,'lift_heavy',index)
    fig.tight_layout(pad=.3);save(fig,'handoff-keyframes')
    rows={}
    with (ROOT/'outputs/agentchord-style-20261109/trials.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if (r['seed'],r['scenario'])==(12006,'right_loss'):rows[r['method']]=r
    row=rows['agentchord_style']
    local=[f for f in row['trace'] if f['phase']=='recovery' and f.get('skill')=='R_right']
    t=np.array([f['time']-row['prefix_state']['time'] for f in local])
    native=np.array([f['evaluator']['holds'][0] for f in local],float)
    widths=np.array([np.abs(f['robot']['fingers'][0]).mean() for f in local])
    proxy=((widths>.004)&(widths<.028)).astype(float)
    fig=plt.figure(figsize=(7.1,4.0));grid=fig.add_gridspec(3,4,height_ratios=(1.15,.85,1.15),hspace=.62,wspace=.08)
    for j in range(4):show(fig.add_subplot(grid[0,j]),'lift_graph',j)
    ax=fig.add_subplot(grid[1,:]);ax.step(t,proxy+.03,where='post',label='Encoder load proxy',color='#258672',lw=1.5)
    ax.step(t,native,where='post',label='Native keeper grasp (audit)',color='#b83c40',lw=1.5)
    ax.axvline(3.10,color='#b83c40',ls=':',lw=.8)
    ax.set(xlim=(0,t[-1]+.05),ylim=(-.1,1.7),yticks=(0,1),yticklabels=('Absent','Present'),xlabel='Seconds after fault, local attempt')
    ax.legend(loc='upper left',ncol=2,fontsize=7,frameon=False)
    ax=fig.add_subplot(grid[2,:]);colors={'wait':'#d9aa4d','local':'#3973ac','global':'#258672','hold':'#a2a6aa'}
    methods=('protected','gated_direct','agentchord_style','global');names=('Protected rule','Frozen MLP + gate','AgentChord-style','Fixed global')
    for i,method in enumerate(methods):
        r=rows[method]
        if method=='agentchord_style':
            events=[e for e in r['skill_events'] if e['candidate']!='continue']
            pieces=[('wait',r['inference_wait_seconds'])]+[('global' if e['candidate']=='R_global' else 'local',e['end_time']-e['start_time']) for e in events]
        else:pieces=[('wait',r['inference_wait_seconds']),('global' if r['candidate']=='R_global' else 'local',r['skill_seconds'])]
        pieces.append(('hold',r['hold_seconds']));start=0.
        for kind,seconds in pieces:ax.barh(i,seconds,left=start,color=colors[kind],height=.54);start+=seconds
        label='protected' if r['success'] and not r['protection_violation'] else 'complete, contract B' if r['task_complete'] else 'failed'
        ax.text(start+.12,i,f'{start:.2f}s; {label}',va='center',fontsize=7)
    ax.set(yticks=range(4),yticklabels=names,xlim=(0,23.5),xlabel='Complete post-fault seconds, including all attempts')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.legend(handles=[Patch(color=v,label=k.title()) for k,v in colors.items()],loc='lower center',ncol=4,fontsize=7,bbox_to_anchor=(.6,-.025),frameon=False)
    fig.subplots_adjust(left=.16,right=.99,top=.96,bottom=.18)
    save(fig,'recovery-evidence')


if __name__=='__main__':main()
