"""Handover safety--time curves, controlled waiting, and audited retry paths."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent.parent/'paper'
METHODS=['visual','proprio','rule','R3']
NAMES={'visual':'Visual','proprio':'Proprio','rule':'Finger rule','R3':'Global R3'}
COLORS={'visual':'#27816f','proprio':'#3879b5','rule':'#c77b2a','R3':'#858992'}
WINDOWS=['actual','1000ms','2000ms']


def save(fig,name):
    for ext in ['pdf','png']:
        fig.savefig(ROOT/(name+'.'+ext),dpi=200,bbox_inches='tight')
        fig.savefig(PAPER/'figures'/(name+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)


def main():
    report=json.loads((ROOT/'report.json').read_text());rows=json.loads((ROOT/'episodes-without-traces.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.6))
    maximum=max(r['audit']['total_seconds'] or 0 for r in rows if r['window']=='actual' and r['scenario']!='nominal')
    for method in METHODS:
        subset=[r for r in rows if r['method']==method and r['window']=='actual' and r['scenario']!='nominal']
        times=sorted(r['audit']['total_seconds'] for r in subset if r['audit']['protected_completion'])
        x=[0.]+times+[maximum+1];y=[0.]+[(i+1)/len(subset) for i in range(len(times))]+[len(times)/len(subset)]
        axes[0].step(x,y,where='post',label=NAMES[method],color=COLORS[method])
        total=np.array([report['methods'][w]['fault'][method]['mean_total_seconds'] for w in WINDOWS])
        waiting=np.array([report['methods'][w]['fault'][method]['mean_wait_seconds'] for w in WINDOWS])
        positions=np.arange(3)+(METHODS.index(method)-1.5)*.19
        axes[1].bar(positions,total-waiting,width=.18,color=COLORS[method],alpha=.75)
        axes[1].bar(positions,waiting,bottom=total-waiting,width=.18,color=COLORS[method],hatch='///',edgecolor='black',linewidth=.3)
    axes[0].set(xlabel='Post-fault time budget (s)',ylabel='Protected completion fraction',ylim=(-.02,1.04),xlim=(0,maximum+1))
    axes[1].set(xticks=range(3),xticklabels=['Measured','Min. 1s','Min. 2s'],xlabel='Physical decision window',ylabel='Mean full post-fault time (s)',ylim=(0,29))
    axes[0].legend(fontsize=7,loc='upper left',ncol=2)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor='white',hatch='///',edgecolor='black',label='Blocking wait')],fontsize=7,loc='upper left')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'handover-audit')
    lines=[r'\begin{tabular}{llrrrrrr}',r'\toprule',r'Window & Policy & Fault $C$ & Fault $Y$ & Nominal $Y$ & Wait (s) & Total (s) & R1$\to$R3 \\',r'\midrule']
    for window in WINDOWS:
        for method in METHODS:
            fault=report['methods'][window]['fault'][method];nominal=report['methods'][window]['nominal'][method]
            label={'actual':'Measured','1000ms':'1 s min.','2000ms':'2 s min.'}[window]
            lines.append(f"{label} & {NAMES[method]} & {fault['terminal_completions']}/40 & {fault['protected_completions']}/40 & {nominal['protected_completions']}/20 & {fault['mean_wait_seconds']:.3f} & {fault['mean_total_seconds']:.3f} & {fault['local_then_global_episodes']} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/handover-audit.tex').write_text('\n'.join(lines)+'\n')
    # One executed retry path, preferring a completed path with a protection loss.
    candidates=[r for r in rows if r['scenario']!='nominal' and r['window']=='actual' and r['audit'].get('local_then_global')]
    candidates.sort(key=lambda r:(not(r['audit'].get('terminal_completion') and not r['audit']['protected_completion']),r['seed'],r['method']))
    if candidates:
        chosen=candidates[0];key=tuple(chosen[k] for k in ('seed','scenario','method','window'))
        with (ROOT/'trials.jsonl').open() as stream:
            row=next(json.loads(s) for s in stream if (lambda r:tuple(r[k] for k in ('seed','scenario','method','window'))==key)(json.loads(s)))
        frames=[f for f in row['trace'] if f['phase'] in ('recovery','release','hold')]
        fig,axes=plt.subplots(2,1,figsize=(7.1,3.0),sharex=True,gridspec_kw={'height_ratios':[1,1.1]})
        start=row['prefix_state']['time'];x=[f['time']-start for f in frames]
        axes[0].step(x,[int(f['evaluator']['giver_holds']) for f in frames],where='post',label='Giver grasp',lw=1.4)
        axes[0].step(x,[int(f['evaluator']['receiver_supports']) for f in frames],where='post',label='Receiver support',lw=1.4)
        axes[0].set(ylim=(-.1,1.2),yticks=[0,1],yticklabels=['Absent','Present'])
        axes[0].legend(fontsize=7,ncol=2,loc='upper right')
        palette={'wait':'#d7ab3c','local':'#357ab0','global':'#288575','release':'#97658e','hold':'#a7adb3'}
        def kind(f):
            return 'wait' if f['stage'] in ('decision_wait','controller_compute_wait') else 'release' if f['phase']=='release' else 'hold' if f['phase']=='hold' else 'global' if f['stage'].startswith('r3') else 'local'
        begin=0;current=kind(frames[0])
        for i in range(1,len(frames)+1):
            if i==len(frames) or kind(frames[i])!=current:
                left=frames[begin]['time']-start-.05;right=frames[i-1]['time']-start
                axes[1].barh(0,right-left,left=left,color=palette[current],height=.4)
                if i<len(frames):begin=i;current=kind(frames[i])
        axes[1].set(yticks=[],xlabel='Seconds after the shared fault prefix')
        for k,color in palette.items():axes[1].plot([],[],color=color,lw=5,label=k)
        axes[1].legend(ncol=5,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.4))
        axes[0].set_title(f"Executed retry: seed {row['seed']}, {row['scenario']}, {NAMES[row['method']]} / {row['audit']['label']}",fontsize=10)
        for ax in axes:ax.grid(axis='x',alpha=.2)
        fig.tight_layout();save(fig,'handover-retry-path')
        (ROOT/'illustrated-case.json').write_text(json.dumps({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],'window':row['window'],'audit':row['audit']},indent=2)+'\n')
    print('Handover table and physical safety--time / retry figures written.')


if __name__=='__main__':main()
