"""Scientific summary of actual frozen-policy load-transfer executions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
NAMES={'protected':'Protected rule','gated_direct':'Frozen gated MLP','global':'Global restart'}
COLORS={'protected':'#207f78','gated_direct':'#b06b32','global':'#586d9e'}


def main():
    report=json.loads((ROOT/'report.json').read_text())
    methods=list(NAMES);loads=('native','density150');windows=('actual','1000ms')
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.6),sharex='col',sharey='row',constrained_layout=True)
    for col,load in enumerate(loads):
        for j,m in enumerate(methods):
            vals=[report['fault_results'][load][w][m] for w in windows]
            x=np.arange(2)+(j-1)*.24
            y=[v['protected_completions']/v['episodes'] for v in vals]
            bars=axes[0,col].bar(x,y,width=.22,color=COLORS[m],label=NAMES[m])
            for b,v in zip(bars,vals):
                axes[0,col].text(b.get_x()+b.get_width()/2,b.get_height()+.018,
                    str(v['protected_completions'])+'/60',ha='center',va='bottom',fontsize=7)
            t=[v['mean_total_seconds'] for v in vals];wait=[v['mean_wait_seconds'] for v in vals]
            axes[1,col].bar(x,t,width=.22,color=COLORS[m],alpha=.9)
            axes[1,col].bar(x,wait,width=.22,bottom=np.asarray(t)-wait,facecolor='none',
                edgecolor='#222222',hatch='////',linewidth=.4)
        axes[0,col].set_title('Native 1.212 kg' if load=='native' else 'Unseen load 1.818 kg',fontsize=10)
        axes[0,col].set_ylim(0,1.17);axes[0,col].set_yticks([0,.25,.5,.75,1.])
        axes[1,col].set_xticks([0,1],['Measured delay','1 s physical wait'],fontsize=8)
        for ax in axes[:,col]:
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
    axes[0,0].set_ylabel('Protected completion / all fault states',fontsize=8)
    axes[1,0].set_ylabel('All-attempt full physical time (s)',fontsize=8)
    axes[0,1].legend(loc='upper center',bbox_to_anchor=(.5,1.31),ncol=3,fontsize=7,frameon=False)
    for ext in ('pdf','png'):
        fig.savefig(ROOT/('load-transfer.'+ext),dpi=200,bbox_inches='tight')
        fig.savefig(REPO/'paper/figures'/('load-transfer.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    # Main-paper shifted-load table: all-fault cost and separate healthy controls.
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Wait & Policy & Fault $Y$ & Healthy $Y$ & Total (s) & Global \\',r'\midrule']
    labels={'protected':'Rule','gated_direct':'Gated MLP','global':'Global'}
    for i,w in enumerate(windows):
        for j,m in enumerate(methods):
            r=report['fault_results']['density150'][w][m]
            h=report['healthy_controls']['density150'][w][m]
            window=('Actual' if w=='actual' else '1 s') if j==0 else ''
            lines.append(f"{window} & {labels[m]} & {r['protected_completions']}/60 & {h['protected_completions']}/20 & {r['mean_total_seconds']:.3f} & {r['choices'].get('R_global',0)} "+r'\\')
        if i==0:lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (REPO/'paper/tables/load-transfer.tex').write_text('\n'.join(lines)+'\n')
    print('Generated actual load-transfer figure and shifted-load table.')


if __name__=='__main__':main()
