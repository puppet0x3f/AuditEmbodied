"""Scientific summary of the completed supervision campaign."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parent.parent/'paper'


def main():
    report=json.loads((ROOT/'report.json').read_text())
    methods=['fixed60','ce','terminal','joint']
    names=['Fixed60','Plain SFT','Terminal only','Joint + set']
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65))
    colors={'A':'#258672','B':'#c55559','C':'#d5ad55','D':'#835e8a','E':'#7f858a'}
    left=np.zeros(4)
    for label,color in colors.items():
        values=np.array([report['outcomes'][m].get(label,0) for m in methods])
        axes[0].barh(names,values,left=left,color=color,label=label)
        for i,(v,l) in enumerate(zip(values,left)):
            if v:axes[0].text(l+v/2,i,str(v),ha='center',va='center',fontsize=8,color='white')
        left+=values
    axes[0].set(xlim=(0,40),xlabel='Executed outcomes / 40 fault states')
    axes[0].legend(ncol=5,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.22),handlelength=1)
    common=report['all_four_common_safe']
    if common['states']:
        times=[common['methods'][m]['mean_total_seconds'] for m in methods]
        axes[1].barh(names,times,color=['#899097','#d07825','#3973ac','#258672'])
        for i,value in enumerate(times):axes[1].text(value+.08,i,f'{value:.3f}',va='center',fontsize=8)
        axes[1].set_xlim(0,max(times)*1.2)
    else:axes[1].text(.5,.5,'No state safe for all four methods',ha='center',va='center',transform=axes[1].transAxes)
    axes[1].set_xlabel(f"Total seconds on same {common['states']} common-safe states")
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    for suffix in ('pdf','png'):
        fig.savefig(ROOT/f'comparison.{suffix}',dpi=200,bbox_inches='tight')
        fig.savefig(PAPER/'figures'/f'supervision.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
        r'Policy & Safe & Complete & Violations & Wait (s) & Total (s) & Admitted & Fallback \\',r'\midrule']
    for method,name in zip(methods,names):
        r=report['methods'][method]
        lines.append(f"{name} & {r['safe_successes']}/40 & {r['terminal_successes']}/40 & {r['protection_violations']} & {r['mean_wait_seconds']:.3f} & {r['mean_total_seconds']:.3f} & {r['model_admissions']} & {r['baseline_fallbacks']} "+r'\\')
    lines+=[r'\bottomrule',r'\end{tabular}']
    (PAPER/'tables/supervision.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'figure_written':True,'common_safe_states':common['states']}))


if __name__=='__main__':main()
