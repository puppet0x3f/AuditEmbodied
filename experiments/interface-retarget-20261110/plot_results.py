"""Publication figures from the complete recorded intervention and native images."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
PAPER=ROOT.parents[1]/'paper/figures'
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42})
METHODS=('fixed','refresh','retarget')
NAMES=('Fixed','Refresh only','Retarget')
COLORS=('#92999f','#4d80ad','#258672')


def save(fig,name):
    for ext in ('png','pdf'):
        file=ROOT/f'{name}.{ext}';fig.savefig(file,dpi=200,bbox_inches='tight')
        (PAPER/file.name).write_bytes(file.read_bytes())
    plt.close(fig)


def main():
    report=json.loads((ROOT/'report.json').read_text())
    scored=[json.loads(l) for l in (ROOT/'scored.jsonl').open()]
    poses=[json.loads(l) for l in (ROOT/'pose-cases.jsonl').open()]
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.1))
    ax=axes[0,0];groups=[(l,w) for l in ('native','density150') for w in ('actual','1000ms')]
    for i,(m,n,c) in enumerate(zip(METHODS,NAMES,COLORS)):
        values=[report['fault_results'][l][w][m]['Y']/40 for l,w in groups]
        x=np.arange(4)+(i-1)*.23;ax.bar(x,values,.23,color=c,label=n)
        for xx,y in zip(x,values):ax.text(xx,y+.025,f'{round(y*40)}',ha='center',va='bottom',fontsize=6)
    ax.set(xticks=range(4),xticklabels=('Native\nactual','Native\n+0.95 s','Heavy\nactual','Heavy\n+0.95 s'),ylim=(0,1.18),ylabel='Protected completion / 40 faults')
    ax.legend(ncol=3,fontsize=6,loc='upper left',frameon=False)
    ax.set_title('(a) Complete-path outcomes',fontsize=9)
    ax=axes[0,1]
    entries=[];labels=[]
    for load,title in [('native','Native'),('density150','Heavy')]:
        p=[r for r in poses if r['load']==load and r['scenario']!='sham']
        before=[r['fixed_miss_cm'] for r in p]
        entry=[max(r['entry_handle_error_mm'])/10 for r in p]
        closed=[100*r['close_miss_m'] for r in scored if r['load']==load and r['scenario']!='sham' and r['method']=='retarget']
        entries.extend([before,entry,closed]);labels.extend([f'{title}\nfixed',f'{title}\nRGB entry',f'{title}\nclose'])
    bp=ax.boxplot(entries,tick_labels=labels,patch_artist=True,showfliers=False,widths=.6)
    for b,c in zip(bp['boxes'],('#92999f','#4d80ad','#258672')*2):b.set_facecolor(c);b.set_alpha(.8)
    ax.set(ylabel='Worst-arm geometry error (cm)');ax.tick_params(axis='x',labelsize=6)
    ax.set_title('(b) Target localization and realized pickup',fontsize=9)
    ax=axes[1,0];x=np.arange(6);bottom=np.zeros(6)
    for count,color in zip((0,1,2),('#bd6f6b','#d5ad55','#258672')):
        values=[report['fault_results']['density150'][w][m]['close_grasp_histogram'].get(str(count),0)/40 for w in ('actual','1000ms') for m in METHODS]
        ax.bar(x,values,bottom=bottom,color=color,label=f'{count}/2 grasps');bottom+=values
    ax.set(xticks=x,xticklabels=['Fixed','Refresh','Retarget']*2,ylim=(0,1.15),ylabel='Heavy faults: fraction at close')
    ax.tick_params(axis='x',labelsize=6);ax.legend(ncol=3,fontsize=6,loc='upper left',frameon=False)
    ax.text(1,-.23,'Actual initial service',ha='center',transform=ax.get_xaxis_transform(),fontsize=7)
    ax.text(4,-.23,'Initial hold +0.95 s',ha='center',transform=ax.get_xaxis_transform(),fontsize=7)
    ax.set_title('(c) Native assigned-pad grasp',fontsize=9)
    ax=axes[1,1];ms=[report['fault_results']['density150'][w][m] for w in ('actual','1000ms') for m in METHODS]
    bottom=np.zeros(6)
    for key,color,title,hatch in [('wait','#d5ad55','Charged wait','//'),('skill','#4d80ad','Motion',None),('hold','#92999f','Final hold',None)]:
        y=np.array([m[key] for m in ms]);ax.bar(x,y,bottom=bottom,color=color,hatch=hatch,label=title);bottom+=y
    ax.set(xticks=x,xticklabels=['Fixed','Refresh','Retarget']*2,ylabel='All-attempt complete time (s)',ylim=(0,max(bottom)*1.2))
    ax.tick_params(axis='x',labelsize=6);ax.legend(ncol=3,fontsize=6,loc='upper left',frameon=False)
    ax.text(1,-.23,'Actual initial service',ha='center',transform=ax.get_xaxis_transform(),fontsize=7)
    ax.text(4,-.23,'Initial hold +0.95 s',ha='center',transform=ax.get_xaxis_transform(),fontsize=7)
    ax.set_title('(d) Full physical cost, failures retained',fontsize=9)
    for ax in axes.ravel():ax.grid(axis='y',alpha=.15)
    fig.subplots_adjust(left=.08,right=.99,top=.95,bottom=.12,wspace=.4,hspace=.55)
    save(fig,'interface-intervention')
    for case in ('rescued','limited'):
        if not (ROOT/'paper-keyframes'/f'{case}-retarget.json').exists():continue
        fig,axes=plt.subplots(2,4,figsize=(7.1,3.9))
        for i,method in enumerate(('refresh','retarget')):
            meta=json.loads((ROOT/'paper-keyframes'/f'{case}-{method}.json').read_text())
            for j,frame in enumerate(meta['frames']):
                ax=axes[i,j];rgb=plt.imread(ROOT/'paper-keyframes'/f'{case}-{method}-{j}-agentview.png')
                ax.imshow(rgb[170:512,15:512]);ax.set_xticks([]);ax.set_yticks([])
                for s in ax.spines.values():s.set_color('#aab2b8');s.set_linewidth(.5)
                title=('Observed resting pose','Pickup entry','Closed fingers','Final verification')[j]
                ax.set_title(title,fontsize=7)
                ax.set_xlabel(f"t={frame['seconds_after_fault']:.2f}s; grasps {sum(frame['evaluator']['holds'])}/2",fontsize=6.5)
            axes[i,0].set_ylabel(f"{'Refresh' if method=='refresh' else 'Retarget'}\n{meta['label']}",fontsize=8)
        fig.subplots_adjust(left=.06,right=.995,top=.94,bottom=.1,hspace=.45,wspace=.05)
        save(fig,f'interface-{case}-keyframes')


if __name__=='__main__':main()
