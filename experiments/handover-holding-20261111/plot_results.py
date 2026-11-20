"""Publication views of all frozen handover holding outcomes."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'reproducibility/handover_holding_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def main():
    rows=[json.loads(s) for s in (HERE/'formal-analysis/scored.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,2,figsize=(7.05,3.7),sharex=True)
    colors=('#b76536','#3976a8','#407d63')
    for col,method in enumerate(audit.METHODS):
        for step,offset,alpha in [(1,-.17,.45),(20,.17,1.)]:
            groups=[[r for r in rows if r['method']==method and r['hold_mode']==hold and
                r['wait_steps']==step and r['scenario']!='nominal'] for hold in audit.HOLDS]
            for ax,field,scale in [(axes[0,col],'native_Y',100),(axes[1,col],'total',1)]:
                ds=[]
                for g in groups:
                    g=[r for r in g if r[field] is not None]
                    seeds=sorted({r['seed'] for r in g})
                    ds.append(audit.interval([np.mean([r[field] for r in g if r['seed']==seed])*scale for seed in seeds]))
                means=[d['estimate'] for d in ds]
                err=np.array([[d['estimate']-d['ci95'][0],d['ci95'][1]-d['estimate']] for d in ds]).T
                ax.bar(np.arange(3)+offset,means,.34,yerr=err,capsize=2,color=colors,alpha=alpha,label=f'{step*.05:.2f} s')
                ax.grid(axis='y',alpha=.14)
        axes[0,col].set(title=audit.METHOD_NAMES[method],ylim=(0,108))
        axes[1,col].set_xticks(range(3),['Registered','Position only','Smooth position'],fontsize=7)
        axes[1,col].set_ylim(bottom=0)
    axes[0,0].set_ylabel('Protected completion (%)')
    axes[1,0].set_ylabel('Complete time (s)')
    axes[0,1].legend(frameon=False,loc='lower right',fontsize=7)
    fig.tight_layout(pad=.7)
    for ext in ('pdf','png'):
        p=HERE/f'handover-holding.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight')
        (ROOT/'paper/figures'/p.name).write_bytes(p.read_bytes())


if __name__=='__main__':main()
