"""Figures use recorded executions; no selection of successful-only time samples."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'reproducibility'))
from holding_components_audit import interval,HOLDS,NAMES

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
COLORS={'registered':'#b76536','measured_position':'#3976a8','measured_rotation':'#8b6ca7','measured_pose':'#407d63'}
TARGET_COLORS={'registered':'#9467bd','measured_pose':'#377eb8'}


def load(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(fig,name):
    fig.tight_layout(pad=.65)
    for suffix in ('pdf','png'):fig.savefig(HERE/(name+'.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)


def curve(ax,rows,hold,field,label,method='closed_loop',scale=1.,style='-'):
    group=[r for r in rows if r['scenario']!='sham' and r['method']==method and r['hold_mode']==hold]
    steps=sorted({r['wait_steps'] for r in group});means=[];lo=[];hi=[]
    for step in steps:
        a=[r for r in group if r['wait_steps']==step];seeds=sorted({r['seed'] for r in a})
        result=interval([np.mean([r[field] for r in a if r['seed']==s]) for s in seeds])
        m=result['estimate']*scale;l,h=np.asarray(result['ci95'])*scale
        means.append(m);lo.append(m-l);hi.append(h-m)
    color=COLORS[hold] if method=='closed_loop' else TARGET_COLORS[hold]
    ax.errorbar(np.asarray(steps)*.05,means,yerr=[lo,hi],label=label,color=color,
        marker='o' if method=='closed_loop' else 's',ms=3,lw=1.25,capsize=2,ls=style)
    ax.set_xticks(np.asarray(steps)*.05);ax.set_xlabel('Initial waiting (s)');ax.grid(alpha=.16,axis='y')


def main():
    rows=load(ROOT/'outputs/wait-hold-geometry-20261111/wait-analysis/scored.jsonl')
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.17))
    for ax,field,scale,title in [(axes[0],'native_Y',100.,'(a) Protected completion (%)'),
                                (axes[1],'total',1.,'(b) Complete time (s)')]:
        for hold,name in [('registered','Registered targets'),('measured_pose','Measured poses')]:
            curve(ax,rows,hold,field,name,scale=scale)
        if field=='native_Y':
            curve(ax,rows,'registered',field,'Target / registered',method='retarget',scale=scale,style=':')
            curve(ax,rows,'measured_pose',field,'Target / measured',method='retarget',scale=scale,style='--')
        ax.set_title(title,pad=18)
    axes[0].set_ylim(-3,103)
    axes[0].legend(frameon=False,fontsize=6.3,loc='upper left',ncol=2,
                   bbox_to_anchor=(0,1.12),handlelength=1.0,handletextpad=.25,
                   columnspacing=.45,labelspacing=.35,borderaxespad=.15,borderpad=.1)
    axes[1].legend(frameon=False,fontsize=6.3,loc='upper left',
                   bbox_to_anchor=(0,1.12),handlelength=1.0,handletextpad=.25,
                   labelspacing=.35,borderaxespad=.15,borderpad=.1)
    x=np.arange(3);width=.34
    for hold,offset,label in [('registered',-width/2,'Registered targets'),
                              ('measured_pose',width/2,'Measured poses')]:
        group=[r for r in rows if r['scenario']!='sham' and r['method']=='closed_loop' and r['hold_mode']==hold and r['wait_steps']==20]
        values=[sum(r['first_close_dual'] for r in group),sum(r['pickup_attempts']>1 for r in group),sum(r['native_Y'] for r in group)]
        bars=axes[2].bar(x+offset,np.asarray(values)/40*100,width,color=COLORS[hold],label=label)
        for b,v in zip(bars,values):axes[2].text(b.get_x()+b.get_width()/2,b.get_height()+2,str(v)+'/40',ha='center',fontsize=6)
    axes[2].set_xticks(x,['First dual\ngrasp','Pickup\nretry','Protected\nfinish']);axes[2].set_ylim(0,105)
    axes[2].set_title('(c) RGB recovery after 1 s wait',pad=18);axes[2].set_ylabel('Fault executions (%)')
    axes[2].legend(frameon=False,fontsize=6.3,loc='upper left',
                   bbox_to_anchor=(0,1.12),handlelength=1.0,handletextpad=.25,
                   labelspacing=.35,borderaxespad=.15,borderpad=.1)
    save(fig,'waiting-control-overview')

    fig,ax=plt.subplots(figsize=(3.42,2.1))
    for hold,name in [('registered','Registered'),('measured_pose','Measured')]:
        for step,style in [(1,'--'),(20,'-')]:
            group=[r for r in rows if r['scenario']!='sham' and r['method']=='closed_loop' and r['hold_mode']==hold and r['wait_steps']==step]
            times=sorted({0.,45.,*[r['total'] for r in group]})
            values=[np.mean([r['native_Y'] and r['total']<=t for r in group])*100 for t in times]
            ax.step(times,values,where='post',color=COLORS[hold],ls=style,lw=1.25,label=f'{name}, {step*.05:.2f} s')
    ax.set(xlabel='Complete execution budget (s)',ylabel='Protected within budget (%)',xlim=(0,45),ylim=(-2,103))
    ax.legend(frameon=False,fontsize=6.6,loc='upper left');ax.grid(alpha=.16)
    save(fig,'waiting-budget')

    for phase in ('mechanism','independent'):
        path=HERE/(phase+'-analysis/scored.jsonl')
        if not path.exists():continue
        a=load(path);holds=[h for h in HOLDS if any(r['hold_mode']==h for r in a)]
        fig,axes=plt.subplots(1,2,figsize=(7.05,2.2))
        x=np.arange(len(holds));width=.34
        for step,offset,alpha in [(1,-width/2,.45),(20,width/2,1.)]:
            for ax,field,scale,title in [(axes[0],'native_Y',100.,'Native-guard protected completion (%)'),
                                        (axes[1],'total',1.,'Complete physical time (s)')]:
                means=[];lower=[];upper=[]
                for hold in holds:
                    g=[r for r in a if r['scenario']!='sham' and r['hold_mode']==hold and r['wait_steps']==step]
                    seeds=sorted({r['seed'] for r in g})
                    v=interval([np.mean([r[field] for r in g if r['seed']==s]) for s in seeds])
                    m=v['estimate']*scale;l,h=np.asarray(v['ci95'])*scale
                    means.append(m);lower.append(m-l);upper.append(h-m)
                ax.bar(x+offset,means,width,color=[COLORS[h] for h in holds],alpha=alpha,
                    yerr=[lower,upper],capsize=2,error_kw={'lw':.8},label=f'{step*.05:.2f} s')
                ax.set_xticks(x,[NAMES[h].replace(' + ',' +\n') for h in holds],fontsize=7)
                ax.set_title(title);ax.grid(alpha=.13,axis='y')
        axes[0].set_ylim(0,103);axes[1].legend(frameon=False,fontsize=7)
        save(fig,'holding-components' if phase=='mechanism' else 'independent-visual')


if __name__=='__main__':main()
