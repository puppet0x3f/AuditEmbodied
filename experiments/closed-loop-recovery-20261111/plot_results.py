"""Scientific plots of frozen controller-transfer results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'analysis/report.json').read_text())
assert r['episodes']==480
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(11,6.7),constrained_layout=True)
for ax,load,title in zip(axs[0],('native','density150'),('Native load','Heavy load')):
    groups=[r['fault_results'][load][w][m] for w in ('actual','1000ms') for m in ('retarget','closed_loop')]
    x=np.arange(4)
    for delta,key,label,color in [(-.23,'C','Terminal C','#aab3c0'),(0,'Y','Full-path Y20','#34836f'),(.23,'native_Y','Native-guard Y500','#586eb3')]:
        ax.bar(x+delta,[g[key] for g in groups],.22,label=label,color=color)
    ax.set(xticks=x,xticklabels=['Target\nActual','RGB loop\nActual','Target\n1 s','RGB loop\n1 s'],
           ylim=(0,43),ylabel='Fault completions /40',title=title)
    ax.set_yticks([0,10,20,30,40]);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axs[0,0].legend(fontsize=8,loc='lower left')
groups=[(load,w,m,r['fault_results'][load][w][m]) for load in ('native','density150') for w in ('actual','1000ms') for m in ('retarget','closed_loop')]
ax=axs[1,0];x=np.arange(8);bottom=np.zeros(8)
for key,label,color in [('wait','Charged waits','#cf9762'),('skill','Skill execution','#638bb5'),('hold','Final verification','#72a698')]:
    values=np.array([g[key] for load,w,m,g in groups]);ax.bar(x,values,.7,bottom=bottom,color=color,label=label);bottom+=values
ax.set(xticks=x,xticklabels=[('N' if l=='native' else 'H')+(' a' if w=='actual' else ' 1')+'\n'+('Target' if m=='retarget' else 'Loop') for l,w,m,g in groups],
       ylabel='All-attempt full time (s)',title='Complete cost; failures retained')
ax.legend(fontsize=8,ncol=3,loc='upper left');ax.set_ylim(0,max(bottom)*1.23)
ax=axs[1,1];x=np.arange(4)
waits=[r['paired_waits'][l][m] for l in ('native','density150') for m in ('retarget','closed_loop')]
values=np.array([w['total_difference'] for w in waits]);intervals=np.array([w['total_difference_ci95'] for w in waits])
ax.errorbar(x,values,yerr=np.stack([values-intervals[:,0],intervals[:,1]-values]),fmt='o',color='#3b627f',capsize=4)
ax.axhline(.95,color='#b97b43',ls='--',lw=1,label='Added initial hold: 0.95 s')
ax.axhline(0,color='grey',lw=.5)
ax.set(xticks=x,xticklabels=['Native\nTarget','Native\nRGB loop','Heavy\nTarget','Heavy\nRGB loop'],
       ylabel='Change in full time (s)',title='Waiting changes the realized path cost')
ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18)
fig.savefig(HERE/'controller-transfer.pdf',bbox_inches='tight')
fig.savefig(HERE/'controller-transfer.png',dpi=180,bbox_inches='tight')
