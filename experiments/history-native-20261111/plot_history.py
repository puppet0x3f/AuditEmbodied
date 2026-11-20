"""Original ten replay gaps plus one exact recorded sampling illustration."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'history-report.json').read_text());rows=r['rows'];case=rows[0]
source=json.loads((ROOT/'source-cases'/f"{case['name']}.json").read_text())
with np.load(ROOT/'native'/f"{case['name']}.npz") as f:a=f['values']
a=a[a[:,1]>=2];t=a[:,0]-source['prefix_state']['time']
frames=[f for f in source['trace'] if f['phase'] in ('recovery','final_hold')]
t20=np.array([f['time']-source['prefix_state']['time'] for f in frames])
missing20=np.array([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames])
starts=np.flatnonzero(np.diff(np.r_[False,missing20,False].astype(int))==1)
ends=np.flatnonzero(np.diff(np.r_[False,missing20,False].astype(int))==-1)
j=int(np.argmax(ends-starts));lo=t20[starts[j]]-.08;hi=t20[ends[j]-1]+.08
fig=plt.figure(figsize=(10.8,4.3));g=fig.add_gridspec(2,2,width_ratios=[1.25,1.05],hspace=.3,wspace=.28)
bar=fig.add_subplot(g[:,0]);support=fig.add_subplot(g[0,1]);force=fig.add_subplot(g[1,1],sharex=support)
x=np.arange(10)
bar.bar(x-.17,[q['primary_gap_ms'] for q in rows],width=.34,label='Primary 20 Hz run',color='#b96542')
bar.bar(x+.17,[q['native_gap_ms'] for q in rows],width=.34,label='Native 500 Hz gap',color='#427eaa')
labels=[str(q['seed'])+('L' if q['scenario']=='left_loss' else 'R')+('\n1s' if q['window']=='1000ms' else '\nActual') for q in rows]
bar.set_xticks(x,labels,rotation=55,ha='right',fontsize=7)
bar.set(ylabel='Longest registered-support absence (ms)',title='All ten original completed history cases',ylim=(0,325))
bar.legend(fontsize=8);bar.grid(axis='y',alpha=.2);bar.set_axisbelow(True)
registered=a[:,3].astype(bool)|a[:,4].astype(bool)|a[:,5].astype(bool)
support.step(t,registered.astype(int),where='post',color='#427eaa',label='Native support')
support.scatter(t20[missing20],np.full(missing20.sum(),-.15),marker='x',s=28,color='#b33c32',label='20 Hz missing')
support.set(ylim=(-.3,1.25),yticks=[0,1],ylabel='Registered support',title=f"{case['seed']}L: 150 ms sampled, 14 ms native")
support.legend(fontsize=7,loc='upper right');support.tick_params(labelbottom=False)
force.plot(t,a[:,12],color='#6c5887');force.set_yscale('symlog',linthresh=1)
force.set(xlim=(lo,hi),xlabel='Seconds after original fault',ylabel='Contact magnitude\nsum (N)')
for ax in (support,force):ax.grid(alpha=.18)
fig.subplots_adjust(bottom=.24,top=.91)
for suffix in ('pdf','png'):fig.savefig(ROOT/f'history-native-replay.{suffix}',dpi=200,bbox_inches='tight')
plt.close(fig)
