"""Standalone scientific comparison from completed report.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'report.json').read_text())
groups=['native/actual','native/1000ms','density150/actual','density150/1000ms']
labels=['Native\nMeasured','Native\n+0.95s','Heavy\nMeasured','Heavy\n+0.95s']
fig,axes=plt.subplots(1,3,figsize=(10.2,3.15))
colors=['#4879a8','#d37b37'];x=np.arange(4)
for i,(method,name) in enumerate([('retarget','Position retarget'),('visual','RGB feedback')]):
 data=[r['fault_results'][g][method] for g in groups];xx=x+(i-.5)*.36
 axes[0].bar(xx,[d['C'] for d in data],width=.34,color=colors[i],alpha=.25)
 axes[0].bar(xx,[d['Y'] for d in data],width=.34,color=colors[i],label=name)
 axes[1].bar(xx,[d['total'] for d in data],width=.34,color=colors[i],label=name)
 axes[1].bar(xx,[d['wait'] for d in data],width=.34,color=colors[i],hatch='///',edgecolor='white',linewidth=.3)
 axes[2].plot(x,[d['Y'] for d in data],marker='o',color=colors[i],label=name+' 20Hz')
 axes[2].plot(x,[d['native500_Y'] for d in data],marker='x',linestyle='--',color=colors[i],label=name+' 500Hz')
for ax in axes:
 ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set(ylabel='Completions / 40 fault states',ylim=(0,42),title='Solid: protected; pale: terminal')
axes[0].legend(fontsize=7)
axes[1].set(ylabel='Full post-fault seconds',title='All attempts; hatch:service holds')
axes[2].set(ylabel='Protected completions /40',ylim=(0,42),title='Recorded path guard sensitivity')
axes[2].legend(fontsize=6)
fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(ROOT/f'cross-controller.{suffix}',dpi=200,bbox_inches='tight')
plt.close(fig)

# Illustrations use all-cohort statistics above; case selection is descriptive.
scored=[json.loads(s) for s in (ROOT/'scored.jsonl').open()]
history=r['history_cases']
examples=[]
if history:examples.append(('sampling-alias',history[0]))
new_witness=next((c for c in scored if c['method']=='visual' and c['label']=='B'),None)
if new_witness:examples.append(('independent-terminal-gap',new_witness))
for filename,case in examples:
 folder=ROOT/'formal'/str(case['seed'])/f"{case['scenario']}-{case['load']}-{case['window']}-{case['method']}"
 source=json.loads((folder/'trial.json').read_text())
 with np.load(folder/'native500.npz') as record:a=record['values']
 a=a[a[:,1]>=2];t=a[:,0]-source['prefix_state']['time']
 fig,axes=plt.subplots(3,1,figsize=(7.2,6.2),sharex=True)
 for j,(name,color) in enumerate([('Arm0','#4879a8'),('Arm1','#d37b37'),('Table','#458b60')]):
  axes[0].step(t,a[:,3+j]+j*1.3,where='post',label=name,color=color)
 frames=[f for f in source['trace'] if f['phase'] in ('recovery','final_hold')]
 t20=np.array([f['time']-source['prefix_state']['time'] for f in frames])
 f20=np.array([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames])
 axes[0].scatter(t20[f20],np.full(np.sum(f20),-.2),marker='x',s=28,color='#bb2525',label='20Hz gap samples',zorder=5)
 axes[0].set(yticks=[],ylabel='Registered support')
 axes[0].legend(ncol=4,fontsize=7,loc='upper right')
 axes[1].plot(t,a[:,8],color='#4879a8');axes[1].set_ylabel('Center z (m)',fontsize=10)
 other=axes[1].twinx();other.plot(t,a[:,9],color='#d37b37',alpha=.7)
 other.set_ylabel('Tilt (deg)',color='#d37b37')
 axes[2].plot(t,a[:,12],color='#624f87');axes[2].set_yscale('symlog',linthresh=1.)
 axes[2].set_ylabel('Contact magnitude\nsum (N)',fontsize=10)
 axes[2].set_xlabel('Seconds after fault')
 flags=~(a[:,3].astype(bool)|a[:,4].astype(bool)|a[:,5].astype(bool))
 starts=np.flatnonzero(np.diff(np.r_[False,flags,False].astype(int))==1)
 ends=np.flatnonzero(np.diff(np.r_[False,flags,False].astype(int))==-1)
 for lo,hi in zip(starts,ends):
  if (hi-lo)*.002<=.1:continue
  for ax in axes:ax.axvspan(t[lo],t[min(hi,len(t)-1)],color='#cb4545',alpha=.18)
 starts20=np.flatnonzero(np.diff(np.r_[False,f20,False].astype(int))==1)
 ends20=np.flatnonzero(np.diff(np.r_[False,f20,False].astype(int))==-1)
 if len(starts20):
  j=int(np.argmax(ends20-starts20));left=t20[starts20[j]]-.12;right=t20[ends20[j]-1]+.12
  axes[2].set_xlim(left,right)
 for ax in axes:ax.grid(alpha=.15)
 fig.suptitle(f"{case['method']}, seed {case['seed']}, {case['scenario']}, {case['load']}, {case['window']}\n"
              f"20Hz sampled run: {case['unsupported_ms']:.0f}ms; native longest gap: {case['native500_gap_ms']:.0f}ms",fontsize=10)
 fig.tight_layout()
 for suffix in ['png','pdf']:fig.savefig(ROOT/f'{filename}.{suffix}',dpi=200,bbox_inches='tight')
 (ROOT/f'{filename}.json').write_text(json.dumps({'selection':'First reference history-only rejection and first independent-executor B case; illustrations do not select aggregate outcomes',
  'case':case,'source':str(folder/'trial.json'),'scope':'Recorded simulation samples; force norms are not net upward support and registered-support gaps are not automatically free flight.'},indent=2)+'\n')
 plt.close(fig)
