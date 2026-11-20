"""Plot measured pose mismatch and actual contacts; no images or trajectories are invented."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
STUDY=ROOT/'outputs/p1-load-transfer-20261109'


def main():
    endpoints=list(csv.DictReader((OUT/'global-cases.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),constrained_layout=True)
    groups=[('native','actual'),('density150','actual'),('native','1000ms'),('density150','1000ms')]
    for i,(load,window) in enumerate(groups):
        rows=[r for r in endpoints if r['method']=='global' and r['scenario'] in ('left_loss','right_loss') and r['load']==load and r['window']==window]
        color='#287c75' if load=='native' else '#b06436'
        x=i+np.linspace(-.14,.14,len(rows))
        for ax,key,scale in [(axes[0],'global_table_settle_tilt_deg',1),(axes[1],'global_close_max_handle_eef_error_m',100)]:
            values=[float(r[key])*scale for r in rows]
            ax.scatter(x,values,s=8,alpha=.65,color=color,edgecolors='none')
            ax.plot([i-.2,i+.2],[np.median(values)]*2,color='#222222',linewidth=1.7)
    for ax in axes:
        ax.set_xticks(range(4),['Native\nMeasured','1.5x load\nMeasured','Native\n1 s','1.5x load\n1 s'],fontsize=7)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    axes[0].set_ylabel('Tilt at table-settle end (deg)',fontsize=8)
    axes[1].set_ylabel('Worst-arm handle--EEF miss\nat close end (cm)',fontsize=8)
    for ext in ('pdf','png'):
        fig.savefig(OUT/('regrasp-pose.'+ext),dpi=200,bbox_inches='tight')
        fig.savefig(ROOT/'paper/figures'/('regrasp-pose.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)

    # A native fault, its heavier counterpart, and the aligned heavier control.
    selected={}
    for row in map(json.loads,(STUDY/'seed-shards/50000.jsonl').open()):
        key=(row['load'],row['scenario'])
        if row['method']=='global' and row['window']=='actual' and key in (('native','left_loss'),('density150','left_loss'),('density150','sham')):
            selected[key]=row
    fig,axes=plt.subplots(3,1,figsize=(7.0,4.45),sharex=True,constrained_layout=True)
    colors=['#287c75','#b06436','#5976a6']
    for (key,row),color in zip(selected.items(),colors):
        frames=[f for f in row['trace'] if f['phase'] in ('recovery','final_hold')]
        times=np.asarray([f['time']-row['prefix_state']['time'] for f in frames])
        tilt=[f['evaluator']['tilt_deg'] for f in frames]
        miss=[np.linalg.norm(np.asarray(f['evaluator']['handle_positions'])-f['robot']['positions'],axis=1).max()*100 for f in frames]
        label='Native, left loss' if key[0]=='native' else '1.5x load, left loss' if key[1]=='left_loss' else '1.5x load, healthy'
        axes[0].plot(times,tilt,label=label,color=color,linewidth=1.2)
        axes[1].plot(times,miss,color=color,linewidth=1.2)
    for scenario,color in [('left_loss','#b06436'),('sham','#5976a6')]:
        rows=[json.loads(x) for x in (OUT/'contacts'/f'50000-{scenario}-density150-actual-global-samples.jsonl').open()]
        origin=selected[('density150',scenario)]['prefix_state']['time']
        times=[r['time']-origin for r in rows]
        values=[r['contact_force_norm_sums_N']['other_robot'] for r in rows]
        axes[2].plot(times,values,color=color,linewidth=1.1)
    reference=selected[('density150','left_loss')];groups=defaultdict(list)
    for f in reference['trace']:
        if f['phase'] in ('recovery','final_hold'):groups[f['stage']].append(f)
    for stage,label in [('global_table_settle','Settle'),('global_close','Close'),('global_lift','Lift')]:
        start=groups[stage][0]['time']-reference['prefix_state']['time']-.05;end=groups[stage][-1]['time']-reference['prefix_state']['time']
        for ax in axes:ax.axvspan(start,end,color='#b9c0c4',alpha=.18,zorder=0)
        axes[0].text((start+end)/2,1.02,label,transform=axes[0].get_xaxis_transform(),ha='center',fontsize=7)
    axes[0].set_ylabel('Tilt (deg)',fontsize=8);axes[1].set_ylabel('Worst-arm miss (cm)',fontsize=8)
    axes[2].set_ylabel('Other robot contact\nforce-norm sum (N)',fontsize=8)
    axes[2].set_xlabel('Physical seconds after fault',fontsize=8)
    axes[0].legend(fontsize=7,frameon=False,ncol=3,loc='upper left')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16);ax.tick_params(labelsize=8)
    for ext in ('pdf','png'):
        fig.savefig(OUT/('regrasp-path.'+ext),dpi=200,bbox_inches='tight')
        fig.savefig(ROOT/'paper/figures'/('regrasp-path.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('Generated population stage measurements and an actual matched-seed/contact path figure.')


if __name__=='__main__':main()
