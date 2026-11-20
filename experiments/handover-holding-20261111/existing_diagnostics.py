"""Describe existing reference interventions; no new physical executions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'outputs/paired-executor-holding-20261111/prior-diagnostics.jsonl'


def interval(values):
    a=np.asarray(values,float)
    draws=np.random.default_rng(20261116).integers(0,len(a),(2000,len(a)))
    return {'mean':float(a.mean()),'ci95':np.quantile(a[draws].mean(1),[.025,.975]).tolist()}


def main():
    rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
    rows=[r for r in rows if r['method']=='closed_loop' and r['scenario']!='sham'
          and r['wait_steps']==20 and r['hold_mode'] in ('registered','measured_position')]
    assert len(rows)==80
    fields=('hold_entry_position_offset_mm','hold_mean_position_error_mm',
            'wait_translation_mm','wait_support_fraction','wait_contact_changes',
            'wait_end_speed_m_s','first_close_dual','native_Y')
    report={'source_cohort':'70000--70019 CAD holding factorial','physical_executions_added':0,
            'faults_per_condition':40,'initialization_seeds':20,'conditions':{}}
    for hold in ('registered','measured_position'):
        g=[r for r in rows if r['hold_mode']==hold]
        seeds=sorted({r['seed'] for r in g})
        report['conditions'][hold]={field:interval([
            np.mean([r[field] for r in g if r['seed']==seed]) for seed in seeds]) for field in fields}
    (HERE/'control-reference-diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.05))
    holds=('registered','measured_position');colors=('#b76536','#3976a8')
    panels=[('hold_entry_position_offset_mm','Entry: preceding target offset','mm'),
            ('hold_mean_position_error_mm','During hold: tracking error','mm'),
            ('wait_translation_mm','Object translation during hold','mm')]
    for ax,(field,title,unit) in zip(axes,panels):
        ds=[report['conditions'][h][field] for h in holds]
        means=[d['mean'] for d in ds]
        error=np.array([[d['mean']-d['ci95'][0],d['ci95'][1]-d['mean']] for d in ds]).T
        ax.bar(range(2),means,yerr=error,capsize=2,color=colors,width=.55)
        for j,d in enumerate(ds):ax.text(j,d['ci95'][1]+.5,f"{d['mean']:.1f}",ha='center',fontsize=7)
        ax.set_xticks(range(2),['Registered','Position only'],fontsize=7)
        ax.set(title=title,ylabel=unit,ylim=(0,max(d['ci95'][1] for d in ds)*1.25+1))
        ax.grid(axis='y',alpha=.15)
    fig.tight_layout(pad=.6)
    for ext in ('pdf','png'):
        path=HERE/f'control-reference-diagnostics.{ext}'
        fig.savefig(path,dpi=220,bbox_inches='tight')
        (ROOT/'paper/figures'/path.name).write_bytes(path.read_bytes())
    print(json.dumps(report))


if __name__=='__main__':main()
