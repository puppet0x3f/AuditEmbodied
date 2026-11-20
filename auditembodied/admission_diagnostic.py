"""Matched physical suffixes for eleven frozen, unadmitted global decisions."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .duration_continuation import run_continuation
from .duration_closed_loop import trial_summary
from .duration_pilot import file_hash
from .mid_recovery_data import read,outcome
from .vlm_mid_recovery import select_forecast


def execute(spec):
    source,prediction,branch=spec
    from .l0 import _import_robosuite
    _import_robosuite(True)
    row=run_continuation((source,branch),transition='defer_rotation',account_control_latency=True)
    if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
    row.update(method='legacy_global' if branch=='R_global' else 'consistent_exit',
        source_wait_steps=source['source_wait_steps'],frozen_forecast=prediction,
        no_safe_candidate=all(x!='A' for x in prediction['labels'].values()),
        original_protection_violation=source['protection_violation'])
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.jsonl').exists():raise ValueError('diagnostic already exists')
    old=Path('outputs/vlm-two-window-rgbd-20261109');model=Path('outputs/vlm-mid-importance-20261110/corrected1024')
    threshold=json.loads((model/'protocol.json').read_text())['threshold']
    selected=[r for r in read(model/'train-predictions.jsonl') if r['decision_step']==30 and r['baseline_fallback']]
    assert len(selected)==11 and sum(all(c!='A' for c in r['labels'].values()) for r in selected)==6
    sources={}
    wanted={(r['seed'],r['scenario'],r['source_wait_steps']) for r in selected}
    for folder in ('dataset','dataset-extension'):
        for line in (old/folder/'trials.jsonl').open():
            r=json.loads(line);key=(r['seed'],r['scenario'],r['source_wait_steps'])
            if r['method']=='LOCAL60' and key in wanted:sources[key]=r
    assert set(sources)==wanted
    frozen={str(model/'adapter'/n):file_hash(model/'adapter'/n) for n in ('adapter_config.json','adapter_model.safetensors')}
    jobs=[]
    for i,r in enumerate(selected):
        legacy=select_forecast(r['records'],threshold,'legacy');fixed=select_forecast(r['records'],threshold,'consistent')
        assert legacy['action']=='R_global' and not legacy['model_admitted'] and fixed['action']=='reject'
        source=sources[(r['seed'],r['scenario'],r['source_wait_steps'])]
        for branch in (('R_global','reject') if i%2==0 else ('reject','R_global')):jobs.append((source,r,branch))
    protocol={'diagnostic_only':True,'used_for_training':False,'new_independent_test':False,
        'physical_states':11,'episodes_planned':22,'frozen':frozen,'threshold':threshold,
        'inference':'Immutable final frozen predictions; no new online model calls. Recorded physical waits replayed exactly, new control overruns and final hold charged. Online latency will be measured in the96 independent episodes.',
        'states':[{'seed':r['seed'],'scenario':r['scenario'],'wait':r['source_wait_steps']} for r in selected]}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (out/'trials.jsonl').open('w') as f:
            for row in pool.map(execute,jobs):
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                print(json.dumps({k:row.get(k) for k in ('seed','scenario','source_wait_steps','method','success','protection_violation','total_post_fault_seconds')}),flush=True)
    assert len(rows)==22 and all(file_hash(n)==h for n,h in frozen.items())
    pairs={}
    for r in rows:pairs.setdefault((r['seed'],r['scenario'],r['source_wait_steps']),{})[r['method']]=r
    cases=[]
    for key,pair in pairs.items():
        x,y=pair['legacy_global'],pair['consistent_exit']
        for name in ('boundary_state','boundary_robot','boundary_position_targets','boundary_rotation_targets','protection_before','boundary_wait_steps'):
            assert x['continuation_diagnostic'][name]==y['continuation_diagnostic'][name],(key,name)
        cases.append({'seed':key[0],'scenario':key[1],'wait':key[2],'all_labels':x['frozen_forecast']['all_labels'],
            'legacy_label':outcome(x),'exit_label':outcome(y),'legacy_s':x['total_post_fault_seconds'],'exit_s':y['total_post_fault_seconds'],
            'prefix_protection':x['continuation_diagnostic']['protection_before'],
            'legacy_protection':x['protection'],'exit_protection':y['protection']})
    report={'diagnostic_only':True,'physical_states':11,'episodes':22,'frozen_unchanged':True,'prefixes_and_waits_exact':True,
        'methods':{m:{**trial_summary([r for r in rows if r['method']==m]),
            'abstentions':sum(bool(r.get('abstained')) for r in rows if r['method']==m)} for m in ('legacy_global','consistent_exit')},'cases':cases}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
