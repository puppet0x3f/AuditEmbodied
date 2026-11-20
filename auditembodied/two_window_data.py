"""Matched LOCAL60 / GLOBAL30 / GLOBAL40 actuator executions and causal inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import l4_lift
from .duration_closed_loop import render_rgb
from .duration_continuation import run_continuation,boundary_prefix
from .duration_stage_data import robot_only
from .mid_recovery_data import load_jobs,read,outcome
from .vlm_duration_data import plan


def collect(spec):
    source,metadata,score,folder=spec;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    from .l0 import _import_robosuite
    _import_robosuite(True)
    original=l4_lift.TwoArmLiftEnv;holder={};packets={};queries={};images={}
    def factory(*a,**k):
        env=original(*a,**k);env.raw.has_offscreen_renderer=True;holder['env']=env;reset=env.reset
        def warmed():reset();render_rgb(env)
        env.reset=warmed;return env
    seed=metadata['seed'];scenario=metadata['scenario'];wait=metadata['wait_steps']
    role='R_left' if scenario=='left_loss' else 'R_right'
    def image(name):
        path=folder/f'{seed}-{scenario}-W{wait}-{name}.npz'
        np.savez_compressed(path,rgb=render_rgb(holder['env']))
        images[name]=str(path.resolve())
    def select(port):
        image('h0');packets['before_robot']=robot_only(port)
        return {'candidate':role,'parallel_local':True}
    def duration(operation,port,candidate,prefix):
        if operation=='start':return
        return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,'enumerated_candidate_not_model_decision':True,'pipeline_wall_seconds':0.}
    def observe(operation,port,candidate):
        if operation=='start':
            packets['entry']=port;image('entry');return
        if operation=='lookahead':queries[port['relevel_step']]=port;return
        return {'action':'continue','forecast_ready':True,'attempt_id':port['attempt_id'],'baseline_fallback':False}
    if source is not None:
        opened=sum(f['stage']=='fault_open' for f in source['trace'])
        withdrawn=sum(f['stage']=='fault_withdraw' for f in source['trace'])
        scope=sum(f['stage']=='inference_wait' for f in source['trace'])
    else:opened=metadata.get('fault_open_steps') or (8 if seed%2==0 else 16);withdrawn=12;scope=19
    cfg=replace(l4_lift.LiftConfig(),local_relevel_steps=60,fault_open_steps=opened,fault_withdraw_steps=withdrawn)
    with patch.object(l4_lift,'TwoArmLiftEnv',factory):
        local=l4_lift.run_trial(seed,scenario,config=cfg,selector=select,fixed_selector_wait_steps=scope,
            duration_selector=duration,refresh_duration_observation=True,fixed_duration_wait_steps=wait,
            continuation_selector=observe,continuation_windows=(30,40),continuation_features=True)
    if local.get('error'):raise RuntimeError(local['error'])
    local.update(method='LOCAL60',split=metadata['split'],source_wait_steps=wait)
    if set(queries)!={20,30}:
        return [local],[],{'seed':seed,'scenario':scenario,'wait':wait,'reason':'no_complete_relevel','local_label':outcome(local)}
    branches=[]
    for step in (30,40):
        row=run_continuation((local,'R_global'),transition='defer_rotation',account_control_latency=True,boundary=f'relevel{step}')
        if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
        row.update(method=f'GLOBAL{step}',split=metadata['split'],source_wait_steps=wait)
        branches.append(row)
    p=plan({'candidate':role,'local_steps':{'local_relevel_steps':60}})
    p.update(post_prefix_wait_steps=wait,final_hold_steps=20)
    examples=[]
    for step,glob in zip((30,40),branches):
        labels={'local':outcome(local),'global':outcome(glob)}
        if step==30:labels['defer']=outcome(branches[1])
        examples.append({'two_window':True,'seed':seed,'scenario':scenario,'split':metadata['split'],
            'before_image':images['h0'],'entry_image':images['entry'],'before_robot':packets['before_robot'],
            'plan':p,'entry':packets['entry'],'query':queries[step-10],
            'decision_step':step,'start_valid_score':score,'labels':labels,
            'all_labels':{'LOCAL60':outcome(local),'GLOBAL30':outcome(branches[0]),'GLOBAL40':outcome(branches[1])},
            'source_wait_steps':wait,'prefix_exact':True,'costs':{'local':(60-step+28)/20,'global':12.6},
            'source_local_total_seconds':local['total_post_fault_seconds'],'actual_global_total_seconds':glob['total_post_fault_seconds']})
    return [local,*branches],examples,None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    p.add_argument('--new-only',action='store_true');p.add_argument('--new-train-seeds',type=int,nargs='+',default=list(range(84,92)))
    p.add_argument('--new-development-seeds',type=int,nargs='+',default=list(range(92,96)));p.add_argument('--fault-open-steps',type=int)
    args=p.parse_args();out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'trials.jsonl').exists():raise ValueError('collection exists')
    old,excluded=load_jobs();jobs=[] if args.new_only else [(*j,str(out/'observations')) for j in old]
    for seed in args.new_train_seeds+args.new_development_seeds:
        for scenario in ('left_loss','right_loss'):
            for wait in (0,66):
                jobs.append((None,{'seed':seed,'scenario':scenario,'wait_steps':wait,
                    'split':'train' if seed in args.new_train_seeds else 'development','fault_open_steps':args.fault_open_steps},None,str(out/'observations')))
    if args.smoke:jobs=[j for j in jobs if j[1]['seed'] in (43,84) and j[1]['scenario']=='left_loss' and j[1]['wait_steps']==0][:2]
    protocol={'families_planned':len(jobs),'excluded_old':excluded,'new_train_seeds':args.new_train_seeds,
        'new_development_seeds':args.new_development_seeds,'new_fault_open_steps':args.fault_open_steps,'existing_splits_preserved':True,
        'diagnostic_seeds_used':False,'branches':['LOCAL60','GLOBAL30','GLOBAL40'],
        'observations':[20,30],'decisions':[30,40],'entry_images':'fresh four RGB, captured inside charged control period',
        'all_waits':'Actual entry acquisition overruns replayed at their original positions; branch prefixes exact; original violation counters retained.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    samples={'train':[],'development':[]};rows=[];failures=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (out/'trials.jsonl').open('w') as f:
            for executed,examples,failure in pool.map(collect,jobs):
                for row in executed:rows.append(row);f.write(json.dumps(row)+'\n')
                f.flush()
                for row in examples:samples[row['split']].append(row)
                if failure:failures.append(failure)
                print(json.dumps({'completed_episodes':len(rows),'seed':executed[0]['seed'],
                    'scenario':executed[0]['scenario'],'wait':executed[0]['source_wait_steps'],
                    'labels':[outcome(r) for r in executed],'failure':failure}),flush=True)
    for split,items in samples.items():(out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    report={**protocol,'episodes':len(rows),'window_examples':{k:len(v) for k,v in samples.items()},'non_relevel_states':failures,
        'wall_seconds':time.perf_counter()-started,'label_pairs':{k:dict(Counter(str(r['decision_step'])+':'+r['labels']['local']+r['labels']['global'] for r in v)) for k,v in samples.items()}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
