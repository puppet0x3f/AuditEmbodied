"""Matched observations before and after the existing feedback grasp prefix."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from . import l4_lift
from .duration_closed_loop import render_rgb
from .duration_pilot import STATE_KEYS,prefix_frames,retimed_config
from .l0 import _import_robosuite
from .vlm_duration_data import outcome,plan


def robot_only(port):
    return {k:port[k] for k in ('positions','fingers','reference_positions')}


def collect_state(job):
    seed,scenario,folder,post_wait=job
    _import_robosuite(True)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    candidate='R_left' if scenario=='left_loss' else 'R_right'
    original=l4_lift.TwoArmLiftEnv;rows=[]
    for n in (30,40,60):
        recorded={};holder={}
        def factory(*args,**kwargs):
            env=original(*args,**kwargs);env.raw.has_offscreen_renderer=True;holder['env']=env
            reset=env.reset
            def warm_reset():
                reset();render_rgb(env)
            env.reset=warm_reset
            return env
        def capture(name,port):
            env=holder['env'];item={'robot':robot_only(port),'history':port.get('observed_history',[]),
                'state':l4_lift.snapshot(env)}
            if n==30:
                tick=time.perf_counter();rgb=render_rgb(env)
                if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid stage RGB')
                image=folder/f'{seed}-{scenario}-{name}.npz';np.savez_compressed(image,rgb=rgb)
                item.update(image=str(image.resolve()),acquisition_seconds=time.perf_counter()-tick)
            recorded[name]=item
        def select(port):
            capture('h0',port)
            return {'candidate':candidate,'parallel_local':True}
        def duration(operation,port,role,prefix):
            if operation=='start':
                capture('h1',port);return
            capture('h2',port)
            return {'candidate':role,'accepted':True,'local_relevel_steps':n,'pipeline_wall_seconds':0.,
                'enumerated_candidate_not_model_decision':True}
        tick=time.perf_counter()
        # All candidates need state captures; only the first renders images.
        if n==30:
            factory_used=factory
        else:
            def factory_used(*args,**kwargs):
                env=original(*args,**kwargs);holder['env']=env;return env
        with patch.object(l4_lift,'TwoArmLiftEnv',factory_used):
            row=l4_lift.run_trial(seed,scenario,config=retimed_config(n/40,'relevel_only'),
                selector=select,duration_selector=duration,fixed_selector_wait_steps=19,
                refresh_duration_observation=True,fixed_duration_wait_steps=post_wait)
        row.update(experiment_stage='post_grasp_observation',scope_wait_steps=19,post_prefix_wait_steps=post_wait,
            stage_capture=recorded,wall_seconds=time.perf_counter()-tick)
        rows.append(row)
    return rows


def collect(output,wait_steps,seeds=range(40,48),workers=4):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    destination=output/f'trials-wait{wait_steps}.jsonl'
    if destination.exists():raise ValueError('collection exists')
    jobs=[(s,q,str((output/'observations'/str(wait_steps)).resolve()),wait_steps)
          for s in seeds for q in ('left_loss','right_loss')]
    rows=[];started=time.perf_counter()
    with destination.open('w') as stream:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for group in pool.map(collect_state,jobs):
                for row in group:rows.append(row);stream.write(json.dumps(row)+'\n')
                stream.flush()
                print(json.dumps({'wait':wait_steps,'seed':group[0]['seed'],'scenario':group[0]['scenario'],
                    'completed':len(rows),'labels':[outcome(r) for r in group]}),flush=True)
    (output/f'collection-wait{wait_steps}.json').write_text(json.dumps(
        {'episodes':len(rows),'wall_seconds':time.perf_counter()-started},indent=2)+'\n')
    return rows


def stage_plan(row):
    p=plan(row)
    p.update(scope_hold_steps=19,post_prefix_wait_steps=row['post_prefix_wait_steps'],final_hold_steps=20)
    return p


def build(output,wait_steps):
    output=Path(output);rows=[]
    for w in (0,wait_steps):
        rows.extend(json.loads(s) for s in (output/f'trials-wait{w}.jsonl').read_text().splitlines())
    groups={}
    for row in rows:groups.setdefault((row['seed'],row['scenario'],row['post_prefix_wait_steps']),[]).append(row)
    examples={'train':[],'development':[]};cases=[]
    for (seed,scenario,w),group in sorted(groups.items()):
        ref=next(r for r in group if r['local_steps']['local_relevel_steps']==30)
        zero=groups[seed,scenario,0][0];canonical=zero['stage_capture']
        for row in group:
            assert not row.get('error') and row.get('fault_class_valid') and not row.get('blocked_calls')
            assert row.get('state_edit_count_after_initialization')==row.get('contact_anchor_count')==row.get('external_wrench_at_end')==0
            assert all(row[k]==ref[k]==zero[k] for k in STATE_KEYS)
            assert prefix_frames(row)==prefix_frames(ref)==prefix_frames(zero)
            for stage in ('h0','h1','h2'):
                c=row['stage_capture'][stage];r=ref['stage_capture'][stage];z=canonical[stage]
                assert all(c[k]==r[k]==z[k] for k in ('state','robot','history'))
            boundary=row['duration_selection']['boundary_state']['time']
            prefix=lambda r:[f for f in r['trace'] if f['time']<=boundary+1e-9]
            assert prefix(row)==prefix(ref)==prefix(zero)
            holds=[f for f in row['trace'] if f['stage']=='duration_budget_wait']
            assert len(holds)==w and all(f['phase']=='recovery' for f in holds)
            assert holds==[f for f in ref['trace'] if f['stage']=='duration_budget_wait']
            assert abs(row['inference_wait_seconds']-(19+w)/20)<1e-8
            assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-8
            keeper=1 if row['candidate']=='R_left' else 0
            label=outcome(row) if row['post_wait_evaluator']['holds'][keeper] else 'E'
            split='train' if seed<=45 else 'development'
            examples[split].append({'seed':seed,'scenario':scenario,'split':split,
                'image':canonical['h1']['image'],'robot':canonical['h1']['robot'],
                'before_image':canonical['h0']['image'],'before_robot':canonical['h0']['robot'],
                'after_prefix_image':canonical['h2']['image'],'after_prefix_robot':canonical['h2']['robot'],
                'early_history':canonical['h1']['history'],'late_history':canonical['h2']['history'],
                'plan':stage_plan(row),'label':label,'whole_episode_label':outcome(row),
                'wait_steps':w,'post_prefix_wait_steps':w,'source':'post_grasp_observation'})
        if w:
            for stage in ('h0','h1','h2'):
                with np.load(ref['stage_capture'][stage]['image']) as a,np.load(canonical[stage]['image']) as b:
                    assert np.array_equal(a['rgb'],b['rgb'])
        cases.append({'seed':seed,'scenario':scenario,'post_prefix_wait_steps':w,
            'labels':{str(r['local_steps']['local_relevel_steps']):outcome(r) for r in group},
            'prefix_protection':ref['duration_selection']['prefix_protection'],
            'h2_holds_evaluation_only':next(f['evaluator']['holds'] for f in reversed(ref['trace'])
                if abs(f['time']-ref['stage_capture']['h2']['state']['time'])<1e-8)})
    assert len(rows)==96 and len(groups)==32 and len(examples['train'])==72 and len(examples['development'])==24
    data=output/'dataset';data.mkdir()
    for split,records in examples.items():(data/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
    report={'physical_episodes':len(rows),'valid':True,'wait_steps':wait_steps,'scope_wait_steps':19,
        'split_classes':{s:dict(Counter(r['label'] for r in v)) for s,v in examples.items()},'cases':cases,
        'labels':'Whole post-fault episode including prefix and all holds; h2 contact truth is diagnostic only, never model input.'}
    (output/'physical-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True)
    p.add_argument('--wait-steps',type=int,required=True);p.add_argument('--build',action='store_true')
    a=p.parse_args()
    if a.build:print(json.dumps(build(a.output_dir,a.wait_steps),indent=2))
    else:collect(a.output_dir,a.wait_steps)


if __name__=='__main__':main()
