"""Predetermined training-only interventions, six matched branches per state."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from auditembodied import l4_lift
from auditembodied.duration_closed_loop import render_rgb
from auditembodied.duration_stage_data import robot_only,stage_plan
from auditembodied.duration_pilot import STATE_KEYS,prefix_frames,retimed_config
from auditembodied.l0 import _import_robosuite
from auditembodied.vlm_duration_data import outcome

ROOT=Path(__file__).resolve().parent


def collect_state(job):
    seed,scenario=job;_import_robosuite(True)
    folder=ROOT/'collection/observations';folder.mkdir(parents=True,exist_ok=True)
    candidate='R_left' if scenario=='left_loss' else 'R_right'
    original=l4_lift.TwoArmLiftEnv;rows=[];canonical_wait=None
    for w in (0,66):
        for n in (30,40,60):
            first=w==0 and n==30;recorded={};holder={}
            def factory(*args,**kwargs):
                env=original(*args,**kwargs);holder['env']=env
                if first:
                    env.raw.has_offscreen_renderer=True;reset=env.reset
                    def warm_reset():reset();render_rgb(env)
                    env.reset=warm_reset
                return env
            def capture(name,port):
                env=holder['env'];item={'robot':robot_only(port),'history':port.get('observed_history',[]),
                    'state':l4_lift.snapshot(env)}
                if first:
                    tick=time.perf_counter();rgb=render_rgb(env)
                    if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid RGB')
                    image=folder/f'{seed}-{scenario}-{name}.npz';np.savez_compressed(image,rgb=rgb)
                    item.update(image=str(image.resolve()),acquisition_seconds=time.perf_counter()-tick)
                recorded[name]=item
            def select(port):
                capture('h0',port)
                return {'candidate':candidate,'parallel_local':True}
            def duration(operation,port,role,prefix):
                if operation=='start':capture('h1',port);return
                capture('h2',port)
                return {'candidate':role,'accepted':True,'local_relevel_steps':n,'pipeline_wall_seconds':0.,
                    'enumerated_candidate_not_model_decision':True}
            cfg=replace(retimed_config(n/40,'relevel_only'),fault_open_steps=8 if seed%2==0 else 16)
            tick=time.perf_counter()
            with patch.object(l4_lift,'TwoArmLiftEnv',factory):
                row=l4_lift.run_trial(seed,scenario,config=cfg,selector=select,duration_selector=duration,
                    fixed_selector_wait_steps=19,refresh_duration_observation=True,fixed_duration_wait_steps=w,
                    account_prefix_observation_latency=first,fixed_prefix_observation_wait_steps=canonical_wait)
            if row.get('error'):raise RuntimeError((seed,scenario,w,n,row['error']))
            if first:
                canonical_wait=row['duration_selection']['prefix_observation_wait_steps']
                if row['selector_wall_seconds']>19/20:
                    raise RuntimeError('h0 acquisition exceeded its fixed physical hold; rerun with a matched larger hold')
            row.update(experiment_stage='training_wait_intervention',scope_wait_steps=19,post_prefix_wait_steps=w,
                prefix_observation_wait_steps=canonical_wait,fault_open_steps=cfg.fault_open_steps,
                stage_capture=recorded,wall_seconds=time.perf_counter()-tick)
            rows.append(row)
    return rows


def main():
    destination=ROOT/'collection/trials.jsonl';destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('collection already exists')
    jobs=[(s,q) for s in range(48,56) for q in ('left_loss','right_loss')]
    started=time.perf_counter();rows=[]
    with destination.open('w') as stream:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            for group in pool.map(collect_state,jobs):
                for row in group:rows.append(row);stream.write(json.dumps(row)+'\n')
                stream.flush()
                print(json.dumps({'seed':group[0]['seed'],'scenario':group[0]['scenario'],'episodes':len(rows),
                    'prefix_capture_wait_steps':group[0]['prefix_observation_wait_steps'],
                    'labels':[outcome(r) for r in group]}),flush=True)
    examples=[];cases=[]
    for seed,scenario in jobs:
        group=[r for r in rows if r['seed']==seed and r['scenario']==scenario]
        canonical=group[0]['stage_capture'];reference=group[0]
        boundary=reference['duration_selection']['boundary_state']['time']
        prefix=lambda r:[f for f in r['trace'] if f['time']<=boundary+1e-9]
        for row in group:
            assert all(row[k]==reference[k] for k in STATE_KEYS)
            assert prefix_frames(row)==prefix_frames(reference) and prefix(row)==prefix(reference)
            for stage in ('h0','h1','h2'):
                assert all(row['stage_capture'][stage][k]==canonical[stage][k] for k in ('state','robot','history'))
            assert row['state_edit_count_after_initialization']==row['contact_anchor_count']==row['external_wrench_at_end']==0
            assert not row.get('blocked_calls')
            assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-7
            assert abs(row['inference_wait_seconds']-(19+row['prefix_observation_wait_steps']+row['post_prefix_wait_steps'])/20)<1e-7
            keeper=1 if row['candidate']=='R_left' else 0
            label=outcome(row) if row['post_wait_evaluator']['holds'][keeper] else 'E'
            p=stage_plan(row);p['prefix_observation_hold_steps']=row['prefix_observation_wait_steps']
            examples.append({'seed':seed,'scenario':scenario,'split':'train','image':canonical['h1']['image'],
                'robot':canonical['h1']['robot'],'before_image':canonical['h0']['image'],'before_robot':canonical['h0']['robot'],
                'after_prefix_image':canonical['h2']['image'],'after_prefix_robot':canonical['h2']['robot'],
                'early_history':canonical['h1']['history'],'late_history':canonical['h2']['history'],
                'plan':p,'label':label,'whole_episode_label':outcome(row),'wait_steps':row['post_prefix_wait_steps'],
                'post_prefix_wait_steps':row['post_prefix_wait_steps'],'source':'training_wait_intervention','replay_source':'new'})
        cases.append({'seed':seed,'scenario':scenario,'fault_open_steps':reference['fault_open_steps'],
            'fault_class_valid':reference['fault_class_valid'],'prefix_capture_wait_steps':reference['prefix_observation_wait_steps'],
            'labels':[outcome(r) for r in group]})
    assert len(examples)==96
    (ROOT/'collection/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    (ROOT/'collection/report.json').write_text(json.dumps({'episodes':len(rows),'wall_seconds':time.perf_counter()-started,
        'matched_prefixes':True,'cases':cases},indent=2)+'\n')
    print(json.dumps({'complete':True,'episodes':len(rows),'wall_seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
