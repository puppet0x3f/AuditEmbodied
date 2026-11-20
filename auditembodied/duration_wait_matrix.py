"""Controlled 19-step hold, matched post-wait RGB, and frozen duration assessment."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from . import l4_lift
from .duration_acquire import before_relevel
from .duration_aligned_loop import read_response
from .duration_closed_loop import render_rgb
from .duration_pilot import SOURCE_FILES,STATE_KEYS,file_hash,prefix_frames,retimed_config
from .l0 import _import_robosuite
from .vlm_duration_data import outcome,plan,read_rows


SPLITS={'train':list(range(8,16)),'development':list(range(20,24))}
ZERO_PHYSICS=('outputs/l4-duration-newstates-20261108/trials.jsonl',
              'outputs/l4-duration-relevel-20261108/confirmation-trials.jsonl')
ZERO_PREDICTIONS=('outputs/vlm-duration-lora-riskreplay512-20261108/train-after-predictions.jsonl',
                  'outputs/vlm-duration-lora-riskreplay512-20261108/after-predictions.jsonl')


def continuation_outcome(row):
    """Labels start at h1; whole-episode guards remain separately reported."""
    keeper=1 if row['candidate']=='R_left' else 0
    before=row['pre_skill_protection'];after=row['protection']
    flight=maximum=0
    for f in row['trace']:
        if f['time']<=row['post_wait_state']['time']+1e-9:continue
        if f['phase'] not in ('recovery','final_hold'):continue
        truth=f['evaluator']
        flight=flight+1 if not (any(truth['holds']) or truth['table_contact']) else 0
        maximum=max(maximum,flight)
    guards={'max_unsupported_seconds':maximum/20,
        'keeper_grasp_lost_steps':after['keeper_grasp_lost_steps']-before['keeper_grasp_lost_steps'],
        'deep_interarm_contact_steps':after['deep_interarm_contact_steps']-before['deep_interarm_contact_steps']}
    violated=maximum>2 or guards['keeper_grasp_lost_steps']>0 or guards['deep_interarm_contact_steps']>0
    label=('E' if not row['post_wait_evaluator']['holds'][keeper] else
           ('B' if violated else 'A') if row['success'] else ('D' if violated else 'C'))
    return label,guards


def collect_state(job):
    split,seed,scenario,folder,wait_steps=job[:5]
    capture_before=bool(job[5]) if len(job)>5 else False
    _import_robosuite(True)
    candidate='R_left' if scenario=='left_loss' else 'R_right'
    rows=[]
    for n in (30,40,60):
        recorded={};holder={};original=l4_lift.TwoArmLiftEnv
        def factory(*args,**kwargs):
            env=original(*args,**kwargs);env.raw.has_offscreen_renderer=True;holder['env']=env
            reset=env.reset
            def warm_reset():
                reset();tick=time.perf_counter();render_rgb(env)
                recorded['camera_setup_seconds']=time.perf_counter()-tick
            env.reset=warm_reset
            return env
        def duration(operation,port,role,prefix):
            if operation=='start':
                if n==30:
                    env=holder['env'];state=l4_lift.snapshot(env);rgb=render_rgb(env)
                    if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid RGB')
                    image=Path(folder)/f'{seed}-{scenario}.npz';np.savez_compressed(image,rgb=rgb)
                    recorded.update(observation_file=str(image.resolve()),state=state,robot=port,
                                    image_std=[float(x.std()) for x in rgb])
                return
            return {'candidate':role,'accepted':True,'local_relevel_steps':n,'pipeline_wall_seconds':0.,
                    'enumerated_candidate_not_model_decision':True}
        def select(port):
            if capture_before and n==30:
                env=holder['env'];state=l4_lift.snapshot(env);rgb=render_rgb(env)
                if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid pre-wait RGB')
                image=Path(folder)/f'{seed}-{scenario}-before.npz';np.savez_compressed(image,rgb=rgb)
                recorded.update(before_image=str(image.resolve()),before_state=state,before_robot=port)
            return {'candidate':candidate,'parallel_local':True}
        tick=time.perf_counter()
        with patch.object(l4_lift,'TwoArmLiftEnv',factory if n==30 else original):
            row=l4_lift.run_trial(seed,scenario,config=retimed_config(n/40,'relevel_only'),
                selector=select,
                duration_selector=duration,fixed_selector_wait_steps=wait_steps)
        row.update(split=split,experiment_stage='fixed_wait_duration_diagnostic',wait_steps=wait_steps,
                   time_scale=n/40,retiming_mode='relevel_only',wall_seconds=time.perf_counter()-tick)
        if recorded:row['capture']=recorded
        rows.append(row)
    return rows


def zero_rows():
    result={}
    for source in ZERO_PHYSICS:
        for r in read_rows(source):
            if r['candidate'] not in ('R_left','R_right'):continue
            if r['seed'] not in sum(SPLITS.values(),[]):continue
            result[(r['seed'],r['scenario'],r['local_steps']['local_relevel_steps'])]=r
    return result


def summarize_cases(cases):
    return {'states':len(cases),'states_with_safe_candidate':sum(bool(c['safe_relevel_steps']) for c in cases),
        'minimal_safe_relevel_counts':dict(Counter(str(c['minimal_safe_relevel']) for c in cases)),
        'short30_unsafe_longer_safe_states':sum(c['labels_by_relevel']['30']!='A' and
            any(c['labels_by_relevel'][str(n)]=='A' for n in (40,60)) for c in cases),
        'unsafe40_safe60_states':sum(c['labels_by_relevel']['40']!='A' and c['labels_by_relevel']['60']=='A' for c in cases),
        'timing_safety_contrast_states':sum(0<len(c['safe_relevel_steps'])<3 for c in cases),
        'by_relevel':{str(n):{'terminal_successes':sum(c['success_by_relevel'][str(n)] for c in cases),
            'safe_successes':sum(c['labels_by_relevel'][str(n)]=='A' for c in cases),
            'process_violations':sum(c['protection_by_relevel'][str(n)] for c in cases),
            'mean_total_seconds':float(np.mean([c['total_seconds_by_relevel'][str(n)] for c in cases]))}
            for n in (30,40,60)}}


def analyze_physics(rows,protocol):
    zeros=zero_rows();groups={};errors=[];cases=[];examples={s:[] for s in SPLITS}
    for r in rows:groups.setdefault((r['seed'],r['scenario']),[]).append(r)
    for key,group in sorted(groups.items()):
        ref=group[0];capture=ref.get('capture',{})
        if sorted(r.get('local_steps',{}).get('local_relevel_steps',-1) for r in group)!=[30,40,60]:
            errors.append([*key,'duration_set']);continue
        for r in group:
            n=r['local_steps']['local_relevel_steps'];zero=zeros[key[0],key[1],n]
            if r.get('error') or not r.get('fault_class_valid') or r.get('blocked_calls'):
                errors.append([*key,n,'physical_execution',r.get('error')]);continue
            for field in STATE_KEYS:
                if r.get(field)!=ref.get(field) or r.get(field)!=zero.get(field):errors.append([*key,n,field])
            if prefix_frames(r)!=prefix_frames(zero):errors.append([*key,n,'no_wait_fault_prefix'])
            if before_relevel(r)!=before_relevel(ref):errors.append([*key,n,'common_recovery_prefix'])
            if r['post_wait_state']!=ref['post_wait_state'] or capture.get('state')!=r['post_wait_state']:
                errors.append([*key,n,'post_wait_observation'])
            if r['duration_selection']['input']!=capture.get('robot'):errors.append([*key,n,'measured_robot_input'])
            waits=[f for f in r['trace'] if f['stage']=='inference_wait']
            last_fault=prefix_frames(r)[-1]['robot']['action']
            if (len(waits)!=protocol['wait_steps'] or any(f['phase']!='recovery' for f in waits) or
                any([f['robot']['action'][i] for i in (6,13)]!=[last_fault[i] for i in (6,13)] for f in waits) or
                abs(r['inference_wait_seconds']-protocol['wait_steps']/20)>1e-8 or
                abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])>1e-8):
                errors.append([*key,n,'wait_or_cost_ledger'])
            if r.get('state_edit_count_after_initialization') or r.get('contact_anchor_count') or r.get('external_wrench_at_end'):
                errors.append([*key,n,'physical_state_edit'])
            label,guards=continuation_outcome(r)
            r['continuation_label']=label;r['continuation_protection']=guards
            examples[r['split']].append({'split':r['split'],'seed':r['seed'],'scenario':r['scenario'],'image':capture.get('observation_file'),
                'robot':capture.get('robot'),'plan':plan(r),'label':label,'whole_episode_label':outcome(r),
                'preceding_wait_steps':protocol['wait_steps']})
        if any('continuation_label' not in r for r in group):continue
        by_n={str(r['local_steps']['local_relevel_steps']):r for r in group}
        labels={n:outcome(r) for n,r in by_n.items()};safe=[int(n) for n,v in labels.items() if v=='A']
        cases.append({'seed':key[0],'scenario':key[1],'split':ref['split'],'labels_by_relevel':labels,
            'continuation_labels_by_relevel':{n:r['continuation_label'] for n,r in by_n.items()},
            'zero_wait_labels_by_relevel':{n:outcome(zeros[key[0],key[1],int(n)]) for n in by_n},
            'safe_relevel_steps':safe,'minimal_safe_relevel':min(safe) if safe else None,
            'success_by_relevel':{n:r['success'] for n,r in by_n.items()},
            'protection_by_relevel':{n:r['protection_violation'] for n,r in by_n.items()},
            'total_seconds_by_relevel':{n:r['total_post_fault_seconds'] for n,r in by_n.items()},
            'wait_protection':ref['pre_skill_protection'],
            'keeper_lost_steps_by_relevel':{n:r['protection']['keeper_grasp_lost_steps'] for n,r in by_n.items()}})
    if len(rows)!=protocol['planned_physical_episodes'] or len(cases)!=24:errors.append(['incomplete_campaign'])
    frozen=all(file_hash(p)==h for p,h in protocol['frozen_artifacts_sha256'].items())
    sources=all(file_hash(p)==h for p,h in protocol['source_sha256'].items())
    references=all(file_hash(p)==h for p,h in protocol['reference_sha256'].items())
    report={'valid':not errors and frozen and sources and references,'physical_episodes':len(rows),'physical_states':len(cases),
        'errors':errors,'model_and_threshold_unchanged':frozen,'sources_unchanged':sources,'references_unchanged':references,
        'by_split':{s:summarize_cases([c for c in cases if c['split']==s]) for s in SPLITS},'all':summarize_cases(cases),
        'waiting_guard_violations_states':sum(c['wait_protection']['max_unsupported_seconds']>.1 or
            c['wait_protection']['keeper_grasp_lost_steps']>0 or c['wait_protection']['deep_interarm_contact_steps']>0 for c in cases),
        'cases':cases,'interpretation':'Controlled 0.95-second hold followed by forced candidates with known repair role. Not an online policy test. Whole-episode outcomes include the hold; continuation labels start at the refreshed h1 observation. Offline scoring time is recorded separately and does not add a second physical intervention.'}
    return report,examples


def score_dataset(output,args):
    examples=[r for s in SPLITS for r in read_rows(output/'dataset'/f'{s}.jsonl')]
    groups={}
    for r in examples:groups.setdefault((r['seed'],r['scenario']),[]).append(r)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    started=time.perf_counter();predictions=[]
    with (output/'model-worker.log').open('w') as err:
        worker=subprocess.Popen([args.gpu_python,'-m','auditembodied.vlm_duration_policy','--working-point',args.working_point,
            '--model-path',args.model_path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env=env)
        try:
            setup=json.loads(worker.stdout.readline())
            if not setup.get('ready'):raise RuntimeError(setup)
            print(json.dumps({'model_ready':setup}),flush=True)
            with (output/'predictions.jsonl').open('w') as f:
                for key,group in sorted(groups.items()):
                    first=group[0];candidate='R_'+first['plan']['repair_arm']
                    request={'op':'score_local_family','image':first['image'],'robot':first['robot'],'candidate':candidate}
                    tick=time.perf_counter();worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
                    result,_=read_response(worker);elapsed=time.perf_counter()-tick
                    by_n={r['plan']['local_relevel_steps']:r for r in group}
                    for record in result['scored_candidates']:
                        n=record['plan']['local_relevel_steps'];example=by_n[n]
                        if record['plan']!=example['plan']:raise ValueError('scored plan differs from executed plan')
                        row={**example,**record,'offline_family_pipeline_seconds':elapsed}
                        predictions.append(row);f.write(json.dumps(row)+'\n')
                    f.flush()
                    print(json.dumps({'scored_seed':key[0],'scenario':key[1],'q_valid':result['start_valid_score'],
                        'admitted':[r['plan']['local_relevel_steps'] for r in result['scored_candidates'] if r['accepted']]}),flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
    return predictions,{'model_setup':setup,'offline_scoring_wall_seconds':time.perf_counter()-started}


def ranking_summary(predictions,threshold):
    groups={};pairs=correct=ties=0;contrasts=[]
    for r in predictions:groups.setdefault((r['seed'],r['scenario']),[]).append(r)
    selected=[]
    for key,group in sorted(groups.items()):
        admitted=sorted((r for r in group if r['joint_A_score']>=threshold),key=lambda r:r['plan']['local_relevel_steps'])
        selected.append(admitted[0] if admitted else None)
        for a in group:
            for b in group:
                if a['label']!='A' or b['label']=='A':continue
                diff=a['joint_A_score']-b['joint_A_score'];pairs+=1;correct+=diff>0;ties+=diff==0
                contrasts.append({'seed':key[0],'scenario':key[1],'safe_steps':a['plan']['local_relevel_steps'],
                    'unsafe_steps':b['plan']['local_relevel_steps'],'unsafe_label':b['label'],'score_margin':diff})
    return {'candidate_examples':len(predictions),'states':len(groups),
        'continuation_class_accuracy':sum(r['prediction']==r['label'] for r in predictions)/len(predictions),
        'accepted_candidates':sum(r['joint_A_score']>=threshold for r in predictions),
        'accepted_true_A_continuations':sum(r['joint_A_score']>=threshold and r['label']=='A' for r in predictions),
        'false_accepts_continuation_by_class':dict(Counter(r['label'] for r in predictions if r['joint_A_score']>=threshold and r['label']!='A')),
        'false_accepts_whole_episode':sum(r['joint_A_score']>=threshold and r['whole_episode_label']!='A' for r in predictions),
        'safe_vs_unsafe_pairs':pairs,'correctly_ranked_pairs':correct,'tied_pairs':ties,'paired_contrasts':contrasts,
        'offline_selected_states':sum(r is not None for r in selected),
        'offline_selected_safe_whole_states':sum(r is not None and r['whole_episode_label']=='A' for r in selected),
        'offline_selected_safe_continuation_states':sum(r is not None and r['label']=='A' for r in selected),
        'offline_selected_durations':dict(Counter(str(r['plan']['local_relevel_steps']) if r else 'abstain' for r in selected)),
        'scope':'Offline cheapest-admitted duration with supplied correct role; no online success claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    parser.add_argument('--model-path',default='/home/lc/AI4Chemical/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--gpu-python',default='/opt/miniconda3/bin/python');parser.add_argument('--gpu',default='2')
    args=parser.parse_args();output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('experiment exists; do not overwrite')
    (output/'observations').mkdir();(output/'dataset').mkdir()
    point=json.loads(Path(args.working_point).read_text());frozen=[Path(args.working_point).resolve()]
    for key in ('start_adapter','outcome_adapter'):
        frozen.extend(Path(point[key])/name for name in ('adapter_model.safetensors','adapter_config.json'))
    source=Path(__file__).resolve().parent
    protocol={'splits':SPLITS,'wait_steps':19,'wait_seconds':.95,'durations':[30,40,60],
        'fixed_approach_close':[16,22],'planned_physical_episodes':72,'threshold':point['threshold'],
        'frozen_artifacts_sha256':{str(p):file_hash(p) for p in frozen},
        'source_sha256':{str(source/name):file_hash(source/name) for name in (*SOURCE_FILES,'duration_wait_matrix.py',
            'duration_closed_loop.py','vlm_duration_policy.py','vlm_duration_lora.py')},
        'reference_sha256':{str(Path(p).resolve()):file_hash(p) for p in (*ZERO_PHYSICS,*ZERO_PREDICTIONS)},
        'physics_intervention':'Exactly 19 physical hold steps with last fault targets/gripper commands, then forced 30/40/60-step local relevel. RGB captured at the post-wait pre-skill h1, once per state. All branches must match through the first relevel boundary.',
        'assessment':'Frozen start and outcome adapters, same token score and threshold. Score all three durations, including candidates rejected by the start score. Inference is offline, not a second online hold.',
        'label_scope':'Store whole-episode guard outcomes and separately reconstruct future-only continuation labels from h1. All waiting remains in whole-episode guards and physical cost.',
        'data_policy':'Only existing training seeds 8-15 and development seeds 20-23; independent test seeds 1000-1003 and 2000-2003 excluded. No training or threshold tuning.',
        'decision':'First establish whether safe post-wait alternatives and state-dependent timing contrasts exist; report all failures. Do not start LoRA training during this diagnostic.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[(split,seed,scenario,str((output/'observations').resolve()),19) for split,seeds in SPLITS.items()
          for seed in seeds for scenario in ('left_loss','right_loss')]
    started=time.perf_counter();rows=[]
    with (output/'trials.jsonl').open('w') as f:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for group in pool.map(collect_state,jobs):
                for r in group:rows.append(r);f.write(json.dumps(r)+'\n')
                f.flush()
                print(json.dumps({'seed':group[0]['seed'],'scenario':group[0]['scenario'],'completed':len(rows),
                    'whole_labels':[outcome(r) if not r.get('error') else r['error'] for r in group]}),flush=True)
    physical,examples=analyze_physics(rows,protocol)
    physical['physical_campaign_wall_seconds']=time.perf_counter()-started
    (output/'physical-report.json').write_text(json.dumps(physical,indent=2)+'\n')
    if not physical['valid']:raise ValueError('physical pairing or accounting failed; see physical-report.json')
    for split,group in examples.items():
        (output/'dataset'/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in group))
    predictions,assessment=score_dataset(output,args)
    zero_predictions=[r for p in ZERO_PREDICTIONS for r in read_rows(p) if r['plan']['kind']=='local'
        and r['seed'] in sum(SPLITS.values(),[]) and r['plan']['repair_arm']==('left' if r['scenario']=='left_loss' else 'right')
        and r['plan']['local_approach_steps']==16 and r['plan']['local_close_steps']==22]
    for r in zero_predictions:
        r.update(joint_A_score=r['class_probabilities'][0],whole_episode_label=r['label'])
    physical['frozen_assessment']=assessment
    physical['ranking_by_split']={split:ranking_summary([r for r in predictions if r['seed'] in seeds],point['threshold']) for split,seeds in SPLITS.items()}
    physical['zero_wait_ranking_by_split']={split:ranking_summary([r for r in zero_predictions if r['seed'] in seeds],point['threshold']) for split,seeds in SPLITS.items()}
    physical['model_and_threshold_unchanged']=all(file_hash(p)==h for p,h in protocol['frozen_artifacts_sha256'].items())
    physical['valid']=physical['valid'] and physical['model_and_threshold_unchanged'] and len(predictions)==72
    physical['campaign_wall_seconds']=time.perf_counter()-started
    (output/'report.json').write_text(json.dumps(physical,indent=2)+'\n')
    with (output/'cases.csv').open('w',newline='') as f:
        fields=['seed','scenario','split','relevel_steps','label','whole_episode_label','joint_A_score','prediction','accepted']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in predictions:writer.writerow({k:(r['plan']['local_relevel_steps'] if k=='relevel_steps' else r[k]) for k in fields})
    print(json.dumps({'valid':physical['valid'],'physical':physical['by_split'],'ranking':physical['ranking_by_split']}),flush=True)


if __name__=='__main__':main()
