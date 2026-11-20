"""Paced common-prefix execution with nonblocking family forecasts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from . import l4_lift
from .duration_aligned_loop import read_response
from .duration_closed_loop import render_rgb,trial_summary
from .duration_stage_data import robot_only
from .duration_pilot import file_hash,STATE_KEYS,prefix_frames
from .l0 import _import_robosuite
from .vlm_duration_data import plan


def family_trial(seed,scenario,output,worker,baseline=False,force60=False,adapter_name='post_wait'):
    holder={};acquisition={};pending={};entry={};queries=[]
    original=l4_lift.TwoArmLiftEnv
    def factory(*args,**kwargs):
        env=original(*args,**kwargs);holder['env']=env;env.raw.has_offscreen_renderer=True
        reset=env.reset
        def warmed():
            reset();tick=time.perf_counter();render_rgb(env)
            acquisition['camera_setup_seconds']=time.perf_counter()-tick
        env.reset=warmed
        return env
    def send(message):
        worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush()
    def select(port):
        tick=time.perf_counter();rgb=render_rgb(holder['env'])
        if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid h0 RGB')
        path=output/'observations'/f'{seed}-{scenario}-{method}-h0.npz'
        np.savez_compressed(path,rgb=rgb)
        acquisition['h0']={'image':str(path.resolve()),'robot':robot_only(port),
            'sim_time':float(holder['env'].sim.data.time),'seconds':time.perf_counter()-tick}
        send({'op':'scope','image':str(path.resolve()),'robot':robot_only(port)})
        decision,_=read_response(worker)
        if baseline and decision.get('parallel_local'):
            decision={**decision,'parallel_local':False,'accepted':True,'local_relevel_steps':60}
        return decision
    method='fixed60' if baseline else 'family_stream_forced60' if force60 else 'family_control' if adapter_name=='family_control' else 'family_set'
    with ThreadPoolExecutor(max_workers=1) as reader:
        def duration(operation,port,candidate,prefix_seconds):
            if operation=='start':
                h0=acquisition['h0'];p=plan({'candidate':candidate})
                p.update(post_prefix_wait_steps=0,final_hold_steps=20)
                entry.update(before_image=h0['image'],before_robot=h0['robot'],robot=robot_only(port),plan=p,
                    entry_seconds_after_h0=float(holder['env'].sim.data.time)-h0['sim_time'])
                tick=time.perf_counter();send({'op':'family_prefill','example':entry,'adapter_name':adapter_name})
                pending['prefill']=(tick,reader.submit(read_response,worker))
                return
            if operation=='lookahead':
                example={**entry,'lookahead_robot':robot_only(port),'lookahead_previous_robot':port['previous_robot'],
                    'lookahead_delta_seconds':port['delta_seconds'],'lookahead_stage':port['stage'],
                    'lookahead_seconds_after_h0':port['seconds_after_h0'],'remaining_close_steps':port['remaining_close_steps'],
                    'prefix_observation_wait_steps':port['prefix_observation_wait_steps']}
                plans=[]
                for n in (30,40,60):
                    p=plan({'candidate':candidate,'local_steps':{'local_relevel_steps':n}})
                    p.update(scope_hold_steps=round(entry['entry_seconds_after_h0']*20),post_prefix_wait_steps=0,
                        prefix_observation_hold_steps=port['prefix_observation_wait_steps'],final_hold_steps=20)
                    plans.append(p)
                attempt=port['attempt_id'];tick=time.perf_counter()
                send({'op':'family_tail','example':example,'plans':plans,'attempt_id':attempt})
                pending[attempt]=(tick,reader.submit(read_response,worker))
                queries.append({'attempt_id':attempt,'issued_sim_time':float(holder['env'].sim.data.time),
                    'remaining_close_steps':port['remaining_close_steps']})
                return
            attempt=port['attempt_id'];pair=pending.get(attempt)
            if pair is None or not pair[1].done():
                return {'candidate':candidate,'accepted':True,'local_relevel_steps':60,'baseline_fallback':True,
                    'forecast_ready':False,'attempt_id':attempt,'reason':'forecast_not_ready_at_action_boundary',
                    'pipeline_wall_seconds':None if pair is None else time.perf_counter()-pair[0]}
            answer,arrived=pair[1].result()
            result={**answer,'candidate':candidate,'forecast_ready':True,
                'pipeline_wall_seconds':arrived-pair[0]}
            if force60:
                result.update(accepted=True,local_relevel_steps=60,baseline_fallback=True,forced60=True)
            elif result['accepted'] and result['local_relevel_steps']<60:
                extra=(port.get('prefix_observation_wait_steps',0)+port.get('controller_compute_wait_steps',0))/20
                if extra>=(60-result['local_relevel_steps'])/20:
                    result.update(local_relevel_steps=60,baseline_fallback=True,reason='extra_blocking_exceeds_motion_saving')
            return result
        tick=time.perf_counter()
        with patch.object(l4_lift,'TwoArmLiftEnv',factory):
            row=l4_lift.run_trial(seed,scenario,selector=select,account_selector_latency=True,
                duration_selector=None if baseline else duration,refresh_duration_observation=True,
                duration_lookahead_steps=None if baseline else 10,pace_recovery=True,
                account_prefix_observation_latency=not baseline)
        finished=time.perf_counter();timings=[]
        # Draining does not delay a robot decision: the episode and final hold have finished.
        for key,(started,future) in pending.items():
            answer,arrived=future.result()
            timings.append({'kind':'prefill' if key=='prefill' else 'tail','attempt_id':None if key=='prefill' else key,
                'pipeline_wall_seconds':arrived-started,'worker_seconds':answer['model_pipeline_seconds']})
        if pending:
            send({'op':'family_release'});read_response(worker)
        row.update(method=method,acquisition=acquisition,async_timing=timings,query_schedule=queries,
            episode_wall_seconds=time.perf_counter()-tick,post_episode_drain_seconds=time.perf_counter()-finished)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--control-adapter')
    parser.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    parser.add_argument('--model-path',default='/home/lc/AuditEmbodied/outputs/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--gpu-python',default='/opt/miniconda3/bin/python')
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--seeds',nargs='+',type=int,default=[40])
    parser.add_argument('--engineering',action='store_true')
    parser.add_argument('--force60',action='store_true')
    parser.add_argument('--warm-data',default='outputs/vlm-duration-family-stream-20261109/dataset/train.jsonl')
    args=parser.parse_args();output=Path(args.output_dir).resolve()
    if (output/'trials.jsonl').exists():raise ValueError('output trial file already exists')
    (output/'observations').mkdir(parents=True,exist_ok=True)
    point=json.loads(Path(args.working_point).read_text());frozen=[Path(args.working_point).resolve()]
    for folder in (point['start_adapter'],point['outcome_adapter'],args.adapter):
        frozen.extend(Path(folder).resolve()/n for n in ('adapter_config.json','adapter_model.safetensors'))
    if args.control_adapter:frozen.extend(Path(args.control_adapter).resolve()/n for n in ('adapter_config.json','adapter_model.safetensors'))
    sources={}
    (output/'source').mkdir(exist_ok=True)
    for name in ('l4_lift.py','vlm_duration_policy.py','vlm_duration_family.py','duration_family_loop.py'):
        path=Path(__file__).resolve().parent/name
        (output/'source'/name).write_bytes(path.read_bytes());sources[str(path)]=file_hash(path)
    protocol={'seeds':args.seeds,'engineering_only':args.engineering,'force60':args.force60,
        'threshold':point['threshold'],'lookahead_steps':10,'post_prefix_wait_steps':0,'pace_recovery':True,
        'frozen':{str(p):file_hash(p) for p in frozen},'source_at_start':sources,
        'scope':'Baseline and learned use the same frozen scope head. Only existing h0 RGB; no h1/h2 capture. Inference during committed prefix, late reply falls back60. Model admission and baseline continuation are separate.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    rows=[]
    with (output/'worker.log').open('w') as log:
        command=[args.gpu_python,'-m','auditembodied.vlm_duration_policy',
            '--working-point',args.working_point,'--model-path',args.model_path,
            '--post-wait-adapter',args.adapter,'--post-wait-mode','family_stream']
        if args.control_adapter:command.extend(['--family-control-adapter',args.control_adapter])
        worker=subprocess.Popen(command,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        try:
            line=worker.stdout.readline()
            if not line:
                worker.wait(timeout=10)
                raise RuntimeError(f'model worker exited with code {worker.returncode}; see {output / "worker.log"}')
            ready=json.loads(line)
            if not ready.get('ready'):raise RuntimeError(ready)
            warm=json.loads(Path(args.warm_data).read_text().splitlines()[0])
            warm_started=time.perf_counter()
            for adapter in (['post_wait','family_control'] if args.control_adapter else ['post_wait']):
                for message in ({'op':'family_prefill','example':warm,'adapter_name':adapter},
                    {'op':'family_tail','example':warm,'attempt_id':0,
                        'plans':[{**warm['plan'],'local_relevel_steps':n} for n in (30,40,60)]},
                    {'op':'family_release'}):
                    worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush();read_response(worker)
            ready['family_training_observation_warmup_seconds']=time.perf_counter()-warm_started
            _import_robosuite(True)
            with (output/'trials.jsonl').open('w') as stream:
                for index,(seed,scenario) in enumerate((s,q) for s in args.seeds for q in ('left_loss','right_loss')):
                    methods=[(True,'post_wait'),(False,'post_wait')]
                    if args.control_adapter:
                        methods.append((False,'family_control'))
                        shift=(index+index//2)%3;order=methods[shift:]+methods[:shift]
                    else:order=methods if (index+index//2)%2==0 else list(reversed(methods))
                    pair=[]
                    for baseline,adapter in order:
                        row=family_trial(seed,scenario,output,worker,baseline,args.force60,adapter)
                        rows.append(row);pair.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                        print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation',
                            'inference_wait_seconds','total_post_fault_seconds','error')}),flush=True)
                    for other in pair[1:]:assert all(pair[0][k]==other[k] for k in STATE_KEYS) and prefix_frames(pair[0])==prefix_frames(other)
        finally:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                except BrokenPipeError:pass
                worker.wait(timeout=30)
    for r in rows:
        assert not r.get('error') and not r.get('blocked_calls'),r.get('error')
        assert r['state_edit_count_after_initialization']==r['contact_anchor_count']==r['external_wrench_at_end']==0
        assert abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-7
        assert not any(f['stage'] in ('duration_budget_wait','duration_deadline_wait') for f in r['trace'])
    assert all(file_hash(p)==digest for p,digest in protocol['frozen'].items())
    methods={name:trial_summary([r for r in rows if r['method']==name]) for name in {r['method'] for r in rows}}
    for name,summary in methods.items():
        group=[r for r in rows if r['method']==name]
        summary.update(abstentions=sum(bool(r.get('abstained')) or r['candidate']=='abstain' for r in group),
            timely=sum(r.get('duration_selection',{}).get('forecast_ready',False) for r in group),
            duration_decisions=sum('duration_selection' in r for r in group),
            baseline_fallbacks=sum(r.get('duration_selection',{}).get('baseline_fallback',False) for r in group),
            model_admissions=sum(bool(r.get('duration_selection',{}).get('decision',{}).get('model_admitted')) and
                not r.get('duration_selection',{}).get('baseline_fallback',False) for r in group),
            explicit_rejections=sum(bool(r.get('duration_selection',{}).get('decision',{}).get('explicit_local_rejection')) and
                not r.get('duration_selection',{}).get('baseline_fallback',False) for r in group),
            deadline_misses=sum(r.get('duration_selection',{}).get('deadline_missed',False) for r in group),
            model_admitted_safe_successes=sum(bool(r.get('duration_selection',{}).get('decision',{}).get('model_admitted')) and
                not r.get('duration_selection',{}).get('baseline_fallback',False) and r['success'] and
                not r['protection_violation'] for r in group),
            model_admitted_process_violations=sum(bool(r.get('duration_selection',{}).get('decision',{}).get('model_admitted')) and
                not r.get('duration_selection',{}).get('baseline_fallback',False) and r['protection_violation'] for r in group),
            duration_counts=dict(Counter(str(r.get('local_steps',{}).get('local_relevel_steps')) for r in group)),
            extra_control_wait_steps=sum(r.get('controller_compute_wait_steps',0) for r in group))
    changed_sources=[path for path,digest in sources.items() if file_hash(Path(path))!=digest]
    report={'engineering_only':args.engineering,'episodes':len(rows),'methods':methods,'setup':ready,
        'frozen_unchanged':True,'all_waits_audited':True,'source_files_changed_after_import':changed_sources}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
