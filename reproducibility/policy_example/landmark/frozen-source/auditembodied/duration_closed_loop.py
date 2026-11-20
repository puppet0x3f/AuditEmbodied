"""Independent frozen-policy trials with physically audited inference waiting."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from . import l4_lift
from .duration_pilot import SOURCE_FILES,STATE_KEYS,file_hash,prefix_frames,retimed_config
from .l0 import CAMERAS,_import_robosuite
from .vlm_duration_policy import candidate_bank


def render_rgb(env):
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    before=l4_lift.snapshot(env)
    rgb=np.stack([env.sim.render(camera_name=c,width=64,height=64)[::convention].copy() for c in CAMERAS])
    if before!=l4_lift.snapshot(env):raise ValueError('camera acquisition changed physical state')
    return rgb


def policy_trial(seed,scenario,output,worker):
    original=l4_lift.TwoArmLiftEnv
    holder={};acquisition={}
    def factory(*args,**kwargs):
        env=original(*args,**kwargs)
        env.raw.has_offscreen_renderer=True
        holder['env']=env
        original_reset=env.reset
        def reset_and_warm_camera():
            original_reset()
            tick=time.perf_counter();render_rgb(env)
            acquisition['camera_setup_seconds']=time.perf_counter()-tick
        env.reset=reset_and_warm_camera
        return env
    def select(port):
        env=holder['env']
        acquisition['observation_state']=l4_lift.snapshot(env)
        rgb=render_rgb(env)
        if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid rendered observation')
        image=output/'observations'/f'{seed}-{scenario}.npz'
        np.savez_compressed(image,rgb=rgb)
        worker.stdin.write(json.dumps({'image':str(image.resolve()),'robot':port})+'\n');worker.stdin.flush()
        line=worker.stdout.readline()
        if not line:raise RuntimeError('model worker exited before decision')
        response=json.loads(line)
        if 'error' in response:raise RuntimeError(response['error'])
        acquisition['observation_file']=str(image.resolve())
        return response
    tick=time.perf_counter()
    with patch.object(l4_lift,'TwoArmLiftEnv',factory):
        row=l4_lift.run_trial(seed,scenario,selector=select,account_selector_latency=True)
    row.update(method='vlm_with_wait',episode_wall_seconds=time.perf_counter()-tick,acquisition=acquisition)
    if 'selection' in row:row['selection']['rgb_frames']=4
    return row


def trial_summary(rows):
    total=[r['total_post_fault_seconds'] for r in rows if 'total_post_fault_seconds' in r]
    mean=lambda key:float(np.mean([r[key] for r in rows if key in r])) if any(key in r for r in rows) else None
    return {'episodes':len(rows),'valid_faults':sum(r.get('fault_class_valid',False) for r in rows),
        'successes':sum(r['success'] for r in rows),
        'safe_successes':sum(r['success'] and not r['protection_violation'] for r in rows),
        'process_violations':sum(r['protection_violation'] for r in rows),
        'abstentions':sum(r['candidate']=='abstain' for r in rows),
        'mean_total_post_fault_seconds':float(np.mean(total)) if total else None,
        'mean_selector_wall_seconds':mean('selector_wall_seconds'),'mean_executed_wait_seconds':mean('inference_wait_seconds'),
        'mean_skill_seconds':mean('skill_seconds'),'mean_final_hold_seconds':mean('hold_seconds'),
        'mean_episode_simulated_seconds':mean('episode_simulated_seconds'),'mean_episode_wall_seconds':mean('episode_wall_seconds'),
        'selected_candidates':dict(Counter(r['candidate'] for r in rows)),
        'selected_local_relevel_steps':dict(Counter(str(r['local_steps']['local_relevel_steps']) for r in rows if 'local_steps' in r)),
        'failure_reasons':dict(Counter(r['failure_reason'] for r in rows if not r['success']))}


def analyze(rows,protocol,worker_setup,wall_seconds):
    paired={};mismatches=[];wait_errors=[];cases=[]
    for r in rows:paired.setdefault((r['seed'],r['scenario']),{})[r['method']]=r
    for key,pair in paired.items():
        a,b=pair['oracle_role_fixed60'],pair['vlm_with_wait']
        for field in STATE_KEYS:
            if a.get(field)!=b.get(field):mismatches.append([*key,field])
        if prefix_frames(a)!=prefix_frames(b):mismatches.append([*key,'entire_initialization_and_fault_trace'])
        for row in pair.values():
            wait=[f for f in row['trace'] if f['stage']=='inference_wait']
            delta=row.get('inference_wait_seconds',0.)
            actual=row.get('selector_wall_seconds',0.) if row['method']=='vlm_with_wait' else 0.
            if len(wait)!=row.get('inference_wait_steps',0) or delta+1e-9<actual or delta-actual>.05+1e-9:
                wait_errors.append([*key,row['method'],'latency_conversion'])
            if any(f['phase']!='recovery' for f in wait):wait_errors.append([*key,row['method'],'unaudited_wait'])
            if 'total_post_fault_seconds' in row and abs(row['total_post_fault_seconds']-row['recovery_seconds']-row['hold_seconds'])>1e-8:
                wait_errors.append([*key,row['method'],'time_accounting'])
        cases.append({'seed':key[0],'scenario':key[1],'policy_candidate':b['candidate'],
            'policy_relevel_steps':b.get('local_steps',{}).get('local_relevel_steps'),
            'policy_success':b['success'],'policy_process_violation':b['protection_violation'],
            'policy_selector_wall_seconds':b.get('selector_wall_seconds'),
            'policy_wait_seconds':b.get('inference_wait_seconds'),'policy_skill_seconds':b.get('skill_seconds'),
            'policy_total_seconds':b.get('total_post_fault_seconds'),'baseline_success':a['success'],
            'baseline_process_violation':a['protection_violation'],'baseline_total_seconds':a.get('total_post_fault_seconds'),
            'policy_wait_keeper_lost_steps':sum(not f['evaluator']['holds'][1 if b['candidate']=='R_left' else 0]
                for f in b['trace'] if f['stage']=='inference_wait') if b['candidate'] in ('R_left','R_right') else None})
    frozen=all(file_hash(p)==digest for p,digest in protocol['frozen_artifacts_sha256'].items())
    execution_errors=[{'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],'error':r.get('error')}
                      for r in rows if r.get('error') or r.get('blocked_calls')]
    return {'seeds':protocol['seeds'],'physical_states':len(paired),'physical_episodes':len(rows),
        'valid_execution':not mismatches and not wait_errors and frozen and not execution_errors,
        'prefix_mismatches':mismatches,'wait_accounting_errors':wait_errors,'frozen_artifacts_unchanged':frozen,
        'execution_errors':execution_errors,'methods':{name:trial_summary([r for r in rows if r['method']==name])
            for name in ['vlm_with_wait','oracle_role_fixed60']},'cases':cases,'model_setup':worker_setup,
        'campaign_wall_seconds':wall_seconds,
        'time_definition':'Post-fault simulation time = ceil(measured decision pipeline latency / 0.05)*0.05 + executed recovery skill + final 1-second hold. Whole-episode simulation and program wall times are also reported.',
        'scope':'Frozen, one-shot high-level VLM selection with feedback skill execution; inference delay replay is audited physically, not paused away. Fixed60 baseline receives the correct repair role and pays no VLM latency.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    parser.add_argument('--model-path',default='/home/lc/AI4Chemical/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--gpu-python',default='/opt/miniconda3/bin/python')
    parser.add_argument('--gpu',default='2')
    parser.add_argument('--seeds',type=int,nargs='+',default=[1000,1001,1002,1003])
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('test run exists; do not overwrite')
    (output/'observations').mkdir()
    point=json.loads(Path(args.working_point).read_text())
    frozen=[Path(args.working_point).resolve()]
    for key in ['start_adapter','outcome_adapter']:
        frozen.extend(Path(point[key])/name for name in ['adapter_model.safetensors','adapter_config.json'])
    protocol={'seeds':args.seeds,'scenarios':['left_loss','right_loss'],'threshold':point['threshold'],
        'planned_physical_episodes':len(args.seeds)*4,'candidate_bank':candidate_bank(),
        'model_path':args.model_path,'working_point':str(Path(args.working_point).resolve()),
        'frozen_artifacts_sha256':{str(p):file_hash(p) for p in frozen},
        'physics_source_sha256':{str(Path(__file__).resolve().parent/name):file_hash(Path(__file__).resolve().parent/name) for name in SOURCE_FILES},
        'latency_policy':'Hold last fault motion targets and gripper commands for ceil(measured sensor+IPC+model latency / 0.05) steps, then execute the stale-observation decision. Audit all waiting steps.',
        'setup_policy':'Model load/warmup is once before episodes; camera initialization is before nominal initialization. Both are recorded separately. No learning or threshold tuning.',
        'baseline':'Oracle repair role, fixed relevel 60, no model/decision latency',
        'model_decisions_receive':['RGB','robot measured positions/fingers','registered robot reference positions'],
        'test_scope':'small independent simulated closed-loop pilot, not a population safety guarantee'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    started=time.perf_counter();rows=[]
    with (output/'model-worker.log').open('w') as err:
        worker=subprocess.Popen([args.gpu_python,'-m','auditembodied.vlm_duration_policy','--working-point',args.working_point,
            '--model-path',args.model_path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env=env)
        try:
            ready=worker.stdout.readline()
            if not ready:raise RuntimeError('model worker failed; see model-worker.log')
            setup=json.loads(ready)
            if not setup.get('ready'):raise RuntimeError(setup)
            print(json.dumps({'model_ready':setup}),flush=True)
            _import_robosuite(True)
            with (output/'trials.jsonl').open('w') as f:
                for seed in args.seeds:
                    for scenario in ['left_loss','right_loss']:
                        tick=time.perf_counter()
                        fixed=l4_lift.run_trial(seed,scenario,'R_left' if scenario=='left_loss' else 'R_right',retimed_config(1.5,'relevel_only'))
                        fixed.update(method='oracle_role_fixed60',episode_wall_seconds=time.perf_counter()-tick)
                        policy=policy_trial(seed,scenario,output,worker)
                        for r in [fixed,policy]:rows.append(r);f.write(json.dumps(r)+'\n')
                        f.flush()
                        print(json.dumps({'seed':seed,'scenario':scenario,'selected':policy['candidate'],
                            'relevel':policy.get('local_steps',{}).get('local_relevel_steps'),
                            'success':policy['success'],'process_violation':policy['protection_violation'],
                            'wait_seconds':policy.get('inference_wait_seconds'),'total_seconds':policy.get('total_post_fault_seconds')}),flush=True)
            report=analyze(rows,protocol,setup,time.perf_counter()-started)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'valid_execution':report['valid_execution'],'methods':report['methods']}),flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                worker.wait(timeout=30)


if __name__=='__main__':main()
