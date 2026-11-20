"""Frozen VLM with refreshed local forecasts consumed at the shared-prefix boundary."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from . import l4_lift
from .duration_closed_loop import policy_trial,render_rgb,trial_summary
from .duration_pilot import SOURCE_FILES,STATE_KEYS,file_hash,prefix_frames,retimed_config
from .l0 import _import_robosuite
from .vlm_duration_policy import candidate_bank


METHODS=('oracle_role_fixed60','vlm_with_wait','prefix_overlap')


def read_response(worker):
    line=worker.stdout.readline()
    if not line:raise RuntimeError('model worker exited before decision')
    result=json.loads(line)
    if 'error' in result:raise RuntimeError(result['error'])
    # Timestamp arrival here, independently of the foreground physics consumer.
    return result,time.perf_counter()


def aligned_trial(seed,scenario,output,worker,post_wait=False,method='prefix_overlap',fixed_duration=None):
    original=l4_lift.TwoArmLiftEnv
    holder={};acquisition={}
    def factory(*args,**kwargs):
        env=original(*args,**kwargs);env.raw.has_offscreen_renderer=True
        holder['env']=env
        original_reset=env.reset
        def reset_and_warm_camera():
            original_reset()
            tick=time.perf_counter();render_rgb(env)
            acquisition['camera_setup_seconds']=time.perf_counter()-tick
        env.reset=reset_and_warm_camera
        return env

    def send(port,op,candidate=None):
        rgb=render_rgb(holder['env'])
        if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid rendered observation')
        image=output/'observations'/f'{seed}-{scenario}-{op}.npz'
        np.savez_compressed(image,rgb=rgb)
        request={'image':str(image.resolve()),'robot':port,'op':op}
        if candidate:request['candidate']=candidate
        state=l4_lift.snapshot(holder['env'])
        if op=='local_durations' and post_wait:
            before=acquisition['scope']
            request.update(post_wait=True,before_image=before['observation_file'],before_robot=before['robot'],
                wait_steps=round((state['time']-before['state']['time'])*20))
        worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
        acquisition[op]={'observation_file':str(image.resolve()),'state':state,'robot':port}

    def select(port):
        send(port,'scope')
        decision=read_response(worker)[0]
        if fixed_duration is not None and decision.get('parallel_local'):
            decision={**decision,'parallel_local':False,'local_relevel_steps':fixed_duration,
                'forced_duration_baseline':True,'accepted':True}
        return decision

    pending={}
    with ThreadPoolExecutor(max_workers=1) as reader:
        def duration_select(operation,port,candidate,prefix_seconds):
            if operation=='start':
                pending['started']=time.perf_counter()
                send(port,'local_durations',candidate)
                pending['future']=reader.submit(read_response,worker)
                return
            result,arrived=pending.pop('future').result()
            return {**result,'pipeline_wall_seconds':arrived-pending['started']}

        tick=time.perf_counter()
        with patch.object(l4_lift,'TwoArmLiftEnv',factory):
            row=l4_lift.run_trial(seed,scenario,selector=select,account_selector_latency=True,
                                  duration_selector=duration_select)
        # Drain any outstanding response even if physical execution failed.
        if 'future' in pending:pending['future'].result()
        row.update(method=method,episode_wall_seconds=time.perf_counter()-tick,acquisition=acquisition)
    if 'selection' in row:row['selection']['rgb_frames']=4
    if 'duration_selection' in row:row['duration_selection']['rgb_frames']=8 if post_wait else 4
    return row


def analyze(rows,protocol,setup,wall_seconds):
    paired={};mismatches=[];ledger_errors=[];cases=[]
    for r in rows:paired.setdefault((r['seed'],r['scenario']),{})[r['method']]=r
    for key,pair in paired.items():
        reference=pair['oracle_role_fixed60']
        for name,row in pair.items():
            for field in STATE_KEYS:
                if row.get(field)!=reference.get(field):mismatches.append([*key,name,field])
            if prefix_frames(row)!=prefix_frames(reference):mismatches.append([*key,name,'initialization_and_fault_trace'])
            wait=[f for f in row['trace'] if f['stage'] in ('inference_wait','duration_deadline_wait')]
            first=[f for f in wait if f['stage']=='inference_wait']
            actual=row.get('selector_wall_seconds',0.)
            if (len(wait)!=row['inference_wait_steps'] or
                abs(len(wait)*.05-row['inference_wait_seconds'])>1e-8 or
                len(first)*.05+1e-9<actual or len(first)*.05-actual>.05+1e-9 or
                any(f['phase']!='recovery' for f in wait)):
                ledger_errors.append([*key,name,'physical_wait_ledger'])
            if 'total_post_fault_seconds' in row:
                total=row['inference_wait_seconds']+row['skill_seconds']+row['hold_seconds']
                if abs(total-row['total_post_fault_seconds'])>1e-8:
                    ledger_errors.append([*key,name,'total_cost_identity'])
            if 'duration_selection' in row and 'decision' in row['duration_selection']:
                d=row['duration_selection'];elapsed=d['pipeline_wall_seconds'];prefix=d['prefix_seconds']
                expected=l4_lift.wait_control_steps(max(0.,elapsed-prefix))
                suffix_wait=[f for f in wait if f['stage']=='duration_deadline_wait']
                if expected!=len(suffix_wait) or d['deadline_missed']!=bool(expected):
                    ledger_errors.append([*key,name,'overlap_ledger'])
                if expected and (not row.get('abstained') or any(f['stage']=='local_relevel' for f in row['trace'])):
                    ledger_errors.append([*key,name,'late_forecast_consumed'])
                if row['duration_selection']['observation_state']!=row['post_wait_state']:
                    ledger_errors.append([*key,name,'forecast_observation_not_refreshed'])
            cases.append({'seed':key[0],'scenario':key[1],'method':name,'candidate':row['candidate'],
                'relevel_steps':row.get('local_steps',{}).get('local_relevel_steps'),
                'success':row['success'],'process_violation':row['protection_violation'],
                'safe_success':row['success'] and not row['protection_violation'],
                'abstained':row.get('abstained',False) or row['candidate']=='abstain',
                'failure_reason':row['failure_reason'],'scope_wall_seconds':actual,
                'duration_wall_seconds':row.get('duration_selection',{}).get('pipeline_wall_seconds',0.),
                'overlapped_seconds':row.get('duration_selection',{}).get('overlapped_seconds',0.),
                'prefix_seconds':row.get('duration_selection',{}).get('prefix_seconds',0.),
                'deadline_missed':row.get('duration_selection',{}).get('deadline_missed',False),
                'wait_seconds':row['inference_wait_seconds'],'skill_seconds':row.get('skill_seconds'),
                'total_seconds':row.get('total_post_fault_seconds'),
                'search_attempts':sum(f['stage']=='local_arc_close' and (i==0 or row['trace'][i-1]['stage']!='local_arc_close') for i,f in enumerate(row['trace'])),
                'keeper_lost_steps':row['protection']['keeper_grasp_lost_steps']})
    frozen=all(file_hash(p)==digest for p,digest in protocol['frozen_artifacts_sha256'].items())
    sources=all(file_hash(p)==digest for p,digest in protocol['source_sha256'].items())
    errors=[{k:r.get(k) for k in ('seed','scenario','method','error','blocked_calls')} for r in rows if r.get('error') or r.get('blocked_calls')]
    summaries={}
    for name in METHODS:
        group=[r for r in rows if r['method']==name]
        s=trial_summary(group)
        s['abstentions']=sum(r.get('abstained',False) or r['candidate']=='abstain' for r in group)
        s['selected_local_relevel_steps']={str(n):sum(r.get('local_steps',{}).get('local_relevel_steps')==n for r in group) for n in (30,40,60)}
        s['mean_duration_pipeline_wall_seconds']=float(np.mean([r.get('duration_selection',{}).get('pipeline_wall_seconds',0.) for r in group]))
        s['mean_overlapped_seconds']=float(np.mean([r.get('duration_selection',{}).get('overlapped_seconds',0.) for r in group]))
        s['mean_continuation_wait_seconds']=float(np.mean([r.get('continuation_wait_seconds',0.) for r in group]))
        s['deadline_misses']=sum(r.get('duration_selection',{}).get('deadline_missed',False) for r in group)
        summaries[name]=s
    sync=summaries['vlm_with_wait'];aligned=summaries['prefix_overlap']
    return {'physical_states':len(paired),'physical_episodes':len(rows),'seeds':protocol['seeds'],
        'valid_execution':len(rows)==protocol['planned_physical_episodes'] and not mismatches and not ledger_errors and not errors and frozen and sources,
        'prefix_mismatches':mismatches,'wait_accounting_errors':ledger_errors,'execution_errors':errors,
        'frozen_artifacts_unchanged':frozen,'sources_unchanged_during_test':sources,
        'methods':summaries,'cases':cases,'model_setup':setup,'campaign_wall_seconds':wall_seconds,
        'aligned_minus_synchronous':{'safe_successes':aligned['safe_successes']-sync['safe_successes'],
            'process_violations':aligned['process_violations']-sync['process_violations'],
            'mean_total_seconds':aligned['mean_total_post_fault_seconds']-sync['mean_total_post_fault_seconds']},
        'time_definition':'Post-fault total = scope/full-selection hold + local-prefix/suffix execution + uncovered duration-pipeline hold + final 1-second hold. Duration latency is measured from before refreshed RGB acquisition to arrival in the background reader. Overlap is already paid as physical prefix execution. Late forecasts are discarded.',
        'scope':'Deterministic delay emulation with concurrent GPU computation and unpaced simulation. Unique start-score admission permits a speculative shared local prefix; it is not a safety guarantee. Full conditional acceptance occurs at the duration boundary. All waits, speculative prefixes and abstentions remain in the denominator. Oracle-role fixed60 receives privileged repair role and no inference cost.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    parser.add_argument('--model-path',default='/home/lc/AI4Chemical/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--gpu-python',default='/opt/miniconda3/bin/python')
    parser.add_argument('--gpu',default='2')
    parser.add_argument('--seeds',type=int,nargs='+',default=[2000,2001,2002,2003])
    args=parser.parse_args();output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('test run exists; do not overwrite')
    (output/'observations').mkdir()
    point=json.loads(Path(args.working_point).read_text());frozen=[Path(args.working_point).resolve()]
    for key in ('start_adapter','outcome_adapter'):
        frozen.extend(Path(point[key])/name for name in ('adapter_model.safetensors','adapter_config.json'))
    directory=Path(__file__).resolve().parent
    source_names=(*SOURCE_FILES,'vlm_duration_policy.py','duration_closed_loop.py','duration_aligned_loop.py')
    protocol={'seeds':args.seeds,'scenarios':['left_loss','right_loss'],'methods':METHODS,
        'planned_physical_episodes':len(args.seeds)*6,'threshold':point['threshold'],
        'working_point':str(Path(args.working_point).resolve()),'model_path':args.model_path,
        'frozen_artifacts_sha256':{str(p):file_hash(p) for p in frozen},
        'source_sha256':{str(directory/name):file_hash(directory/name) for name in source_names},
        'candidate_bank':candidate_bank(),'ordering':'Fixed60 first; synchronous then overlap on even case index, reverse on odd index.',
        'scope_gate':'Both frozen start scores at h0. Exactly one >= unchanged threshold permits its shared local prefix; otherwise run original full synchronous selection with cached start scores.',
        'timing_policy':'Execute ceil(scope pipeline seconds / 0.05) holds. Refresh RGB/proprioception at h1 and start local q*p(A|valid,plan) selection while executing the unchanged 16-step approach and 22-step close (repeat search if needed). Consume only before relevel. Count ceil(max(0, duration pipeline wall seconds - physical prefix seconds)/0.05) additional holds; discard late forecasts and abstain. No accepted duration also abstains. All prefix and hold steps are audited.',
        'no_extra_model_information':'Policy receives only four RGB images, measured robot positions/fingers and registered reference positions. The local role is supplied by the frozen start model, never the test fault label.',
        'setup':'Once-only model loading/warmup and pre-initialization camera setup recorded separately; full episode program wall time also reported.',
        'freeze':'No training, score/threshold tuning or controller changes after test begins.',
        'test_scope':'8 fresh fault states, 24 physical simulation rollouts; small independent pilot, no population guarantee.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    started=time.perf_counter();rows=[]
    with (output/'model-worker.log').open('w') as err:
        worker=subprocess.Popen([args.gpu_python,'-m','auditembodied.vlm_duration_policy','--working-point',args.working_point,
            '--model-path',args.model_path],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,env=env)
        try:
            setup=json.loads(worker.stdout.readline())
            if not setup.get('ready'):raise RuntimeError(setup)
            print(json.dumps({'model_ready':setup}),flush=True)
            _import_robosuite(True)
            with (output/'trials.jsonl').open('w') as f:
                for index,(seed,scenario) in enumerate((s,q) for s in args.seeds for q in ('left_loss','right_loss')):
                    tick=time.perf_counter()
                    fixed=l4_lift.run_trial(seed,scenario,'R_left' if scenario=='left_loss' else 'R_right',retimed_config(1.5,'relevel_only'))
                    fixed.update(method='oracle_role_fixed60',episode_wall_seconds=time.perf_counter()-tick)
                    order=('vlm_with_wait','prefix_overlap') if index%2==0 else ('prefix_overlap','vlm_with_wait')
                    result=[fixed]
                    for method in order:
                        trial=policy_trial if method=='vlm_with_wait' else aligned_trial
                        result.append(trial(seed,scenario,output,worker))
                    for row in result:
                        rows.append(row);f.write(json.dumps(row)+'\n')
                        print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','inference_wait_seconds','total_post_fault_seconds','failure_reason')}
                            | {'relevel_steps':row.get('local_steps',{}).get('local_relevel_steps'),
                               'deadline_missed':row.get('duration_selection',{}).get('deadline_missed')}),flush=True)
                    f.flush()
            report=analyze(rows,protocol,setup,time.perf_counter()-started)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            with (output/'cases.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(report['cases'][0]));writer.writeheader();writer.writerows(report['cases'])
            print(json.dumps({'valid_execution':report['valid_execution'],'methods':report['methods']}),flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)


if __name__=='__main__':main()
