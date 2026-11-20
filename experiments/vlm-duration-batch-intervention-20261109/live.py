"""Eight untouched states, frozen learned policy versus same-scope fixed60; reuse identical visual features across candidates."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np

from auditembodied import l4_lift
from auditembodied.duration_aligned_loop import read_response
from auditembodied.duration_closed_loop import render_rgb,trial_summary
from auditembodied.duration_pilot import STATE_KEYS,prefix_frames,file_hash
from auditembodied.duration_stage_data import robot_only
from auditembodied.l0 import _import_robosuite
from auditembodied.vlm_duration_data import plan as recovery_plan

ROOT=Path(__file__).resolve().parent


def live_trial(seed,scenario,method,worker,output):
    holder={};observations={};original=l4_lift.TwoArmLiftEnv;pending={}
    def factory(*args,**kwargs):
        env=original(*args,**kwargs);holder['env']=env;env.raw.has_offscreen_renderer=True
        reset=env.reset
        def warmed_reset():
            reset();tick=time.perf_counter();render_rgb(env)
            observations['camera_setup_seconds']=time.perf_counter()-tick
        env.reset=warmed_reset
        return env
    def capture(name,port):
        tick=time.perf_counter();rgb=render_rgb(holder['env'])
        if any(float(x.std())<=1 for x in rgb):raise ValueError('invalid RGB')
        path=output/'observations'/f'{seed}-{scenario}-{method}-{name}.npz'
        np.savez_compressed(path,rgb=rgb)
        item={'image':str(path.resolve()),'robot':robot_only(port),'history':port.get('observed_history',[]),
            'state':l4_lift.snapshot(holder['env']),'acquisition_seconds':time.perf_counter()-tick}
        observations[name]=item
        return item
    def send(message):
        worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush()
    def select(port):
        h0=capture('h0',port);send({'op':'scope','image':h0['image'],'robot':h0['robot']})
        result,_=read_response(worker)
        if method=='fixed60' and result.get('parallel_local'):
            result={**result,'parallel_local':False,'accepted':True,'local_relevel_steps':60,'fixed60_reference':True}
        return result
    with ThreadPoolExecutor(max_workers=1) as reader:
        def duration(operation,port,candidate,prefix):
            if operation=='start':
                tick=time.perf_counter();h1=capture('h1',port)
                p=recovery_plan({'candidate':candidate})
                send({'op':'cache_start','image':h1['image'],'robot':h1['robot'],'plan':p})
                pending['future']=reader.submit(read_response,worker);pending['started']=tick
                return
            # Residual prerequisite computation/queue is included if not ready.
            tick=time.perf_counter();cached,arrived=pending.pop('future').result()
            h2=capture('h2',port);h1=observations['h1'];h0=observations['h0']
            scope_wait=round((h1['state']['time']-h0['state']['time'])*20)
            # The runner measured the same synchronous capture+enqueue callback.
            prefix_hold=port['prefix_observation_wait_steps']
            plans=[]
            for n in (30,40,60):
                p=recovery_plan({'candidate':candidate,'local_steps':{'local_relevel_steps':n}})
                p.update(scope_hold_steps=scope_wait,post_prefix_wait_steps=66,final_hold_steps=20,
                    prefix_observation_hold_steps=prefix_hold)
                plans.append(p)
            request={'op':'choose_examples_shared','post_wait':True,'image':h1['image'],'robot':h1['robot'],
                'before_image':h0['image'],'before_robot':h0['robot'],'early_history':h1['history'],
                'after_prefix_image':h2['image'],'after_prefix_robot':h2['robot'],'late_history':h2['history'],'plans':plans}
            send(request);answer,arrived_batch=read_response(worker)
            accepted=[r for r in answer['records'] if r['accepted']]
            result={'candidate':candidate,'accepted':bool(accepted),'scored_candidates':answer['records'],
                'pipeline_wall_seconds':arrived_batch-tick,'start_cache':cached,'batch':answer,
                'prerequisite_wall_seconds':arrived-pending['started'],
                'residual_prerequisite_wait_seconds':max(0.,arrived-tick)}
            if accepted:result['local_relevel_steps']=accepted[0]['plan']['local_relevel_steps']
            result['pipeline_wall_seconds']=time.perf_counter()-tick
            return result
        tick=time.perf_counter()
        with patch.object(l4_lift,'TwoArmLiftEnv',factory):
            row=l4_lift.run_trial(seed,scenario,selector=select,account_selector_latency=True,
                duration_selector=duration if method=='vlm_shared' else None,
                refresh_duration_observation=True,fixed_duration_wait_steps=66 if method=='vlm_shared' else None,
                account_prefix_observation_latency=method=='vlm_shared')
        if 'future' in pending:pending['future'].result()
        row.update(method=method,episode_wall_seconds=time.perf_counter()-tick,acquisition=observations)
    return row


def main():
    p=json.loads((ROOT/'experiment-plan.json').read_text());output=ROOT/'closed-loop'
    if (output/'trials.jsonl').exists():raise ValueError('independent run already exists')
    (output/'observations').mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'frozen-inputs-before.json').read_text())
    for name in ('adapter_model.safetensors','adapter_config.json'):
        path=ROOT/'mixed512/adapter'/name;frozen[str(path)]=file_hash(path)
    for name in ('l4_lift.py','vlm_duration_policy.py','vlm_duration_temporal.py'):
        path=Path('/home/lc/AuditEmbodied/auditembodied')/name;frozen[str(path)]=file_hash(path)
    (output/'frozen-before.json').write_text(json.dumps(frozen,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')
    rows=[];started=time.perf_counter()
    with (output/'worker.log').open('w') as log:
        worker=subprocess.Popen(['/opt/miniconda3/bin/python','-m','auditembodied.vlm_duration_policy',
            '--working-point',p['frozen_working_point'],'--model-path',p['model_path'],
            '--post-wait-adapter',str(ROOT/'mixed512/adapter'),'--post-wait-mode','stage_joint'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        try:
            ready=json.loads(worker.stdout.readline())
            if not ready.get('ready'):raise RuntimeError(ready)
            # All serving warmup observations are training examples.
            warm=json.loads((ROOT/'dataset/train.jsonl').read_text().splitlines()[0])
            family=[json.loads(s) for s in (ROOT/'dataset/train.jsonl').read_text().splitlines()[:3]]
            request={k:warm[k] for k in ('image','robot','before_image','before_robot','after_prefix_image',
                'after_prefix_robot','early_history','late_history')}
            request.update(op='score_examples_shared',post_wait=True,plans=[r['plan'] for r in family])
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();read_response(worker)
            _import_robosuite(True)
            with (output/'trials.jsonl').open('w') as stream:
                for index,(seed,scenario) in enumerate((s,q) for s in p['test_seeds'] for q in ('left_loss','right_loss')):
                    methods=['fixed60','vlm_shared'] if (index+index//2)%2==0 else ['vlm_shared','fixed60']
                    for method in methods:
                        row=live_trial(seed,scenario,method,worker,output)
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                        print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation',
                            'inference_wait_seconds','total_post_fault_seconds','failure_reason','error')}
                            | {'N':row.get('local_steps',{}).get('local_relevel_steps')}),flush=True)
        finally:
            if worker.poll() is None:
                worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
    cases=[]
    for row in rows:
        if row.get('error'):raise RuntimeError(row['error'])
        assert not row.get('blocked_calls')
        assert row['state_edit_count_after_initialization']==row['contact_anchor_count']==row['external_wrench_at_end']==0
        assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-7
        holds=[f for f in row['trace'] if f['stage'] in ('inference_wait','prefix_observation_wait','duration_budget_wait','duration_deadline_wait')]
        assert len(holds)==row['inference_wait_steps'] and abs(len(holds)/20-row['inference_wait_seconds'])<1e-7
        d=row.get('duration_selection',{})
        cases.append({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],'candidate':row['candidate'],
            'N':row.get('local_steps',{}).get('local_relevel_steps'),'success':row['success'],
            'process_violation':row['protection_violation'],'safe_success':row['success'] and not row['protection_violation'],
            'abstained':row.get('abstained',False) or row['candidate']=='abstain',
            'pipeline_seconds':d.get('pipeline_wall_seconds'),'deadline_missed':d.get('deadline_missed',False),
            'accepted_forecast':bool(d.get('decision',{}).get('accepted')),
            'accepted_plan_discarded_for_deadline':bool(d.get('deadline_missed') and d.get('decision',{}).get('accepted')),
            'wait_seconds':row['inference_wait_seconds'],'total_post_fault_seconds':row['total_post_fault_seconds'],
            'episode_wall_seconds':row['episode_wall_seconds']})
    for seed in p['test_seeds']:
        for scenario in ('left_loss','right_loss'):
            a,b=[r for r in rows if r['seed']==seed and r['scenario']==scenario]
            assert all(a[k]==b[k] for k in STATE_KEYS) and prefix_frames(a)==prefix_frames(b)
    assert len(rows)==16 and all(file_hash(k)==v for k,v in frozen.items())
    summaries={}
    for method in ('fixed60','vlm_shared'):
        group=[r for r in rows if r['method']==method];c=[r for r in cases if r['method']==method]
        summaries[method]={**trial_summary(group),'deadline_misses':sum(r['deadline_missed'] for r in c),
            'accepted_plans_discarded_for_deadline':sum(r['accepted_plan_discarded_for_deadline'] for r in c),
            'abstentions':sum(r['abstained'] for r in c),'selected_N':[r['N'] for r in c]}
    report={'independent_states':8,'episodes':16,'methods':summaries,'cases':cases,'frozen_unchanged':True,
        'setup':ready,'campaign_wall_seconds':time.perf_counter()-started,
        'scope':'Fresh3000-3003, same frozen scope model for both policies, no fault role oracle. Fixed60 skips post-prefix VLM. All observed captures/queue/transfer/inference are charged as20Hz physical holds; all late replies discarded and final hold retained. Small simulation pilot.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
