"""Conditional fresh-seed comparison, with every measured wait audited."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from auditembodied.duration_aligned_loop import aligned_trial,read_response
from auditembodied.duration_closed_loop import trial_summary
from auditembodied.duration_pilot import STATE_KEYS,SOURCE_FILES,prefix_frames
from auditembodied.l0 import _import_robosuite
from auditembodied.vlm_duration_policy import candidate_bank


out=Path(__file__).resolve().parent
assessment=json.loads((out/'assessment.json').read_text())
if not assessment['paired_development_gate_passed']:
    raise SystemExit('Development gate failed; fresh seeds must remain unused.')
plan=json.loads((out/'experiment-plan.json').read_text());dest=out/'closed-loop'
if dest.exists():raise RuntimeError('independent run exists; refusing overwrite')
dest.mkdir();methods=('old_overlap','paired_overlap','same_gate_fixed60')
for method in methods:(dest/method/'observations').mkdir(parents=True)
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen={**plan['frozen_files_sha256'],**{str(p):digest(p) for p in (out/'paired512/adapter').glob('adapter_*')}}
source=Path('auditembodied').resolve()
protocol={'seeds':plan['conditional_independent_loop']['seeds'],'scenarios':['left_loss','right_loss'],
    'methods':methods,'planned_episodes':24,'threshold':plan['threshold'],'frozen_sha256':frozen,
    'source_sha256':{str(source/n):digest(source/n) for n in (*SOURCE_FILES,'duration_aligned_loop.py','vlm_duration_policy.py','vlm_duration_temporal.py')},
    'order':'Rotate the three methods by case index before any results.',
    'fixed60':'Same frozen visual role gate and measured initial hold, fixed 60 local relevel; no privileged fault label.',
    'latency':'Scope hold is physically replayed; duration computation overlaps the shared prefix. Uncovered waiting is physically replayed and late forecasts discarded. All cost and guard steps retained.',
    'scope':'Small independent simulated closed-loop pilot using deterministic delay emulation, not a realtime robot deployment.'}
(dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(Path.cwd()))
started=time.perf_counter();rows=[]
with (dest/'model-worker.log').open('w') as log:
    worker=subprocess.Popen(['/opt/miniconda3/bin/python','-m','auditembodied.vlm_duration_policy',
        '--working-point','outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json',
        '--model-path',plan['model_path'],'--post-wait-adapter',str(out/'paired512/adapter'),'--post-wait-mode','paired'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    try:
        setup=json.loads(worker.stdout.readline())
        if not setup.get('ready'):raise RuntimeError(setup)
        # Warm the new route on a training observation only, before test episodes.
        examples=[json.loads(s) for s in (out/'collection/dataset/train.jsonl').read_text().splitlines()]
        example=next(r for r in examples if r['label']!='E')
        request={'op':'score_examples','image':example['image'],'robot':example['robot'],'post_wait':True,
            'before_image':example['before_image'],'before_robot':example['before_robot'],'wait_steps':example['wait_steps'],
            'plans':[c['plan'] for c in candidate_bank() if c['candidate']=='R_'+example['plan']['repair_arm']]}
        tick=time.perf_counter();worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();read_response(worker)
        setup['post_wait_warmup_seconds']=time.perf_counter()-tick
        _import_robosuite(True)
        with (dest/'trials.jsonl').open('w') as f:
            for index,(seed,scenario) in enumerate((s,q) for s in protocol['seeds'] for q in protocol['scenarios']):
                order=methods[index%3:]+methods[:index%3]
                for method in order:
                    row=aligned_trial(seed,scenario,dest/method,worker,post_wait=method=='paired_overlap',
                        method=method,fixed_duration=60 if method=='same_gate_fixed60' else None)
                    rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                    print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','inference_wait_seconds','total_post_fault_seconds','failure_reason')}),flush=True)
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)

groups={};errors=[];cases=[]
for r in rows:groups.setdefault((r['seed'],r['scenario']),[]).append(r)
for key,group in groups.items():
    ref=group[0]
    for r in group:
        if any(r.get(k)!=ref.get(k) for k in STATE_KEYS) or prefix_frames(r)!=prefix_frames(ref):errors.append([*key,r['method'],'fault_prefix'])
        if r.get('error') or r.get('blocked_calls') or not r.get('fault_class_valid'):errors.append([*key,r['method'],'execution_or_fault'])
        waits=[f for f in r['trace'] if f['stage'] in ('inference_wait','duration_deadline_wait')]
        first=[f for f in waits if f['stage']=='inference_wait']
        if (len(waits)!=r['inference_wait_steps'] or abs(len(waits)*.05-r['inference_wait_seconds'])>1e-8 or
            any(f['phase']!='recovery' for f in waits) or len(first)*.05+1e-9<r['selector_wall_seconds'] or
            len(first)*.05-r['selector_wall_seconds']>.05+1e-9):errors.append([*key,r['method'],'wait_ledger'])
        if 'total_post_fault_seconds' in r and abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])>1e-8:
            errors.append([*key,r['method'],'total_cost'])
        d=r.get('duration_selection',{})
        if d.get('deadline_missed') and (not r.get('abstained') or any(f['stage']=='local_relevel' for f in r['trace'])):
            errors.append([*key,r['method'],'late_forecast_consumed'])
        cases.append({'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],'success':r['success'],
            'process_violation':r['protection_violation'],'abstained':r.get('abstained',False) or r['candidate']=='abstain',
            'duration':r.get('local_steps',{}).get('local_relevel_steps'),'wait_seconds':r['inference_wait_seconds'],
            'total_seconds':r.get('total_post_fault_seconds'),'deadline_missed':d.get('deadline_missed',False),
            'keeper_lost_steps':r['protection']['keeper_grasp_lost_steps']})
summaries={}
for method in methods:
    group=[r for r in rows if r['method']==method];s=trial_summary(group)
    s['abstentions']=sum(r.get('abstained',False) or r['candidate']=='abstain' for r in group)
    s['deadline_misses']=sum(r.get('duration_selection',{}).get('deadline_missed',False) for r in group)
    summaries[method]=s
unchanged=all(digest(p)==h for p,h in frozen.items()) and all(digest(p)==h for p,h in protocol['source_sha256'].items())
report={'valid_execution':len(rows)==24 and len(groups)==8 and not errors and unchanged,'errors':errors,
    'frozen_files_and_sources_unchanged':unchanged,'methods':summaries,'cases':cases,
    'model_setup':setup,'campaign_wall_seconds':time.perf_counter()-started,'scope':protocol['scope']}
(dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
with (dest/'cases.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
print(json.dumps({'valid_execution':report['valid_execution'],'methods':summaries}),flush=True)
