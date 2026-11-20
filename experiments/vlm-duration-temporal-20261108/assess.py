"""Frozen-threshold comparison, old-route retention, and serving parity."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response


out=Path(__file__).resolve().parent
read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
plan=json.loads((out/'experiment-plan.json').read_text());threshold=plan['threshold']
data=out/'collection/dataset'


def metrics(rows):
    safe=[r for r in rows if r['label']=='A'];accepted=[r for r in rows if r['class_probabilities'][0]>=threshold]
    return {'examples':len(rows),'safe_candidates':len(safe),'safe_admitted':sum(r['label']=='A' for r in accepted),
        'false_admissions':sum(r['label']!='A' for r in accepted),
        'false_by_class':dict(Counter(r['label'] for r in accepted if r['label']!='A')),
        'native_class_accuracy':sum(r['prediction']==r['label'] for r in rows)/len(rows)}


def families(rows):
    groups={}
    for r in rows:
        p=r['plan']
        if p['kind']!='local' or p['local_approach_steps']!=16 or p['local_close_steps']!=22:continue
        key=r['image'],json.dumps(r['robot'],sort_keys=True),p['repair_arm']
        groups.setdefault(key,{})[p['local_relevel_steps']]=r
    cases=[]
    for group in groups.values():
        if set(group)!={30,40,60}:continue
        ref=group[30];safe=sorted(n for n,r in group.items() if r['label']=='A')
        admitted=sorted(n for n,r in group.items() if r['class_probabilities'][0]>=threshold)
        cases.append({'seed':ref['seed'],'scenario':ref['scenario'],'safe_durations':safe,'admitted_durations':admitted,
            'all_rejected':not admitted,'chosen':admitted[0] if admitted else None,
            'safe_choice':bool(admitted) and admitted[0] in safe,'safe_covered':bool(set(safe)&set(admitted))})
    no_safe=[c for c in cases if not c['safe_durations']]
    return {'states':len(cases),'recoverable_states':sum(bool(c['safe_durations']) for c in cases),
        'safe_choice_states':sum(c['safe_choice'] for c in cases),'safe_covered_states':sum(c['safe_covered'] for c in cases),
        'no_safe_states':len(no_safe),'no_safe_states_fully_rejected':sum(c['all_rejected'] for c in no_safe),'cases':cases}


def grouped_worker_scores(worker,examples,post_wait):
    groups={};result={};timings=[]
    for i,r in enumerate(examples):
        key=r['image'],json.dumps(r['robot'],sort_keys=True)
        groups.setdefault(key,[]).append((i,r))
    for group in groups.values():
        ref=group[0][1]
        request={'op':'score_examples','image':ref['image'],'robot':ref['robot'],'plans':[r['plan'] for _,r in group]}
        if post_wait:request.update(post_wait=True,before_image=ref['before_image'],before_robot=ref['before_robot'],wait_steps=ref['wait_steps'])
        worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer,_=read_response(worker)
        if len(answer['records'])!=len(group):raise ValueError('worker score count')
        for (i,row),score in zip(group,answer['records']):
            if row['plan']!=score['plan']:raise ValueError('worker plan mismatch')
            result[i]={**row,**score}
        timings.extend(answer['forward_timings'])
    return [result[i] for i in range(len(examples))],timings


old_dev=read(Path('outputs/l4-duration-newstates-20261108/dataset/development.jsonl'))
old_scores=read(Path('outputs/vlm-duration-lora-riskreplay512-20261108/after-predictions.jsonl'))
old_wait=read(Path('outputs/l4-duration-wait-matrix-20261108/predictions.jsonl'))
old_wait=[r for r in old_wait if r['split']=='development']
for r in old_wait:r['joint_A_score']=r['class_probabilities'][0]
baseline={'old_development':metrics(old_scores),'wait_development':metrics(old_wait),'wait_families':families(old_wait)}
variants={};flat=[]
for mode in ('static','paired'):
    directory=out/f'{mode}512';trained=json.loads((directory/'result.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text());updates=read(directory/'training.jsonl')
    if trained['completed_steps']!=512 or len(updates)!=512 or not trained['frozen_start_unchanged'] or not trained['data_unchanged']:
        raise ValueError('training budget/freeze failure')
    if [u['step'] for u in updates]!=list(range(1,513)):raise ValueError('optimizer step sequence')
    scores={s:read(directory/('train-after-predictions.jsonl' if s=='train' else 'after-predictions.jsonl')) for s in ('train','development')}
    examples={s:read(data/f'{s}.jsonl') for s in ('train','development')}
    cohorts={}
    for split,rows in scores.items():
        if len(rows)!=len(examples[split]):raise ValueError('evaluation count')
        for a,b in zip(rows,examples[split]):
            if any(a[k]!=b[k] for k in ('seed','scenario','plan','label','image','robot')):raise ValueError('evaluation identity')
        for source in sorted({r['source'] for r in rows}):
            subset=[r for r in rows if r['source']==source]
            cohorts[f'{split}_{source}']={'candidates':metrics(subset),'families':families(subset)}
            flat.append({'variant':mode,'cohort':f'{split}_{source}',**metrics(subset)})
    post_examples=[r for r in examples['development'] if r['source']=='existing_wait']
    post_scores=[r for r in scores['development'] if r['source']=='existing_wait']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    with (directory/'serving-check.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy','--working-point',
            'outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json',
            '--model-path',plan['model_path'],
            '--post-wait-adapter',str(directory/'adapter'),'--post-wait-mode',mode],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        try:
            ready=json.loads(worker.stdout.readline())
            if not ready.get('ready'):raise RuntimeError(ready)
            retained,old_timings=grouped_worker_scores(worker,old_dev,False)
            runtime,new_timings=grouped_worker_scores(worker,post_examples,True)
        finally:
            if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
    retained_error=max(abs(a-b) for x,y in zip(retained,old_scores) for a,b in zip(x['class_probabilities'],y['class_probabilities']))
    runtime_error=max(abs(a-b) for x,y in zip(runtime,post_scores) for a,b in zip(x['class_probabilities'],y['class_probabilities']))
    q_error=max(abs(a['start_valid_score']-b['start_valid_score']) for a,b in zip(runtime,old_wait))
    expected_images=8 if mode=='paired' else 4
    route_valid=(all(t['adapter'] in ('default','start') and t['images']==4 for t in old_timings) and
        all(t['images']==4 and t['adapter']=='start' if t['task']=='start' else
            t['images']==expected_images and t['adapter']=='post_wait' for t in new_timings))
    m=cohorts['development_existing_wait']['candidates'];f=cohorts['development_existing_wait']['families']
    gate={'old_route_retained':retained_error<=1e-6,'serving_matches_training':runtime_error<=1e-6,
        'start_scores_unchanged':q_error==0.,'correct_input_routing':route_valid,
        'false_admissions_at_most_2':m['false_admissions']<=2,'safe_admission_at_least_13':m['safe_admitted']>=13,
        'both_no_safe_families_rejected':f['no_safe_states']==f['no_safe_states_fully_rejected']==2}
    variants[mode]={'training_seconds':trained['training_seconds'],'peak_gpu_bytes':trained['peak_allocated_gpu_bytes'],
        'presentations':sum(len(u['example_indices']) for u in updates),'training_plan_sha256':hashlib.sha256((directory/'sampling-plan.jsonl').read_bytes()).hexdigest(),
        'cohorts':cohorts,'old_development_retained':metrics(retained),'old_probability_max_error':retained_error,
        'serving_probability_max_error':runtime_error,'start_probability_max_error':q_error,
        'gate':gate,'gate_passed':all(gate.values()),'model_setup':ready}
    (directory/'serving-check.json').write_text(json.dumps({k:v for k,v in variants[mode].items() if k!='cohorts'},indent=2)+'\n')
    print(json.dumps({'variant':mode,'wait_development':m,'no_safe_rejected':f['no_safe_states_fully_rejected'],'gate':gate}),flush=True)
frozen=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in plan['frozen_files_sha256'].items())
same_schedule=variants['static']['training_plan_sha256']==variants['paired']['training_plan_sha256']
if not frozen or not same_schedule:raise ValueError('frozen files or shared training schedule changed')
report={'threshold':threshold,'frozen_files_unchanged':frozen,'same_training_schedule':same_schedule,
    'baseline':baseline,'variants':variants,'paired_development_gate_passed':variants['paired']['gate_passed'],
    'scope':'Offline development comparison of two final 512-step checkpoints. Retention is an unchanged old route; post-wait generalization is evaluated separately. No threshold search or test-set training.'}
prior=json.loads(Path('outputs/vlm-duration-lora-postwait512-20261108/comparison.json').read_text())
report['previous_joint_finetune']={name:prior['cohorts'][name]['after'] for name in ('original_development','post_wait_development')}
report['base_reference_check']=json.loads((out/'base-reference-check.json').read_text())
report['base_provenance']=json.loads((out/'base-provenance.json').read_text())
(out/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
