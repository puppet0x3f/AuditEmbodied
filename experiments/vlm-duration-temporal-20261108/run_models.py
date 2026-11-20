"""Verify restored base against frozen scores, then run the two fixed budgets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from auditembodied.duration_aligned_loop import read_response

out=Path(__file__).resolve().parent
plan_path=out/'experiment-plan.json';plan=json.loads(plan_path.read_text())
base=Path('outputs/models/Qwen3-VL-8B-Instruct').resolve()
if len(list(base.glob('model-*-of-*.safetensors')))!=4:raise RuntimeError('base download incomplete')
plan['model_path']=str(base);plan_path.write_text(json.dumps(plan,indent=2)+'\n')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
    TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(Path.cwd()))
read=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
examples=read('outputs/l4-duration-newstates-20261108/dataset/development.jsonl')
expected=read('outputs/vlm-duration-lora-riskreplay512-20261108/after-predictions.jsonl')
groups={};actual={}
for i,r in enumerate(examples):groups.setdefault((r['image'],json.dumps(r['robot'],sort_keys=True)),[]).append((i,r))
with (out/'base-reference-check.log').open('w') as log:
    worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy','--working-point',
        'outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json','--model-path',str(base)],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    try:
        ready=json.loads(worker.stdout.readline())
        if not ready.get('ready'):raise RuntimeError(ready)
        for group in groups.values():
            r=group[0][1];request={'op':'score_examples','image':r['image'],'robot':r['robot'],'plans':[x['plan'] for _,x in group]}
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();response,_=read_response(worker)
            for (i,ex),score in zip(group,response['records']):actual[i]=score
        error=max(abs(x-y) for i,r in enumerate(expected) for x,y in zip(r['class_probabilities'],actual[i]['class_probabilities']))
        check={'examples':len(expected),'probability_max_error':error,'reference_ready':ready,'base_model_path':str(base),
            'reference_scope':'Original 56 development examples, old outcome and start adapters, unchanged prompts. Only verify restored base identity, no learning or model selection.'}
        (out/'base-reference-check.json').write_text(json.dumps(check,indent=2)+'\n')
        if error>1e-6:raise ValueError('restored base does not reproduce original scores')
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
print('base_reference_verified',error,flush=True)
failed=out/'static512'
if failed.exists() and not (failed/'training.jsonl').exists():failed.rename(out/'startup-missing-base')
for mode in ('static','paired'):
    directory=out/f'{mode}512';directory.mkdir()
    with (directory/'run.log').open('w') as log:
        subprocess.run([sys.executable,'-m','auditembodied.vlm_duration_lora',
            '--data-dir',str(out/'collection/dataset'),'--model-path',str(base),'--output-dir',str(directory),
            '--steps','512','--learning-rate','0.00003','--seed','20261108','--sampling','outcome-state',
            '--temporal-mode',mode,'--init-adapter',plan['old_outcome_adapter'],'--start-adapter',plan['frozen_start_adapter']],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('training_complete',mode,flush=True)
with (out/'assessment.log').open('w') as log:
    subprocess.run([sys.executable,str(out/'assess.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('assessment_complete',flush=True)
