"""Choose W from training-only post-prefix request latency, never outcomes."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response
from auditembodied.duration_stage_data import stage_plan

root=Path(__file__).resolve().parent
plan=json.loads((root/'experiment-plan.json').read_text())
rows=[json.loads(s) for s in (root/'collection/trials-wait0.jsonl').read_text().splitlines()]
assert len(rows)==48
families=[[r for r in rows if r['seed']==s and r['scenario']==q]
    for s in (40,41) for q in ('left_loss','right_loss')]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
measurements=[]
with (root/'calibration-worker.log').open('w') as log:
    worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy',
        '--working-point',plan['frozen_working_point'],'--model-path',plan['model_path'],
        '--post-wait-adapter',plan['initial_outcome_adapter'],'--post-wait-mode','stage_late'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    try:
        ready=json.loads(worker.stdout.readline())
        if not ready.get('ready'):raise RuntimeError(ready)
        def query(group):
            ref=next(r for r in group if r['local_steps']['local_relevel_steps']==30);c=ref['stage_capture']
            request={'op':'score_examples','post_wait':True,'image':c['h1']['image'],'robot':c['h1']['robot'],
                'before_image':c['h0']['image'],'before_robot':c['h0']['robot'],
                'after_prefix_image':c['h2']['image'],'after_prefix_robot':c['h2']['robot'],
                'early_history':c['h1']['history'],'late_history':c['h2']['history'],
                'plans':[stage_plan(r) for r in group]}
            tick=time.perf_counter();worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
            answer,_=read_response(worker);elapsed=time.perf_counter()-tick
            assert len(answer['records'])==3
            assert [t['images'] for t in answer['forward_timings']]==[4,8,8,8]
            return {'seed':ref['seed'],'scenario':ref['scenario'],'request_wall_seconds':elapsed,
                'h2_acquisition_seconds':c['h2']['acquisition_seconds'],
                'full_pipeline_seconds':elapsed+c['h2']['acquisition_seconds']}
        query(families[0])
        for rep in range(3):
            for group in families:
                item={'repetition':rep,**query(group)};measurements.append(item);print(json.dumps(item),flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
maximum=max(r['full_pipeline_seconds'] for r in measurements)
w=math.ceil(1.2*maximum/.05)
report={'training_seeds':[40,41],'heldout_used':False,'warmup_families':1,'measurements':measurements,
    'maximum_pipeline_seconds':maximum,'safety_factor':1.2,'W_steps':w,'W_seconds':w/20,'model_setup':ready,
    'note':'Frozen pre-regression outcome/start models, stage-late eight-image input; classification results never choose W.'}
(root/'wait-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
plan['W_steps']=w;plan['W_seconds']=w/20
(root/'experiment-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'W_steps':w,'W_seconds':w/20}),flush=True)
