"""Final late-adapter parity and latency on the fixed heldout W families."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response

root=Path(__file__).resolve().parent;plan=json.loads((root/'experiment-plan.json').read_text())
read=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
rows=[r for r in read(root/'collection/dataset/development.jsonl') if r['post_prefix_wait_steps']==plan['W_steps']]
expected=read(root/'stage_late512/after-predictions.jsonl')
expected={(r['seed'],r['scenario'],r['wait_steps'],r['plan']['local_relevel_steps']):r for r in expected}
captures={(r['seed'],r['scenario']):r['stage_capture']['h2']['acquisition_seconds']
    for r in read(root/'collection/trials-wait0.jsonl') if r['local_steps']['local_relevel_steps']==30}
groups={}
for r in rows:groups.setdefault((r['seed'],r['scenario']),[]).append(r)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
records=[];errors=[]
with (root/'serving-worker.log').open('w') as log:
    worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy',
        '--working-point',plan['frozen_working_point'],'--model-path',plan['model_path'],
        '--post-wait-adapter',str(root/'stage_late512/adapter'),'--post-wait-mode','stage_late'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    try:
        ready=json.loads(worker.stdout.readline())
        if not ready.get('ready'):raise RuntimeError(ready)
        warm=[r for r in read(root/'collection/dataset/train.jsonl') if r['seed']==40 and
            r['scenario']=='left_loss' and r['post_prefix_wait_steps']==plan['W_steps']]
        ref=warm[0]
        warm_request={k:ref[k] for k in ('image','robot','before_image','before_robot','after_prefix_image',
            'after_prefix_robot','early_history','late_history')}
        warm_request.update(op='score_examples',post_wait=True,plans=[r['plan'] for r in warm])
        worker.stdin.write(json.dumps(warm_request)+'\n');worker.stdin.flush();read_response(worker)
        for key,group in groups.items():
            r=group[0]
            request={k:r[k] for k in ('image','robot','before_image','before_robot','after_prefix_image',
                'after_prefix_robot','early_history','late_history')}
            request.update(op='score_examples',post_wait=True,plans=[x['plan'] for x in group])
            tick=time.perf_counter();worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
            answer,_=read_response(worker);elapsed=time.perf_counter()-tick
            for ex,score in zip(group,answer['records']):
                assert score['plan']==ex['plan']
                ref=expected[ex['seed'],ex['scenario'],ex['wait_steps'],ex['plan']['local_relevel_steps']]
                errors.extend(abs(a-b) for a,b in zip(ref['class_probabilities'],score['class_probabilities']))
            total=elapsed+captures[key]
            records.append({'seed':key[0],'scenario':key[1],'request_wall_seconds':elapsed,
                'capture_seconds':captures[key],'pipeline_seconds':total,'W_seconds':plan['W_seconds'],
                'deadline_missed':total>plan['W_seconds'],'forward_timings':answer['forward_timings']})
    finally:
        if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
report={'model_setup':ready,'stage_warmup':'Training seed40 left, before heldout requests','families':records,'probability_max_error':max(errors),
    'deadline_misses':sum(r['deadline_missed'] for r in records),
    'scope':'Serving parity/timing of final checkpoint; W is unchanged. No selected-policy physical rollouts. Late-result rejection/extra audited holding is verified separately in integration tests.'}
assert report['probability_max_error']<=1e-6
(root/'serving-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'probability_max_error':report['probability_max_error'],'deadline_misses':report['deadline_misses'],'families':len(records)}),flush=True)
