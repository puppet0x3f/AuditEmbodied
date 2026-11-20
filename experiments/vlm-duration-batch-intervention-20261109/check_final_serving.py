"""Final adapter serving parity and actual remaining deadline on fixed dev."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response

root=Path(__file__).resolve().parent
p=json.loads((root/'experiment-plan.json').read_text())
read=lambda x:[json.loads(s) for s in Path(x).read_text().splitlines()]
groups={}
for row in read(root/'dataset/development.jsonl'):groups.setdefault((row['seed'],row['scenario'],row['wait_steps']),[]).append(row)
expected={(r['seed'],r['scenario'],r['wait_steps'],r['plan']['local_relevel_steps']):r for r in read(root/'mixed512/after-predictions.jsonl')}
captures={(r['seed'],r['scenario']):r['stage_capture']['h2']['acquisition_seconds']
    for r in read(Path(p['source_collection'])/'trials-wait0.jsonl') if r['local_steps']['local_relevel_steps']==30}
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
errors=[];records=[];deployment=[]
with (root/'final-serving-worker.log').open('w') as log:
    worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy',
        '--working-point',p['frozen_working_point'],'--model-path',p['model_path'],
        '--post-wait-adapter',str(root/'mixed512/adapter'),'--post-wait-mode','stage_joint'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    try:
        ready=json.loads(worker.stdout.readline())
        if not ready.get('ready'):raise RuntimeError(ready)
        def request(message):
            tick=time.perf_counter();worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush()
            answer,_=read_response(worker)
            return answer,time.perf_counter()-tick
        def score(group,op='score_examples_shared'):
            row=group[0]
            request({'op':'cache_start','image':row['image'],'robot':row['robot'],'plan':row['plan']})
            message={k:row[k] for k in ('image','robot','before_image','before_robot','after_prefix_image',
                'after_prefix_robot','early_history','late_history')}
            message.update(op=op,post_wait=True,plans=[r['plan'] for r in group])
            return request(message)
        score(read(root/'dataset/train.jsonl')[:3])
        for key,group in groups.items():
            answer,elapsed=score(group)
            for row,result in zip(group,answer['records']):
                ref=expected[(*key, row['plan']['local_relevel_steps'])]
                assert result['accepted']==(ref['class_probabilities'][0]>=p['threshold'])
                errors.extend(abs(a-b) for a,b in zip(ref['class_probabilities'],result['class_probabilities']))
            records.append({'seed':key[0],'scenario':key[1],'W_steps':key[2],'request_wall_seconds':elapsed,
                'recorded_h2_capture_seconds':captures[key[:2]],'pipeline_seconds':elapsed+captures[key[:2]],
                'deadline_missed':key[2]==66 and elapsed+captures[key[:2]]>3.3,'forward_timings':answer['forward_timings']})
            if key[2]==66:
                lazy,elapsed=score(group,'choose_examples_shared')
                selected=lambda rows:next((r['plan']['local_relevel_steps'] for r in rows if r['accepted']),None)
                assert selected(lazy['records'])==selected(answer['records'])
                for short,full in zip(lazy['records'],answer['records']):
                    assert short['class_probabilities']==full['class_probabilities']
                deployment.append({'seed':key[0],'scenario':key[1],'W_steps':66,'request_wall_seconds':elapsed,
                    'pipeline_seconds':elapsed+captures[key[:2]],'deadline_missed':elapsed+captures[key[:2]]>3.3,
                    'scored_candidates':len(lazy['records']),'selected_steps':selected(lazy['records']),
                    'selection_identical':True,'forward_timings':lazy['forward_timings']})
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
assert max(errors)<=1e-6
report={'examples':24,'families':records,'max_probability_error':max(errors),'admission_identical':True,
    'W66_deadline_misses':sum(r['deadline_missed'] for r in records),'W66_families':4,'setup':ready,
    'deployment_families':deployment,'deployment_deadline_misses':sum(r['deadline_missed'] for r in deployment),
    'scope':'Final adapter only; same cached start and shared visual features as live policy. Fixed development parity/timing, no tuning; actual held-out physics is separate.'}
(root/'final-serving.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('families','setup')}),flush=True)
