"""Frozen training-only serial versus cached batch3 deployment check."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response

parser=argparse.ArgumentParser();parser.add_argument('--shared-vision',action='store_true');args=parser.parse_args()
optimized='score_examples_shared' if args.shared_vision else 'score_examples_batch'
suffix='-shared' if args.shared_vision else ''
root=Path(__file__).resolve().parent
plan=json.loads((root/'experiment-plan.json').read_text())
read=lambda p:[json.loads(x) for x in Path(p).read_text().splitlines()]
rows=read(Path(plan['previous_round'])/'dataset/train.jsonl')
groups={}
for r in rows:
    if r['wait_steps']==66:groups.setdefault((r['seed'],r['scenario']),[]).append(r)
captures={(r['seed'],r['scenario']):r['stage_capture']['h2']['acquisition_seconds']
    for r in read(Path(plan['source_collection'])/'trials-wait0.jsonl')
    if r['local_steps']['local_relevel_steps']==30}
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
records=[]
with (root/('benchmark-worker'+suffix+'.log')).open('w') as log:
    worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy',
        '--working-point',plan['frozen_working_point'],'--model-path',plan['model_path'],
        '--post-wait-adapter',plan['initial_outcome_adapter'],'--post-wait-mode','stage_joint'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
    try:
        ready=json.loads(worker.stdout.readline())
        if not ready.get('ready'):raise RuntimeError(ready)
        print(json.dumps(ready),flush=True)
        def request(message):
            tick=time.perf_counter();worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush()
            answer,_=read_response(worker)
            return answer,time.perf_counter()-tick
        def message(group,op):
            row=group[0]
            result={k:row[k] for k in ('image','robot','before_image','before_robot',
                'after_prefix_image','after_prefix_robot','early_history','late_history')}
            return {**result,'op':op,'post_wait':True,'plans':[r['plan'] for r in group]}
        warm=next(iter(groups.values()))
        request(message(warm,optimized))
        for repetition in range(3):
            for key,group in groups.items():
                row=group[0]
                cached,qwall=request({'op':'cache_start','image':row['image'],'robot':row['robot'],'plan':row['plan']})
                # Interleave serial and batch to reduce shared-device drift.
                operations=['score_examples',optimized]
                if (repetition+row['seed'])%2:operations.reverse()
                answers={op:request(message(group,op)) for op in operations}
                serial,swall=answers['score_examples'];batch,bwall=answers[optimized]
                choose=lambda a:next((r['plan']['local_relevel_steps'] for r in a['records'] if r['accepted']),None)
                admission_same=[r['accepted'] for r in serial['records']]==[r['accepted'] for r in batch['records']]
                error=max(abs(x-y) for a,b in zip(serial['records'],batch['records'])
                    for x,y in zip(a['class_probabilities'],b['class_probabilities']))
                record={'seed':key[0],'scenario':key[1],'repetition':repetition,'start_cache_wall_seconds':qwall,
                    'start_cache_hit':cached['start_cache_hit'],'serial_wall_seconds':swall,'batch_wall_seconds':bwall,
                    'h2_recorded_capture_seconds':captures[key],'serial_pipeline_seconds':swall+captures[key],
                    'batch_pipeline_seconds':bwall+captures[key],'admission_identical':admission_same,
                    'selected_identical':choose(serial)==choose(batch),'serial_selected':choose(serial),'batch_selected':choose(batch),
                    'probability_max_error':error,'batch_start_cache_hits':batch['start_cache_hits'],
                    'batch_forward_timings':batch['forward_timings']}
                records.append(record)
                with (root/('benchmark-cases'+suffix+'.jsonl')).open('a') as stream:stream.write(json.dumps(record)+'\n')
                print(json.dumps({k:record[k] for k in ('seed','scenario','repetition','batch_pipeline_seconds','admission_identical','selected_identical')}),flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
report={'operation':optimized,'requests':len(records),'setup':ready,'same_admission':all(r['admission_identical'] for r in records),
    'same_selection':all(r['selected_identical'] for r in records),
    'probability_max_error':max(r['probability_max_error'] for r in records),
    'serial_deadline_misses':sum(r['serial_pipeline_seconds']>3.3 for r in records),
    'batch_deadline_misses':sum(r['batch_pipeline_seconds']>3.3 for r in records),
    'batch_pipeline_mean':sum(r['batch_pipeline_seconds'] for r in records)/len(records),'cases':records,
    'scope':'Training-only36 requests; h2 recorded real capture added; h1 cache preparation separately charged in later live physics; batch uses identical prompts, bf16 weights and threshold.'}
(root/('benchmark'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
assert report['same_admission'] and report['same_selection']
print(json.dumps({k:v for k,v in report.items() if k not in ('cases','setup')}),flush=True)
