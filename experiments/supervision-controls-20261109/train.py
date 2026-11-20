"""Run the three predeclared objectives sequentially at the same finite budget."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
p=json.loads((root/'protocol.json').read_text())
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=p['training_gpu'],OMP_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(root/'frozen-source'))
for variant in p['variants']:
    target=root/(variant+'512')
    command=[sys.executable,'-m','auditembodied.vlm_duration_lora','--data-dir',p['data_dir'],
        '--model-path',p['model_path'],'--output-dir',str(target),'--steps',str(p['steps']),
        '--learning-rate',str(p['learning_rate']),'--seed',str(p['seed']),
        '--sampling','outcome-family','--temporal-mode','family_stream',
        '--init-adapter',p['initial_outcome_adapter'],'--start-adapter',p['frozen_start_adapter'],
        '--admission-threshold',str(p['scope_threshold']),'--family-replay-balanced',
        '--supervision',variant,'--final-eval-only']
    if variant=='joint':command+=['--process-loss-weight','.5','--admission-loss-weight','.1',
        '--admission-margin','1','--family-boundary','set']
    print(json.dumps({'starting':variant,'steps':p['steps']}),flush=True)
    with (root/('train-'+variant+'.log')).open('x') as stream:
        subprocess.run(command,cwd=root/'frozen-source',env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=json.loads((target/'result.json').read_text())
    assert result['completed_steps']==p['steps'] and result['frozen_start_unchanged'] and result['data_unchanged']
    assert (target/'sampling-plan.jsonl').read_bytes()==(root/'sampling-plan.jsonl').read_bytes()
    print(json.dumps({'completed':variant,'training_seconds':result['training_seconds'],
        'peak_allocated_gpu_bytes':result['peak_allocated_gpu_bytes']}),flush=True)
