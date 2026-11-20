"""Matched final512 experiments; all checkpoints initialize from the same source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=('set','per-candidate'),required=True)
parser.add_argument('--gpu',required=True);args=parser.parse_args()
root=Path(__file__).resolve().parent;p=json.loads((root/'experiment-plan.json').read_text())
output=root/('set512' if args.variant=='set' else 'candidate512')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
command=[sys.executable,'-m','auditembodied.vlm_duration_lora','--data-dir',str(root/'dataset'),
    '--model-path',p['model_path'],'--output-dir',str(output),'--steps','512','--learning-rate','0.00003',
    '--seed','20261108','--sampling','outcome-family','--temporal-mode','family_stream',
    '--init-adapter',p['initial_outcome_adapter'],'--start-adapter',p['frozen_start_adapter'],
    '--admission-threshold',str(p['threshold']),'--admission-loss-weight','0.1','--admission-margin','1',
    '--process-loss-weight','0.5','--family-replay-balanced','--family-stratify-safe-sets',
    '--family-boundary',args.variant,'--final-eval-only']
with (root/('training-'+args.variant+'.log')).open('w') as log:
    subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
r=json.loads((output/'result.json').read_text())
assert r['completed_steps']==512 and r['frozen_start_unchanged'] and r['data_unchanged']
print(json.dumps({'variant':args.variant,'complete':True,'training_seconds':r['training_seconds']}),flush=True)
