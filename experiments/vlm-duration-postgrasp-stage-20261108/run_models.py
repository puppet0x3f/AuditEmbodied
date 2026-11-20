"""Two final512-step fits with identical data, sampling and frozen prerequisites."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
plan=json.loads((root/'experiment-plan.json').read_text())
physical=json.loads((root/'collection/physical-report.json').read_text())
assert physical['valid'] and physical['physical_episodes']==96
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
for mode in ('stage_early','stage_late'):
    output=root/(mode+'512')
    with (root/(mode+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-m','auditembodied.vlm_duration_lora',
            '--data-dir',str(root/'collection/dataset'),'--model-path',plan['model_path'],
            '--output-dir',str(output),'--steps','512','--learning-rate',str(plan['learning_rate']),
            '--seed',str(plan['seed']),'--sampling','outcome-full','--temporal-mode',mode,
            '--init-adapter',plan['initial_outcome_adapter'],'--start-adapter',plan['frozen_start_adapter'],
            '--admission-threshold',str(plan['threshold']),'--admission-loss-weight','0.1',
            '--admission-margin','1.0','--final-eval-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'completed':mode,'steps':512}),flush=True)
assert (root/'stage_early512/sampling-plan.jsonl').read_bytes()==(root/'stage_late512/sampling-plan.jsonl').read_bytes()
print('both_final_checkpoints_complete',flush=True)
