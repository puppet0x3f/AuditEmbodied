"""Fresh full-data512 fit; always evaluate the final checkpoint on both splits."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
command = [sys.executable, '-m', 'auditembodied.vlm_duration_lora',
    '--data-dir', str(root / 'dataset'), '--model-path', plan['model_path'], '--output-dir', str(root / 'full512'),
    '--steps', '512', '--learning-rate', str(plan['learning_rate']), '--seed', str(plan['seed']),
    '--sampling', 'outcome-family', '--temporal-mode', 'stage_joint',
    '--init-adapter', plan['initial_outcome_adapter'], '--start-adapter', plan['frozen_start_adapter'],
    '--admission-threshold', str(plan['threshold']), '--admission-loss-weight', '0.1',
    '--admission-margin', '1.0', '--process-loss-weight', '0.5', '--support-time-loss-weight', '0', '--final-eval-only']
with (root / 'training.log').open('w') as log:
    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
result = json.loads((root / 'full512/result.json').read_text())
assert result['completed_steps'] == 512 and result['development'] is not None
print(json.dumps({'full512_complete': True, 'train_and_development_evaluated': True,
    'initialization': plan['initial_outcome_adapter'], 'training_accuracy_gate': None}), flush=True)
