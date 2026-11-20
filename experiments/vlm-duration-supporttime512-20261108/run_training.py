"""One fixed512-step fit; same primary examples/updates as the saved control."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
with (root / 'training.log').open('w') as log:
    subprocess.run([sys.executable, '-m', 'auditembodied.vlm_duration_lora',
        '--data-dir', str(root / 'dataset'), '--model-path', plan['model_path'],
        '--output-dir', str(root / 'supporttime512'), '--steps', '512', '--learning-rate', '3e-5',
        '--seed', str(plan['seed']), '--sampling', 'outcome-full', '--temporal-mode', 'stage_late',
        '--init-adapter', plan['initial_outcome_adapter'], '--start-adapter', plan['frozen_start_adapter'],
        '--admission-threshold', str(plan['threshold']), '--admission-loss-weight', '0.1',
        '--admission-margin', '1.0', '--support-time-loss-weight', str(plan['support_time_loss_weight']),
        '--final-eval-only'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
baseline = Path(plan['baseline_dir'])
assert (root / 'supporttime512/sampling-plan.jsonl').read_bytes() == (baseline / 'sampling-plan.jsonl').read_bytes()
print(json.dumps({'completed': True, 'steps': 512, 'same_primary_sampling': True}), flush=True)
