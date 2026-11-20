"""Run the declared128-step train gate, then512 fresh only if it passes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from assess import assess_diagnostic

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')


def train(directory, data, steps, train_only):
    command = [sys.executable, '-m', 'auditembodied.vlm_duration_lora',
        '--data-dir', str(data), '--model-path', plan['model_path'], '--output-dir', str(root / directory),
        '--steps', str(steps), '--learning-rate', str(plan['learning_rate']), '--seed', str(plan['seed']),
        '--sampling', 'outcome-family', '--temporal-mode', 'stage_joint',
        '--init-adapter', plan['initial_outcome_adapter'], '--start-adapter', plan['frozen_start_adapter'],
        '--admission-threshold', str(plan['threshold']), '--admission-loss-weight', '0.1',
        '--admission-margin', '1.0', '--process-loss-weight', '0.5', '--support-time-loss-weight', '0', '--final-eval-only']
    if train_only:
        command.append('--train-only')
    with (root / (directory + '.log')).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


train('diagnostic128', root / 'diagnostic-data', 128, True)
gate = assess_diagnostic(root)
print(json.dumps({'diagnostic_complete': True, 'passed': gate['passed'], 'actual': gate['gate_actual']}), flush=True)
if gate['passed']:
    data = root / 'full-data'
    data.mkdir()
    for split in ('train', 'development'):
        shutil.copyfile(Path(plan['source_collection']) / 'dataset' / (split + '.jsonl'), data / (split + '.jsonl'))
    train('full512', data, 512, False)
    print(json.dumps({'full512_complete': True, 'fresh_initialization': True}), flush=True)
else:
    print('Stopped at the declared training gate; development and full512 remain untouched.', flush=True)
