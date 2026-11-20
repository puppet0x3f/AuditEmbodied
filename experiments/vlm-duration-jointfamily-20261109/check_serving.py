"""Final-checkpoint parity and fixed-budget timing, with no dev access after a failed gate."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
gate = json.loads((root / 'diagnostic-gate.json').read_text())
read = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
directory = root / ('full512' if gate['passed'] else 'diagnostic128')
data = root / ('full-data' if gate['passed'] else 'diagnostic-data')
split = 'development' if gate['passed'] else 'train'
prediction_file = 'after-predictions.jsonl' if gate['passed'] else 'train-after-predictions.jsonl'
rows = [r for r in read(data / (split + '.jsonl')) if r['post_prefix_wait_steps'] == 66]
expected = {(r['seed'], r['scenario'], r['wait_steps'], r['plan']['local_relevel_steps']): r
    for r in read(directory / prediction_file)}
# Physical capture costs are from the same TRAIN states unless the full stage was authorized by its gate.
allowed_seeds = {r['seed'] for r in rows}
captures = {}
with (Path(plan['source_collection']) / 'trials-wait0.jsonl').open() as stream:
    for line in stream:
        seed = int(re.match(r'\s*\{\s*"seed"\s*:\s*(\d+)\s*,', line).group(1))
        if seed not in allowed_seeds:
            continue
        row = json.loads(line)
        if row['local_steps']['local_relevel_steps'] == 30:
            captures[row['seed'], row['scenario']] = row['stage_capture']['h2']['acquisition_seconds']
groups = {}
for row in rows:
    groups.setdefault((row['seed'], row['scenario']), []).append(row)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
records, errors = [], []
with (root / 'serving-worker.log').open('w') as log:
    worker = subprocess.Popen([sys.executable, '-m', 'auditembodied.vlm_duration_policy',
        '--working-point', plan['frozen_working_point'], '--model-path', plan['model_path'],
        '--post-wait-adapter', str(directory / 'adapter'), '--post-wait-mode', 'stage_joint'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env)
    try:
        ready = json.loads(worker.stdout.readline())
        if not ready.get('ready'):
            raise RuntimeError(ready)
        def request(group):
            row = group[0]
            message = {k: row[k] for k in ('image', 'robot', 'before_image', 'before_robot',
                'after_prefix_image', 'after_prefix_robot', 'early_history', 'late_history')}
            message.update(op='score_examples', post_wait=True, plans=[r['plan'] for r in group])
            tick = time.perf_counter()
            worker.stdin.write(json.dumps(message) + '\n')
            worker.stdin.flush()
            answer, _ = read_response(worker)
            return answer, time.perf_counter() - tick
        warm = [r for r in read(data / 'train.jsonl') if r['seed'] == 40 and
            r['scenario'] == 'left_loss' and r['post_prefix_wait_steps'] == 66]
        request(warm)
        for key, group in groups.items():
            answer, elapsed = request(group)
            assert len(answer['records']) == 3
            assert [r['task'] for r in answer['forward_timings']] == ['start', 'outcome', 'outcome', 'outcome']
            for row, score in zip(group, answer['records']):
                ref = expected[row['seed'], row['scenario'], 66, row['plan']['local_relevel_steps']]
                errors.extend(abs(a - b) for a, b in zip(ref['class_probabilities'], score['class_probabilities']))
            total = elapsed + captures[key]
            records.append({'seed': key[0], 'scenario': key[1], 'request_wall_seconds': elapsed,
                'recorded_physical_capture_seconds': captures[key], 'pipeline_seconds': total,
                'W_seconds': 3.3, 'deadline_missed': total > 3.3, 'forward_timings': answer['forward_timings']})
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            worker.wait(timeout=30)
report = {'model_setup': ready, 'checkpoint': str((directory / 'adapter').resolve()), 'split': split,
    'stage_warmup': 'Training seed40 left, no development warmup', 'families': records,
    'probability_max_error': max(errors), 'deadline_misses': sum(r['deadline_missed'] for r in records),
    'scope': 'Final serving parity/timing only; includes existing actual capture timing, no new physics or policy rollouts. W remains fixed.'}
assert report['probability_max_error'] <= 1e-6
(root / 'serving-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'split': split, 'families': len(records), 'probability_max_error': report['probability_max_error'],
    'deadline_misses': report['deadline_misses']}), flush=True)
