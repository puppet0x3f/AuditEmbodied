"""Final full512 parity and measured deadline against the unchanged waiting budget."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--reference-baseline', action='store_true')
args = parser.parse_args()
checkpoint_dir = Path(plan['baseline_dir']) if args.reference_baseline else root / 'full512'
input_mode = 'stage_late' if args.reference_baseline else 'stage_joint'
suffix = '-baseline' if args.reference_baseline else ''
read = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
rows = [r for r in read(root / 'dataset/development.jsonl') if r['post_prefix_wait_steps'] == 66]
expected = {(r['seed'], r['scenario'], r['wait_steps'], r['plan']['local_relevel_steps']): r
    for r in read(checkpoint_dir / 'after-predictions.jsonl')}
captures = {(r['seed'], r['scenario']): r['stage_capture']['h2']['acquisition_seconds']
    for r in read(Path(plan['source_collection']) / 'trials-wait0.jsonl') if r['local_steps']['local_relevel_steps'] == 30}
groups = {}
for r in rows:
    groups.setdefault((r['seed'], r['scenario']), []).append(r)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
records, errors = [], []
with (root / ('serving-worker' + suffix + '.log')).open('w') as log:
    worker = subprocess.Popen([sys.executable, '-m', 'auditembodied.vlm_duration_policy',
        '--working-point', plan['frozen_working_point'], '--model-path', plan['model_path'],
        '--post-wait-adapter', str(checkpoint_dir / 'adapter'), '--post-wait-mode', input_mode],
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
        warm = [r for r in read(root / 'dataset/train.jsonl') if r['seed'] == 40 and
            r['scenario'] == 'left_loss' and r['post_prefix_wait_steps'] == 66]
        request(warm)
        for key, group in groups.items():
            answer, elapsed = request(group)
            assert len(answer['records']) == 3
            assert [r['task'] for r in answer['forward_timings']] == ['start', 'outcome', 'outcome', 'outcome']
            for row, score in zip(group, answer['records']):
                assert row['plan'] == score['plan']
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
report = {'model_setup': ready, 'checkpoint': str((checkpoint_dir / 'adapter').resolve()), 'input_mode': input_mode,
    'split': 'development', 'stage_warmup': 'Training seed40 left before any development request',
    'families': records, 'probability_max_error': max(errors),
    'deadline_misses': sum(r['deadline_missed'] for r in records),
    'scope': 'Final-model parity and timing only; existing real capture costs included, no new physical collection or independent closed-loop execution. W unchanged.'}
assert report['probability_max_error'] <= 1e-6
(root / ('serving-check' + suffix + '.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'families': len(records), 'probability_max_error': report['probability_max_error'],
    'deadline_misses': report['deadline_misses']}), flush=True)
