"""Verify unchanged native A-D serving and the already fixed3.3-second budget."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response

root = Path(__file__).resolve().parent
plan = json.loads((root / 'experiment-plan.json').read_text())
collection = Path(plan['source_collection'])
read = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
rows = [r for r in read(root / 'dataset/development.jsonl') if r['post_prefix_wait_steps'] == 66]
expected = {(r['seed'], r['scenario'], r['wait_steps'], r['plan']['local_relevel_steps']): r
    for r in read(root / 'supporttime512/after-predictions.jsonl')}
captures = {(r['seed'], r['scenario']): r['stage_capture']['h2']['acquisition_seconds']
    for r in read(collection / 'trials-wait0.jsonl') if r['local_steps']['local_relevel_steps'] == 30}
groups = {}
for r in rows:
    groups.setdefault((r['seed'], r['scenario']), []).append(r)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='2', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
records, errors = [], []
with (root / 'serving-worker.log').open('w') as log:
    worker = subprocess.Popen([sys.executable, '-m', 'auditembodied.vlm_duration_policy',
        '--working-point', '/home/lc/AuditEmbodied/outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json',
        '--model-path', plan['model_path'], '--post-wait-adapter', str(root / 'supporttime512/adapter'),
        '--post-wait-mode', 'stage_late'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=log, text=True, bufsize=1, env=env)
    try:
        ready = json.loads(worker.stdout.readline())
        if not ready.get('ready'):
            raise RuntimeError(ready)
        warm = [r for r in read(root / 'dataset/train.jsonl') if r['seed'] == 40 and
            r['scenario'] == 'left_loss' and r['post_prefix_wait_steps'] == 66]
        def request(group):
            r = group[0]
            message = {k: r[k] for k in ('image', 'robot', 'before_image', 'before_robot',
                'after_prefix_image', 'after_prefix_robot', 'early_history', 'late_history')}
            message.update(op='score_examples', post_wait=True, plans=[x['plan'] for x in group])
            tick = time.perf_counter()
            worker.stdin.write(json.dumps(message) + '\n')
            worker.stdin.flush()
            answer, _ = read_response(worker)
            return answer, time.perf_counter() - tick
        request(warm)
        for key, group in groups.items():
            answer, elapsed = request(group)
            assert len(answer['records']) == 3
            assert [f['task'] for f in answer['forward_timings']] == ['start', 'outcome', 'outcome', 'outcome']
            for example, score in zip(group, answer['records']):
                assert score['plan'] == example['plan']
                ref = expected[example['seed'], example['scenario'], 66, example['plan']['local_relevel_steps']]
                errors.extend(abs(a - b) for a, b in zip(ref['class_probabilities'], score['class_probabilities']))
            total = elapsed + captures[key]
            records.append({'seed': key[0], 'scenario': key[1], 'request_wall_seconds': elapsed,
                'capture_seconds': captures[key], 'pipeline_seconds': total, 'W_seconds': 3.3,
                'deadline_missed': total > 3.3, 'forward_timings': answer['forward_timings']})
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n')
            worker.stdin.flush()
            worker.wait(timeout=30)
report = {'model_setup': ready, 'stage_warmup': 'Training seed40 left, before development requests',
    'families': records, 'probability_max_error': max(errors),
    'deadline_misses': sum(r['deadline_missed'] for r in records), 'auxiliary_time_forward_count': 0,
    'scope': 'Serving parity/timing only. Same fixed W and real previously collected hold costs; no new physical or selected-policy rollouts.'}
assert report['probability_max_error'] <= 1e-6
(root / 'serving-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'probability_max_error': report['probability_max_error'],
    'deadline_misses': report['deadline_misses'], 'families': len(records)}), flush=True)
