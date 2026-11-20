"""Scientific figure from the executed 12006 right-loss trajectories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

root = Path(__file__).resolve().parent
rows = {}
for line in (root / 'trials.jsonl').open():
    row = json.loads(line)
    if row['seed'] == 12006 and row['scenario'] == 'right_loss':
        rows[row['method']] = row
row = rows['agentchord_style']
frames = [f for f in row['trace'] if f['phase'] == 'recovery' and f.get('skill') == 'R_right']
t = np.array([f['time'] - row['prefix_state']['time'] for f in frames])
truth = np.array([f['evaluator']['holds'][0] for f in frames], dtype=float)
width = np.array([np.abs(f['robot']['fingers'][0]).mean() for f in frames])
proxy = ((width > .004) & (width < .028)).astype(float)
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), gridspec_kw={'height_ratios': [1, 1.45]})
axes[0].step(t, proxy + .04, where='post', label='Encoder load proxy', color='#258672', lw=2)
axes[0].step(t, truth, where='post', label='Keeper grasp (evaluation only)', color='#b83c40', lw=2)
first = t[np.flatnonzero(truth == 0)[0]]
axes[0].axvline(first, color='#b83c40', ls=':', alpha=.7)
axes[0].text(first + .04, .38, 'First true loss\nproxy still positive', fontsize=9)
axes[0].set(xlim=(0, t[-1] + .05), ylim=(-.15, 1.3), yticks=[0, 1],
    yticklabels=['Absent', 'Present'], xlabel='Seconds after fault (local recovery only)')
axes[0].legend(loc='upper left', fontsize=8, ncol=2)
axes[0].set_title('A valid start and continuous monitoring miss a support interruption', fontsize=11)
colors = {'wait': '#d9aa4d', 'local': '#3973ac', 'global': '#258672', 'hold': '#a2a6aa'}
labels = ['Protected rule', 'Frozen MLP + start gate', 'AgentChord-style', 'Fixed global']
methods = ['protected', 'gated_direct', 'agentchord_style', 'global']
for i, (method, label) in enumerate(zip(methods, labels)):
    r = rows[method]
    if method == 'agentchord_style':
        events = [e for e in r['skill_events'] if e['candidate'] != 'continue']
        segments = [('wait', r['inference_wait_seconds'])] + [
            ('global' if e['candidate'] == 'R_global' else 'local', e['end_time'] - e['start_time']) for e in events]
    else:
        segments = [('wait', r['inference_wait_seconds']),
                    ('global' if r['candidate'] == 'R_global' else 'local', r['skill_seconds'])]
    segments.append(('hold', r['hold_seconds']))
    start = 0.
    for name, length in segments:
        axes[1].barh(i, length, left=start, color=colors[name], height=.55)
        start += length
    status = 'safe' if r['success'] and not r['protection_violation'] else 'complete, unsafe' if r['task_complete'] else 'failed, unsafe'
    axes[1].text(start + .12, i, f"{start:.2f}s; {status}", fontsize=8, va='center')
axes[1].set(yticks=range(4), yticklabels=labels, xlim=(0, 22.5), xlabel='Total post-fault seconds (all attempts and final hold)')
axes[1].invert_yaxis()
fig.legend(handles=[Patch(color=c, label=n.title()) for n, c in colors.items()],
    loc='lower center', bbox_to_anchor=(.6, .005), ncol=4, fontsize=8)
axes[1].set_title('A later global recovery restores completion, while the earlier violation remains', fontsize=11)
for axis in axes:
    axis.grid(axis='x', alpha=.2)
fig.suptitle('Seed12006 / right loss: identical physical fault prefix and registered skills')
fig.tight_layout(rect=(0, .06, 1, 1))
for suffix in ('png', 'pdf'):
    fig.savefig(root / f'case-12006.{suffix}', dpi=180, bbox_inches='tight')
plt.close(fig)
