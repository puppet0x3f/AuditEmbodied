"""Scientific figures from the recorded mechanism and geometry results."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'reproducibility'))
import mechanism_geometry_audit as audit

COLORS = {'registered': '#bd593b', 'measured_position': '#337b8d'}
NAMES = {'registered': 'Registered', 'measured_position': 'Position only'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'axes.titlesize': 9})


def save(fig, stem):
    fig.tight_layout(pad=.8)
    for suffix in ('pdf', 'png'):
        fig.savefig(HERE / f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def mechanism():
    report = json.loads((HERE / 'mechanism-analysis/report.json').read_text())
    rows = [json.loads(s) for s in (HERE / 'mechanism-analysis/cases.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.5))
    ax = axes[0, 0]
    for j, hold in enumerate(('registered', 'measured_position')):
        cells = [next(c for c in report['cells'] if c['hold_mode'] == hold and c['wait_steps'] == w
                      and not c['healthy']) for w in (1, 20)]
        values = [c['metrics']['arrived'] for c in cells]
        means = np.array([v['estimate'] for v in values]) * 100
        intervals = np.array([v['ci95'] for v in values]) * 100
        x = np.arange(2) + (j - .5) * .22
        ax.errorbar(x, means, yerr=[means - intervals[:, 0], intervals[:, 1] - means],
                    color=COLORS[hold], marker='o', capsize=3, label=NAMES[hold])
    ax.set(xticks=[0, 1], xticklabels=['0.05 s', '1.00 s'], ylim=(-4, 104),
           ylabel='First-approach arrival (%)', title='(a) Both-arm arrival before closing')
    ax.legend(frameon=False, fontsize=7)
    ax = axes[0, 1]
    for hold in ('registered', 'measured_position'):
        group = [r for r in rows if r['hold_mode'] == hold and r['wait_steps'] == 20 and r['scenario'] != 'sham']
        ax.scatter([r['free_nearest_limit_rad'] for r in group], [r['free_position_error_mm'] for r in group],
                   color=COLORS[hold], s=14, alpha=.7, label=NAMES[hold])
    ax.axvline(.02, ls=':', color='.5', lw=.9)
    ax.set(xlabel='Free-arm nearest joint-limit margin (rad)', ylabel='Pre-close tracking error (mm)',
           title='(b) All 40 long-wait faults per hold')
    selected = []
    with gzip.open(HERE / 'mechanism-analysis/recorded-inputs.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line)
            if r['seed'] == 72000 and r['scenario'] == 'left_loss' and r['wait_steps'] == 20:
                selected.append(r)
    assert len(selected) == 2
    for r in selected:
        frames = r['frames']
        first = next(i for i, f in enumerate(frames) if f['stage'] == 'global_close')
        close_entry = frames[first - 1]['time']
        start = first - 1
        while start > 0 and frames[start]['stage'] == 'global_arrival_settle':
            start -= 1
        chosen = frames[start:]
        times = [f['time'] - close_entry for f in chosen]
        target = np.asarray(r['target'])[0]
        errors = [np.linalg.norm(np.asarray(f['robot']['positions'])[0] - target) * 1000 for f in chosen]
        joints = {round(d['time'], 8): d for d in r['joints']}
        margins = [audit.joint_state(joints[round(f['time'], 8)], 0)['joint7_limit_rad'] for f in chosen]
        color, label = COLORS[r['hold_mode']], NAMES[r['hold_mode']]
        axes[1, 0].plot(times, errors, color=color, label=label)
        axes[1, 1].plot(times, margins, color=color, label=label)
    for ax in axes[1]:
        ax.axvline(0, ls=':', color='.5', lw=.9)
        ax.set_xlabel('Time relative to closing entry (s)')
    axes[1, 0].set(ylabel='Free-arm tracking error (mm)', title='(c) Fixed-target settling and closing')
    axes[1, 1].set(ylabel='Free-arm joint 7 limit margin (rad)', title='(d) Same paired execution, seed 72000 / left')
    axes[1, 0].legend(frameon=False, fontsize=7)
    save(fig, 'arrival-mechanism')


def geometry():
    path = HERE / 'geometry-analysis/scored.jsonl'
    if not path.exists():
        return
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.55))
    for ax, method, title in zip(axes[:2], ('retarget', 'closed_loop'), ('Target', 'CAD feedback')):
        for hold in ('registered', 'measured_position'):
            estimates = []
            for wait in (1, 20):
                group = [r for r in rows if r['method'] == method and r['hold_mode'] == hold
                         and r['wait_steps'] == wait and r['scenario'] != 'sham']
                estimates.append(audit.seed_mean(group, 'native_Y'))
            means = np.array([r['estimate'] for r in estimates]) * 100
            bounds = np.array([r['ci95'] for r in estimates]) * 100
            ax.errorbar([.05, 1], means, yerr=[means - bounds[:, 0], bounds[:, 1] - means],
                        color=COLORS[hold], marker='o', capsize=3, label=NAMES[hold])
        ax.set(title=title, xlabel='Total initial holding (s)', xticks=[.05, 1], ylim=(-3, 104))
    axes[0].set_ylabel('Protected completion (%)')
    axes[1].legend(frameon=False, fontsize=7)
    report = json.loads((HERE / 'geometry-analysis/report.json').read_text())
    for i, method in enumerate(('retarget', 'closed_loop')):
        result = report['contrasts'][method + '_hold_wait']['native_Y']
        mean = result['estimate'] * 100
        low, high = np.asarray(result['ci95']) * 100
        axes[2].errorbar(mean, i, xerr=[[mean-low], [high-mean]], color='#34495e', marker='o', capsize=3)
    axes[2].axvline(0, color='.5', ls=':', lw=.9)
    axes[2].set(yticks=[0, 1], yticklabels=['Target', 'CAD'], xlabel='Holding × waiting interaction (pp)',
                title='20 whole-seed paired intervals', ylim=(-.6, 1.6))
    save(fig, 'geometry-holding')


if __name__ == '__main__':
    mechanism()
    geometry()
