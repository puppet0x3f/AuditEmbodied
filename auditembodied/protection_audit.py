"""Separate shared physical guards from local role obligations in saved rollouts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COHORTS = {
    'scope': 'outputs/agentchord-style-20261109/trials.jsonl',
    'supervision': 'outputs/supervision-controls-20261109/closed-loop/trials.jsonl',
}
GRACE_MS = (0, 50, 100)


def longest_run(flags, keys=None):
    """Count consecutive sampled violations, breaking at an obligation change."""
    run = maximum = 0
    previous = None
    for index, flag in enumerate(flags):
        key = keys[index] if keys is not None else 0
        run = (run + 1 if key == previous else 1) if flag else 0
        maximum = max(maximum, run)
        previous = key
    return maximum


def role_signal(row):
    flags, keys = [], []
    for frame in row['trace']:
        skill = frame.get('skill', row['candidate'])
        keeper = 1 if skill == 'R_left' else 0 if skill == 'R_right' else None
        applicable = frame['phase'] == 'recovery' and keeper is not None
        keys.append(keeper if applicable else None)
        flags.append(bool(applicable and not frame['evaluator']['holds'][keeper]))
    return flags, keys


def analyze_episode(row, cohort):
    frames = [f for f in row['trace'] if f['phase'] in ('recovery', 'final_hold')]
    assert frames
    deltas = np.diff([row['prefix_state']['time']] + [f['time'] for f in frames])
    assert np.allclose(deltas, .05, atol=1e-8), 'Expected native20Hz trace'
    flags, keys = role_signal(row)
    keeper_steps = sum(flags)
    keeper_run = longest_run(flags, keys)
    unsupported = [not (any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames]
    unsupported_run = longest_run(unsupported)
    assert keeper_steps == row['protection']['keeper_grasp_lost_steps']
    assert abs(unsupported_run / 20 - row['protection']['max_unsupported_seconds']) < 1e-8
    collision_steps = row['protection']['deep_interarm_contact_steps']
    physical_bad = bool(unsupported_run > 2 or collision_steps)
    hard_bad = bool(row['blocked_calls'] or row.get('failure_reason') == 'numerical_instability')
    original_bad = bool(physical_bad or keeper_steps or hard_bad)
    assert original_bad == row['protection_violation'], 'Original strict verdict must reproduce'
    center = np.array([f['evaluator']['pot_center'] for f in frames])
    initial_center = np.array(row['post_fault']['pot_center'])
    local_bad = [f for f, flag in zip(row['trace'], flags) if flag]
    starts = next((f for f in row['trace'] if f['phase'] == 'recovery'), None)
    initial_skill = starts.get('skill', row['candidate'])
    initial_keeper = 1 if initial_skill == 'R_left' else 0 if initial_skill == 'R_right' else None
    valid_start = initial_keeper is None or row['post_fault']['holds'][initial_keeper]
    assert valid_start, 'Keep invalid starts separate from later role violations'
    result = {
        'cohort': cohort, 'split': 'healthy' if row['scenario'] == 'sham' else 'fault',
        'seed': row['seed'], 'scenario': row['scenario'], 'method': row['method'],
        'task_complete': bool(row['task_complete']), 'accepted_completion': bool(row['success']),
        'strict_safe': bool(row['success'] and not original_bad),
        'shared_physical_violation': physical_bad, 'role_violation': bool(keeper_steps),
        'role_only_violation': bool(keeper_steps and not physical_bad and not hard_bad),
        'hard_failure': hard_bad, 'valid_local_start': bool(valid_start),
        'keeper_lost_samples': keeper_steps, 'max_keeper_gap_ms': keeper_run * 50,
        'max_unsupported_ms': unsupported_run * 50,
        'deep_collision_samples': collision_steps,
        'first_role_loss_stage': local_bad[0]['stage'] if local_bad else None,
        'first_role_loss_seconds': local_bad[0]['time'] - row['prefix_state']['time'] if local_bad else None,
        'first_role_loss_other_arm_holds': bool(local_bad[0]['evaluator']['holds'][1-keys[row['trace'].index(local_bad[0])]]) if local_bad else None,
        'first_role_loss_table_contact': bool(local_bad[0]['evaluator']['table_contact']) if local_bad else None,
        'maximum_tilt_deg': max(f['evaluator']['tilt_deg'] for f in frames),
        'maximum_downward_displacement_m': max(0., float(initial_center[2] - center[:, 2].min())),
        'maximum_center_displacement_m': float(np.linalg.norm(center - initial_center, axis=1).max()),
        'minimum_clearance_m': min(f['evaluator']['clearance'] for f in frames),
        'role_loss_maximum_tilt_deg': max((f['evaluator']['tilt_deg'] for f in local_bad), default=None),
        'role_loss_minimum_clearance_m': min((f['evaluator']['clearance'] for f in local_bad), default=None),
        'role_loss_maximum_downward_displacement_m': max((max(0.,initial_center[2]-f['evaluator']['pot_center'][2]) for f in local_bad), default=None),
        'physical_guard_only_completion': bool(row['success'] and not physical_bad and not hard_bad),
        'total_seconds': row['total_post_fault_seconds'],
    }
    for grace in GRACE_MS:
        result[f'protected_completion_grace{grace}ms'] = bool(row['success'] and not physical_bad and not hard_bad and keeper_run * 50 <= grace)
    return result


def aggregate(rows):
    return {
        'episodes': len(rows), 'terminal_completions': sum(r['task_complete'] for r in rows),
        'strict_safe_completions': sum(r['strict_safe'] for r in rows),
        'shared_physical_violations': sum(r['shared_physical_violation'] for r in rows),
        'role_violations': sum(r['role_violation'] for r in rows),
        'role_only_violations': sum(r['role_only_violation'] for r in rows),
        'physical_guard_only_completions': sum(r['physical_guard_only_completion'] for r in rows),
        'deep_collision_episodes': sum(bool(r['deep_collision_samples']) for r in rows),
        'single_sample_role_episodes': sum(r['max_keeper_gap_ms'] == 50 for r in rows),
        'keeper_gap_max_ms': max((r['max_keeper_gap_ms'] for r in rows), default=0),
        'unsupported_max_ms': max((r['max_unsupported_ms'] for r in rows), default=0),
        'grace_safe_completions': {str(g): sum(r[f'protected_completion_grace{g}ms'] for r in rows) for g in GRACE_MS},
    }


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(root, output):
    protocol = json.loads((output/'protocol.json').read_text())
    for name, digest in protocol['frozen_original_files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    results = []
    counts = {}
    for cohort, relative in COHORTS.items():
        with (root/relative).open() as stream:
            source = [json.loads(line) for line in stream]
        counts[cohort] = len(source)
        results.extend(analyze_episode(row, cohort) for row in source)
    assert counts == {'scope': 300, 'supervision': 160}
    groups = {}
    for cohort in COHORTS:
        for split in ('fault', 'healthy'):
            subset = [r for r in results if r['cohort'] == cohort and r['split'] == split]
            if subset:
                groups[f'{cohort}_{split}'] = {m: aggregate([r for r in subset if r['method'] == m]) for m in sorted({r['method'] for r in subset})}
    report = {
        'episodes': len(results), 'cohorts': counts, 'all_original_strict_labels_reproduced': True,
        'all_original_unsupported_and_keeper_counters_reproduced': True,
        'role_grace_ms': list(GRACE_MS), 'shared_unsupported_limit_ms_unchanged': 100,
        'shared_interarm_penetration_limit_m_unchanged': .01,
        'methods': groups,
        'violating_episodes': [r for r in results if r['shared_physical_violation'] or r['role_violation'] or r['hard_failure']],
        'interpretation': [
            'Physical guards cover observed unsupported intervals and deep inter-arm penetration, not every possible physical hazard.',
            'Unsupported means absence of both registered two-finger grasps and table contact. It does not establish absence of all object contact forces or free flight.',
            'Role obligations apply to active local recovery only, matching each frozen executor; completed violations persist across switches.',
            'Global table-cycle verification stays in accepted_completion. Intentional setdown motion is not classified as a fall by displacement alone.',
            'Grace0 is the unchanged primary metric. Grace50/100 are separate retrospective sensitivity analyses, not new policy results.',
            'Durations are sampled-run estimates. Existing20Hz data cannot establish sub-frame contact events or continuous-time safety.',
            'Original20Hz collision counters are preserved; raw inter-arm contact depths were not stored and cannot be retrospectively rethresholded.',
        ],
    }
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    write_csv(output/'episodes.csv', results)
    write_csv(output/'severity.csv', report['violating_episodes'])
    summary = [{'group': group, 'method': method, **{k:v for k,v in data.items() if not isinstance(v,dict)},
                **{f'grace{g}ms':data['grace_safe_completions'][str(g)] for g in GRACE_MS}}
               for group, methods in groups.items() for method, data in methods.items()]
    write_csv(output/'summary.csv', summary)
    print(json.dumps({'episodes':len(results),'methods':groups}), flush=True)


def plot(output, paper):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads((output/'report.json').read_text())
    names = {'protected':'Protected rule','gated_direct':'MLP + start gate','agentchord_style':'AgentChord-style',
             'global':'Global restart','fixed60':'Fixed60','ce':'Plain SFT','terminal':'Terminal only','joint':'Joint + set'}
    fig, axes = plt.subplots(1,2,figsize=(7.1,2.65))
    for axis, group, order, title in zip(axes, ('scope_fault','supervision_fault'),
            (('protected','gated_direct','agentchord_style','global'),('fixed60','ce','terminal','joint')),
            ('Scope:60 fault states per policy','Supervision:40 fault states per policy')):
        for i, method in enumerate(order):
            values = report['methods'][group][method]
            yy = [values['grace_safe_completions'][str(g)] for g in GRACE_MS]
            axis.plot(GRACE_MS, yy, marker=['o','s','^','D'][i], linestyle=['-','--',':','-.'][i],
                      label=names[method],alpha=.85,linewidth=1.8)
        axis.set(title=title, xlabel='Allowed consecutive keeper gap (ms)', xticks=GRACE_MS)
        n=report['methods'][group][order[0]]['episodes']
        values=[v['grace_safe_completions'][str(g)] for v in report['methods'][group].values() for g in GRACE_MS]
        axis.set(ylim=(min(values)-1,max(values)+1),ylabel=f'Protected completions / {n}')
        axis.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axis.grid(alpha=.2);axis.legend(fontsize=7,loc='lower right')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(output/f'duration-sensitivity.{suffix}',dpi=200,bbox_inches='tight')
        fig.savefig(paper/'figures'/f'protection-sensitivity.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & $S$ fail & $O$ only & $Y_0$ & $Y_{50,100}$ \\',r'\midrule']
    for group, order in [('scope_fault',('protected','gated_direct','agentchord_style','global')),
                         ('supervision_fault',('fixed60','ce','terminal','joint'))]:
        lines.append(r'\multicolumn{5}{l}{\textit{'+('Scope: 60 faults per policy' if group=='scope_fault' else 'Supervision: 40 faults per policy')+r'}}\\')
        for method in order:
            v=report['methods'][group][method]
            assert v['grace_safe_completions']['50']==v['grace_safe_completions']['100']
            lines.append(f"{names[method]} & {v['shared_physical_violations']} & {v['role_only_violations']} & {v['grace_safe_completions']['0']} & {v['grace_safe_completions']['50']} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (paper/'tables'/'protection-validity.tex').write_text('\n'.join(lines)+'\n')


def frequency(output):
    protocol=json.loads((output/'protocol.json').read_text())
    results=[];flat=[]
    for case in protocol['frequency_diagnostic']['cases']:
        name='-'.join(str(case[k]) for k in ('cohort','seed','scenario','method'))
        meta=json.loads((output/'frequency'/f'{name}.json').read_text())
        assert meta['all_20hz_frames_match'] and not meta['error']
        with np.load(output/'frequency'/f'{name}.npz') as archive:
            data={k:archive[k] for k in archive.files}
        assert np.allclose(np.diff(data['time']),.002,atol=1e-9)
        keeper=data['keeper'];holds=data['holds']
        role=np.array([k>=0 and not h[k] for k,h in zip(keeper,holds)])
        unsupported=~(holds.any(axis=1) | data['table_contact'])
        def measure(indices,hz):
            k=keeper[indices].tolist()
            role_ms=longest_run(role[indices].tolist(),k)*1000/hz
            unsupported_ms=longest_run(unsupported[indices].tolist())*1000/hz
            collision=bool(np.any(data['interarm_penetration_m'][indices]>.01))
            physical=unsupported_ms>100+1e-8 or collision
            return {'keeper_gap_ms':role_ms,'unsupported_ms':unsupported_ms,'physical_violation':physical,
                    'protected_completions':{str(g):bool(meta['accepted_completion'] and not physical and role_ms<=g+1e-8) for g in GRACE_MS}}
        native=measure(np.arange(len(keeper)),500)
        original=measure(np.arange(24,len(keeper),25),20)
        assert original['protected_completions']['0']==meta['original_safe']
        assert abs(original['unsupported_ms']-1000*meta['original_protection']['max_unsupported_seconds'])<1e-8
        sweeps={}
        for hz in (500,100,50,20):
            stride=500//hz
            phases=[measure(np.arange(offset,len(keeper),stride),hz) for offset in range(stride)]
            sweep={'phase_count':stride,'minimum_keeper_gap_ms':min(p['keeper_gap_ms'] for p in phases),
                'maximum_keeper_gap_ms':max(p['keeper_gap_ms'] for p in phases),
                'minimum_unsupported_ms':min(p['unsupported_ms'] for p in phases),
                'maximum_unsupported_ms':max(p['unsupported_ms'] for p in phases),
                'phase_protected_completions':{str(g):sum(p['protected_completions'][str(g)] for p in phases) for g in GRACE_MS}}
            sweeps[str(hz)]=sweep
            flat.append({'case':name,'hz':hz,**{k:v for k,v in sweep.items() if not isinstance(v,dict)},
                         **{f'protected_phases_grace{g}ms':sweep['phase_protected_completions'][str(g)] for g in GRACE_MS}})
        results.append({**case,'case':name,'native500hz':native,'original20hz':original,'phase_sweeps':sweeps,
                        'all_original_frames_match':True,'high_frequency_samples':len(keeper)})
    report={'diagnostic_only':True,'cases':results,'physical_replays':len(results),
            'high_frequency_samples':sum(r['high_frequency_samples'] for r in results),
            'all_original20hz_frames_match':True,'original_completion_predicate_retained':True,
            'interpretation':'Native physics observations with an unchanged20Hz controller. Every grid phase is swept deterministically; phase counts are not independent episodes or probability estimates. Known cases are not a population prevalence sample.'}
    (output/'frequency-report.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'frequency-sensitivity.csv',flat)
    print(json.dumps(report),flush=True)


def contacts(output):
    protocol=json.loads((output/'contact-protocol.json').read_text());results=[]
    for case in protocol['cases']:
        name='-'.join(str(case[k]) for k in ('cohort','seed','scenario','method'))
        meta=json.loads((output/'contact-diagnostic'/f'{name}.json').read_text())
        assert meta['all_20hz_frames_match'] and not meta['error']
        with np.load(output/'contact-diagnostic'/f'{name}.npz') as d:
            lacking=~(d['holds'].any(axis=1)|d['table_contact'])
            unloaded=d['active_pot_contacts']==0
            force=d['pot_contact_force_norm_sum'];center=d['pot_center']
            results.append({**case,'case':name,'samples':len(lacking),
                'registered_support_absent_samples':int(lacking.sum()),
                'registered_support_absent_with_force_contact_samples':int(np.sum(lacking & ~unloaded)),
                'minimum_object_contact_force_norm_sum_during_registered_absence_N':float(force[lacking].min()) if lacking.any() else None,
                'maximum_no_force_contact_run_ms':longest_run(unloaded.tolist())*2,
                'registered_absence_maximum_tilt_deg':float(d['tilt_deg'][lacking].max()) if lacking.any() else None,
                'registered_absence_downward_displacement_from_first_recovery_sample_m':max(0.,float(center[0,2]-center[lacking,2].min())) if lacking.any() else None,
                'all_original20hz_observations_match':True})
    report={'diagnostic_only':True,'completed_contact_recording_replays':len(results),
        'force_contact_threshold_N':protocol['contact_force_norm_threshold_N'],'cases':results,
        'conclusion':'Every recorded sample lacking registered grasp/table support still has force-bearing object contact. The frozen unsupported guard is a registered-support criterion, not literal contact-free flight. Force-norm sums do not certify net vertical load-bearing or secure grasp.'}
    (output/'contact-report.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'contact-consequences.csv',results)
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--plot',action='store_true')
    parser.add_argument('--frequency',action='store_true')
    parser.add_argument('--contacts',action='store_true')
    args=parser.parse_args()
    if args.contacts:contacts(args.output_dir)
    elif args.frequency:frequency(args.output_dir)
    elif args.plot:plot(args.output_dir,args.root/'paper')
    else:analyze(args.root,args.output_dir)


if __name__=='__main__':main()
