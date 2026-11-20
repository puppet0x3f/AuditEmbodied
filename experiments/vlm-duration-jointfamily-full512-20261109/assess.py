"""Final-checkpoint candidate selection and actual trajectory costs, including waiting."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from auditembodied.vlm_duration_lora import joint_marginal_metrics, start_key, timing_contrasts


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def load_physics(plan):
    physical = {}
    for wait in (0, 66):
        for row in read(Path(plan['source_collection']) / f'trials-wait{wait}.jsonl'):
            n = row['local_steps']['local_relevel_steps']
            assert not row.get('error')
            assert row['state_edit_count_after_initialization'] == row['contact_anchor_count'] == row['external_wrench_at_end'] == 0
            assert len([f for f in row['trace'] if f['stage'] == 'local_relevel']) == n
            assert len([f for f in row['trace'] if f['stage'] == 'duration_budget_wait']) == wait
            initial = row['prefix_state']['time']
            prefix = row['stage_capture']['h2']['state']['time'] - initial + wait / 20
            total = row['total_post_fault_seconds']
            assert abs(total - row['inference_wait_seconds'] - row['skill_seconds'] - row['hold_seconds']) < 1e-7
            assert abs(row['inference_wait_seconds'] - (19 + wait) / 20) < 1e-7
            assert abs(total - prefix - n / 20 - .4 - row['hold_seconds']) < 1e-7
            label = ('B' if row['protection_violation'] else 'A') if row['success'] else ('D' if row['protection_violation'] else 'C')
            key = row['seed'], row['scenario'], wait, n
            physical[key] = {'label': label, 'post_fault_seconds': total,
                'episode_simulated_seconds': row['episode_simulated_seconds'],
                'wait_seconds': row['inference_wait_seconds'], 'skill_seconds': row['skill_seconds'],
                'final_hold_seconds': row['hold_seconds'], 'decision_prefix_seconds': prefix,
                'path_m': row['recovery_path_m']}
    assert len(physical) == 96
    for seed in range(40, 48):
        for scenario in ('left_loss', 'right_loss'):
            for wait in (0, 66):
                costs = [physical[seed, scenario, wait, n] for n in (30, 40, 60)]
                assert max(c['decision_prefix_seconds'] for c in costs) - min(c['decision_prefix_seconds'] for c in costs) < 1e-7
    return physical


def selection_metrics(rows, threshold, physical, fixed_n=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row['seed'], row['scenario'], row['wait_steps']].append(row)
    cases = []
    for (seed, scenario, wait), group in groups.items():
        assert {r['plan']['local_relevel_steps'] for r in group} == {30, 40, 60}
        ordered = sorted(group, key=lambda r: r['plan']['local_relevel_steps'])
        admitted = [r for r in ordered if r['class_probabilities'][0] >= threshold] if fixed_n is None else [r for r in ordered if r['plan']['local_relevel_steps'] == fixed_n]
        for row in group:
            assert physical[seed, scenario, wait, row['plan']['local_relevel_steps']]['label'] == row['label']
        chosen = admitted[0] if admitted else None
        actual = physical[seed, scenario, wait, chosen['plan']['local_relevel_steps']] if chosen else None
        prefix = physical[seed, scenario, wait, 30]['decision_prefix_seconds']
        cases.append({'seed': seed, 'scenario': scenario, 'W_steps': wait,
            'has_safe_candidate': any(r['label'] == 'A' for r in group),
            'safe_covered': any(r['label'] == 'A' for r in admitted),
            'refused': chosen is None,
            'selected_steps': chosen['plan']['local_relevel_steps'] if chosen else None,
            'selected_label': chosen['label'] if chosen else None,
            'safe_selected': bool(chosen and chosen['label'] == 'A'),
            'process_violation_selected': bool(chosen and chosen['label'] in 'BD'),
            'incomplete_selected': bool(chosen and chosen['label'] in 'CD'),
            'decision_prefix_seconds': prefix,
            'selected_cost': actual,
            'known_post_fault_cost_seconds': actual['post_fault_seconds'] if chosen else prefix,
            'terminal_cost_observed': chosen is not None})
    selected = [r for r in cases if not r['refused']]
    safe = [r for r in selected if r['safe_selected']]
    refused = [r for r in cases if r['refused']]
    no_safe = [r for r in cases if not r['has_safe_candidate']]
    admitted_rows = [r for r in rows if r['class_probabilities'][0] >= threshold] if fixed_n is None else [r for r in rows if r['plan']['local_relevel_steps'] == fixed_n]
    return {'examples': len(rows), 'families': len(cases), 'class_counts': dict(Counter(r['label'] for r in rows)),
        'safe_candidates': sum(r['label'] == 'A' for r in rows),
        'safe_admitted_candidates': sum(r['label'] == 'A' for r in admitted_rows),
        'false_admitted_candidates': sum(r['label'] != 'A' for r in admitted_rows),
        'recoverable_families': sum(r['has_safe_candidate'] for r in cases),
        'safe_covered_families': sum(r['safe_covered'] for r in cases),
        'selected_families': len(selected), 'safe_selected_families': len(safe),
        'non_A_selected_families': len(selected) - len(safe),
        'task_completed_selected_families': sum(r['selected_label'] in 'AB' for r in selected),
        'process_violating_selected_families': sum(r['process_violation_selected'] for r in selected),
        'incomplete_selected_families': sum(r['incomplete_selected'] for r in selected),
        'refused_families': len(refused), 'no_safe_families': len(no_safe),
        'no_safe_families_rejected': sum(r['refused'] for r in no_safe) if no_safe else None,
        'native_correct': sum(r['conditional_prediction'] == r['label'] for r in rows) if fixed_n is None else None,
        'marginals': joint_marginal_metrics(rows) if fixed_n is None else None,
        'costs': {'selected_rollouts': len(selected), 'safe_selected_rollouts': len(safe), 'refused_states': len(refused),
            'mean_selected_post_fault_seconds': mean(r['selected_cost']['post_fault_seconds'] for r in selected),
            'mean_safe_selected_post_fault_seconds': mean(r['selected_cost']['post_fault_seconds'] for r in safe),
            'mean_selected_whole_episode_seconds': mean(r['selected_cost']['episode_simulated_seconds'] for r in selected),
            'mean_selected_wait_seconds': mean(r['selected_cost']['wait_seconds'] for r in selected),
            'mean_selected_skill_seconds': mean(r['selected_cost']['skill_seconds'] for r in selected),
            'mean_selected_final_hold_seconds': mean(r['selected_cost']['final_hold_seconds'] for r in selected),
            'mean_selected_path_m': mean(r['selected_cost']['path_m'] for r in selected),
            'mean_refused_observed_prefix_seconds': mean(r['decision_prefix_seconds'] for r in refused),
            'mean_all_state_observed_cost_lower_bound_seconds': mean(r['known_post_fault_cost_seconds'] for r in cases),
            'all_terminal_costs_observed': not refused,
            'note': 'Selected costs are actual complete post-fault trajectories including initial wait, planned inference hold, all motion and final verification. Refusals have only a physically observed common-prefix cost; their subsequent holding/termination was not collected and is not assigned zero.'},
        'timing_40_60': timing_contrasts([{**r, 'prediction': r['conditional_prediction']} for r in rows]) if fixed_n is None else None,
        'family_cases': cases}


def paired_costs(reference, candidate):
    index = lambda m: {(r['seed'], r['scenario'], r['W_steps']): r for r in m['family_cases']}
    old, new = index(reference), index(candidate)
    common = [k for k in old.keys() & new.keys() if old[k]['selected_cost'] is not None and new[k]['selected_cost'] is not None]
    safe_common = [k for k in common if old[k]['safe_selected'] and new[k]['safe_selected']]
    def summarize(keys):
        return {'families': len(keys),
            'reference_mean_post_fault_seconds': mean(old[k]['selected_cost']['post_fault_seconds'] for k in keys),
            'new_mean_post_fault_seconds': mean(new[k]['selected_cost']['post_fault_seconds'] for k in keys),
            'new_minus_reference_seconds': mean(new[k]['selected_cost']['post_fault_seconds'] - old[k]['selected_cost']['post_fault_seconds'] for k in keys)}
    return {'both_selected': summarize(common), 'both_safely_selected': summarize(safe_common)}


def deadline_accounting(serving, selection):
    """Charge every measured overrun at20Hz; never reuse post-W outcome labels after a late reply."""
    families = {(r['seed'], r['scenario']): r for r in selection['family_cases']}
    cases = []
    for timing in serving['families']:
        matrix = families[timing['seed'], timing['scenario']]
        excess = max(0., timing['pipeline_seconds'] - 3.3)
        extra_steps = math.ceil(excess * 20)
        late = extra_steps > 0
        selected = matrix['selected_steps'] is not None and not late
        # The existing executor discards late replies and skips relevel/settle,
        # then executes its fixed20-step final hold. Cost is a schedule calculation;
        # the extra-hold physical outcomes were not collected in this experiment.
        scheduled = matrix['selected_cost']['post_fault_seconds'] if selected else matrix['decision_prefix_seconds'] + extra_steps / 20 + 1.
        cases.append({'seed': timing['seed'], 'scenario': timing['scenario'],
            'pipeline_seconds': timing['pipeline_seconds'], 'measured_budget_overrun_seconds': excess,
            'charged_extra_wait_steps': extra_steps, 'charged_extra_wait_seconds': extra_steps / 20,
            'charged_total_inference_wait_seconds': .95 + 3.3 + extra_steps / 20,
            'late_reply_discarded': late, 'matrix_selected_steps': matrix['selected_steps'],
            'deadline_valid_selected_steps': matrix['selected_steps'] if selected else None,
            'deadline_valid_safe_coverage': matrix['safe_covered'] and not late,
            'deadline_valid_safe_selection': matrix['safe_selected'] and not late,
            'post_fault_seconds_by_controller_schedule': scheduled,
            'complete_physical_trajectory_observed': selected,
            'cost_source': 'Actual matched candidate trajectory' if selected else 'Known common prefix + charged extra wait + fixed1s final hold; computed schedule, not a new rollout'})
    return {'families': len(cases), 'deadline_misses': sum(r['late_reply_discarded'] for r in cases),
        'deadline_valid_selected_families': sum(r['deadline_valid_selected_steps'] is not None for r in cases),
        'deadline_valid_safe_covered_families': sum(r['deadline_valid_safe_coverage'] for r in cases),
        'deadline_valid_safe_selected_families': sum(r['deadline_valid_safe_selection'] for r in cases),
        'mean_measured_pipeline_seconds': mean(r['pipeline_seconds'] for r in cases),
        'mean_charged_extra_wait_seconds': mean(r['charged_extra_wait_seconds'] for r in cases),
        'mean_charged_total_inference_wait_seconds': mean(r['charged_total_inference_wait_seconds'] for r in cases),
        'mean_post_fault_seconds_by_controller_schedule': mean(r['post_fault_seconds_by_controller_schedule'] for r in cases),
        'cases': cases,
        'scope': 'Timing-admissible plans and control-time accounting under the unchanged deadline rule. After an overrun, do not add wait to the old relevel rollout and keep its outcome label. Computed abort timing is reported separately from measured complete physical trajectories.'}


def main():
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'experiment-plan.json').read_text())
    physical = load_physics(plan)
    result = json.loads((root / 'full512/result.json').read_text())
    protocol = json.loads((root / 'full512/protocol.json').read_text())
    training = read(root / 'full512/training.jsonl')
    assert len(training) == result['completed_steps'] == 512
    assert sum(len(r['example_indices']) for r in training) == 1024
    assert Counter(r['sample_kind'] for r in training) == Counter(duration_family=256, uniform_single=256)
    assert protocol['init_adapter'] == plan['initial_outcome_adapter'] and 'diagnostic128' not in protocol['init_adapter']
    assert protocol['temporal_mode'] == 'stage_joint' and protocol['process_loss_weight'] == .5
    assert protocol['fixed_admission_threshold'] == plan['threshold'] and protocol['final_checkpoint_evaluation_only']
    assert protocol['train_examples'] == 72 and protocol['development_examples'] == 24 and not protocol['train_only']
    assert not list((root / 'full512').glob('*before*'))
    assert result['frozen_start_unchanged'] and result['data_unchanged']
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_norm_before_clip']) for r in training)
    assert all('support_time_loss' not in r for r in training)
    frozen = json.loads((root / 'frozen-inputs-before.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == d for p, d in frozen.items())
    assert all(hashlib.sha256((root / 'dataset' / name).read_bytes()).hexdigest() == d for name, d in plan['data_sha256'].items())
    report = {'scope': plan['scope'], 'threshold': plan['threshold'], 'W_steps': 66, 'W_seconds': 3.3,
        'completed_steps': 512, 'candidate_presentations': 1024, 'training_accuracy_gate': None,
        'training_seconds': result['training_seconds'], 'peak_allocated_gpu_bytes': result['peak_allocated_gpu_bytes'],
        'new_physical_episodes': 0, 'original_models_start_threshold_and_data_unchanged': True,
        'variants': {}, 'fixed60_references': {}, 'paired_cost_comparisons': {}}
    directories = {'previous_aftergrasp': Path(plan['baseline_dir']), 'previous_time_auxiliary': Path(plan['previous_time_aux_dir']), 'joint_family': root / 'full512'}
    flat = []
    predictions = {}
    for split, filename in [('train', 'train-after-predictions.jsonl'), ('development', 'after-predictions.jsonl')]:
        examples = read(root / 'dataset' / (split + '.jsonl'))
        for name, directory in directories.items():
            rows = read(directory / filename)
            assert len(rows) == len(examples)
            for a, b in zip(rows, examples):
                assert all(a[k] == b[k] for k in ('seed', 'scenario', 'image', 'robot', 'plan', 'label', 'wait_steps'))
            predictions[name, split] = rows
            report['variants'].setdefault(name, {})[split] = {}
            for wait in ('all', 0, 66):
                subset = rows if wait == 'all' else [r for r in rows if r['wait_steps'] == wait]
                m = selection_metrics(subset, plan['threshold'], physical)
                report['variants'][name][split][f'W{wait}'] = m
                flat.append({'variant': name, 'split': split, 'W_steps': wait,
                    **{k: m[k] for k in ('families', 'recoverable_families', 'safe_covered_families', 'selected_families', 'safe_selected_families',
                        'non_A_selected_families', 'false_admitted_candidates', 'safe_admitted_candidates', 'no_safe_families', 'no_safe_families_rejected', 'native_correct')},
                    **{k: m['costs'][k] for k in ('mean_selected_post_fault_seconds', 'mean_safe_selected_post_fault_seconds', 'mean_selected_wait_seconds', 'mean_refused_observed_prefix_seconds')}})
        report['fixed60_references'][split] = {}
        for wait in (0, 66):
            rows = [r for r in predictions['joint_family', split] if r['wait_steps'] == wait]
            report['fixed60_references'][split][f'W{wait}'] = selection_metrics(rows, plan['threshold'], physical, fixed_n=60)
        report['paired_cost_comparisons'][split] = {}
        for wait in ('all', 0, 66):
            new = report['variants']['joint_family'][split][f'W{wait}']
            for name in ('previous_aftergrasp', 'previous_time_auxiliary'):
                old = report['variants'][name][split][f'W{wait}']
                report['paired_cost_comparisons'][split][f'{name}_W{wait}'] = paired_costs(old, new)
    report['start_probability_max_difference'] = max(abs(a['start_valid_score'] - b['start_valid_score'])
        for split in ('train', 'development') for a, b in zip(predictions['previous_aftergrasp', split], predictions['joint_family', split]))
    assert report['start_probability_max_difference'] == 0
    starts = json.loads((root / 'full512/frozen-start-scores.json').read_text())
    assert max(abs(r['start_valid_score'] - starts[start_key(r)]) for r in predictions['joint_family', 'train']) == 0
    report['serving_check'] = json.loads((root / 'serving-check.json').read_text())
    report['deadline_accounting'] = {'joint_family': deadline_accounting(report['serving_check'], report['variants']['joint_family']['development']['W66'])}
    if (root / 'serving-check-baseline.json').exists():
        report['serving_check_baseline_current_resource'] = json.loads((root / 'serving-check-baseline.json').read_text())
        report['deadline_accounting']['previous_aftergrasp'] = deadline_accounting(report['serving_check_baseline_current_resource'], report['variants']['previous_aftergrasp']['development']['W66'])
    report['planned_W_matrix_decisions_meet_measured_deadline'] = report['serving_check']['deadline_misses'] == 0
    report['timing_note'] = ('All final development requests fit the fixed W, so no additional inference waiting is omitted from the matched matrix cost.'
        if report['planned_W_matrix_decisions_meet_measured_deadline'] else
        'Late replies must be discarded by the existing executor. Matrix costs assume the planned W; physical outcomes and final costs after additional late waiting were not collected, so these are not validated closed-loop costs for late states.')
    report['budget_note'] = '1024 primary presentations; previous after-grasp527 primary, time-auxiliary527 primary+527 auxiliary. Different sequence lengths and exposure allocation; no equal-FLOPs claim.'
    report['fixed60_reference_note'] = plan['fixed60_reference']
    report['false_admissions_W66'] = [{k: r[k] for k in ('seed', 'scenario', 'plan', 'label', 'conditional_prediction', 'class_probabilities')}
        for r in predictions['joint_family', 'development'] if r['wait_steps'] == 66 and r['label'] != 'A' and r['class_probabilities'][0] >= plan['threshold']]
    old = report['variants']['previous_aftergrasp']['development']['W66']
    new = report['variants']['joint_family']['development']['W66']
    report['development_W66_comparison'] = {'safe_selection_before': old['safe_selected_families'], 'safe_selection_after': new['safe_selected_families'],
        'state_coverage_before': old['safe_covered_families'], 'state_coverage_after': new['safe_covered_families'],
        'false_admissions_before': old['false_admitted_candidates'], 'false_admissions_after': new['false_admitted_candidates'],
        'mean_selected_total_before_seconds': old['costs']['mean_selected_post_fault_seconds'],
        'mean_selected_total_after_seconds': new['costs']['mean_selected_post_fault_seconds'],
        'families': new['families'], 'development_no_safe_families': new['no_safe_families']}
    (root / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    with (root / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    decision = {'status': 'FULL512_AND_FINAL_TRAIN_DEVELOPMENT_EVALUATION_COMPLETED',
        'training_accuracy_gate': None, 'completed_steps': 512, 'fresh_pre_regression_initialization': True,
        'new_physical_episodes': 0, 'independent_closed_loop': False, 'checkpoint_promoted': False,
        'old_route_start_threshold_and_wait_retained': True, 'development_W66': report['development_W66_comparison'],
        'serving_deadline_misses': report['serving_check']['deadline_misses'],
        'deadline_accounting': report['deadline_accounting']['joint_family'],
        'assessment': str((root / 'assessment.json').resolve())}
    (root / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps({'comparison': report['development_W66_comparison'], 'training_seconds': report['training_seconds'],
        'serving_deadline_misses': report['serving_check']['deadline_misses']}, indent=2))


if __name__ == '__main__':
    main()
