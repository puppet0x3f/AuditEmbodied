"""Released native-policy recovery after an explicit receiver grasp-loss intervention."""
from collections import defaultdict
import numpy as np
from ranking_closure_audit import interval


def score(row):
    records=row['records'];injection=row['intervention_records']
    assert len(row['actions'])==len(records)==row['active_policy_steps']==700
    assert row['state_edits_after_reset']==0
    triggered=bool(row['event_triggered'])
    if triggered:
        assert not any(f['C'] for f in records if f['policy_step']<=row['event_active_step']), 'Task already complete before intervention'
        fault=[f for f in injection if f['phase']==row['scenario']]
        wait=[f for f in injection if f['phase']=='wait']
        assert len(fault)==12 and len(wait)==row['wait_steps']
        holds=fault[-1]['payload_grasps']
        valid=bool(holds[0] and (not holds[1] if row['scenario']=='fault' else holds[1]))
        assert valid==row['fault_valid']
        assert all(f['action'][6]==1. for f in injection)
        assert all(f['action'][13]==(-1. if row['scenario']=='fault' else 1.) for f in injection)
        native=any(f['C'] for f in records+injection)
    else:
        assert not injection;valid=False;native=any(f['C'] for f in records)
    for f in records+injection:
        assert f['C']==bool(f['payload_in_target_bin'] and f['trash_in_trash_bin'])
    assert row['C_native']==native and row['C']==bool(native and triggered and valid)
    seconds=.05*(700+len(injection))
    assert abs(row['simulated_seconds']-seconds)<1e-7
    assert abs(row['charged_seconds']-seconds-sum(row['policy_query_seconds'])-row['observation_service_seconds'])<1e-7
    result={k:row[k] for k in ('seed','scenario','hold_mode','wait_steps','C','C_native','event_triggered','fault_valid',
        'event_active_step','charged_seconds','simulated_seconds','query_seconds','observation_service_seconds','discarded_chunk_actions')}
    result['entry_max_reference_error_mm']=float(np.linalg.norm(
        np.asarray(row['holding_input']['registered_positions'])-row['holding_input']['measured_positions'],axis=1).max()*1000) if triggered else None
    return result


def pairing(rows):
    before=defaultdict(list);faults=defaultdict(list)
    for r in rows:
        before[r['seed']].append((r['initial_state'],r['fault_prefix'],r['event_active_step']))
        faults[r['seed'],r['scenario']].append(r['fault_end'])
    for key,g in before.items():assert all(r==g[0] for r in g),key
    for key,g in faults.items():assert all(r==g[0] for r in g),key
    return {'matched_initial_seed_groups':len(before),
            'matched_pre_fault_seed_groups':sum(g[0][1] is not None for g in before.values()),
            'attempted_post_fault_groups':len(faults),
            'matched_post_fault_hold_wait_groups':sum(g[0] is not None for g in faults.values())}


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['scenario'],r['hold_mode'],r['wait_steps']].append(r)
    cells=[]
    for (scenario,hold,wait),g in sorted(groups.items()):
        offsets=[r['entry_max_reference_error_mm'] for r in g if r['entry_max_reference_error_mm'] is not None]
        cells.append({'scenario':scenario,'hold_mode':hold,'wait_steps':wait,'episodes':len(g),
            **{k:sum(bool(r[k]) for r in g) for k in ('C','C_native','event_triggered','fault_valid')},
            **{k:float(np.mean([r[k] for r in g])) for k in ('charged_seconds','simulated_seconds','query_seconds','observation_service_seconds')},
            'event_active_steps':[r['event_active_step'] for r in sorted(g,key=lambda x:x['seed'])],
            'entry_max_reference_error_mm':float(np.mean(offsets)) if offsets else None})
    effects={};timing={}
    for scenario in ('fault','sham'):
        grouped=defaultdict(dict)
        for r in rows:
            if r['scenario']==scenario:grouped[r['seed']][r['hold_mode'],r['wait_steps']]=r
        for field in ('C','charged_seconds'):
            values=[float(g['measured_position',20][field])-float(g['registered',20][field])-
                    float(g['measured_position',1][field])+float(g['registered',1][field]) for s,g in sorted(grouped.items())]
            gains=[float(g['measured_position',20][field])-float(g['registered',20][field]) for s,g in sorted(grouped.items())]
            if field=='C':
                effects[f'{scenario}_{field}_hold_wait']=interval(values)
                effects[f'{scenario}_{field}_long_gain']=interval(gains)
            else:
                timing[f'{scenario}_{field}_hold_wait']=float(np.mean(values))
                timing[f'{scenario}_{field}_long_gain']=float(np.mean(gains))
    return {'formal_executions':len(rows),'independent_seeds':len({r['seed'] for r in rows}),
            'conditions':cells,'effects':effects,'descriptive_batch_timing_contrasts':timing,
            'timing_uncertainty_note':'One vectorized policy batch per condition. Shared batch query latency is not 20 independent timing observations; no seed-bootstrap interval or deployment-efficiency claim is made for these costs.',
            'native_completion_is_not_lifting_Y500':True,
            'statistics':'10000 complete-seed resamples, random seed 20261122; all conditions retained.'}


def table(report):
    lines=[r'Condition & Hold & Wait & Triggered & Valid & $R$ & Native $C$ & Sim. (s) & Native time (s)\\',r'\midrule']
    for c in report['conditions']:
        lines.append(f"{c['scenario'].capitalize()} & {'Registered' if c['hold_mode']=='registered' else 'Position only'} & {c['wait_steps']*.05:.2f} & {c['event_triggered']}/20 & {c['fault_valid']}/20 & {c['C']}/20 & {c['C_native']}/20 & {c['simulated_seconds']:.3f} & {c['charged_seconds']:.3f}"+r'\\')
    return '\n'.join([r'\begin{tabular}{lllrrrrrr}',r'\toprule',*lines,r'\bottomrule',r'\end{tabular}'])+'\n'
