"""Diagnostic physical continuations of the frozen family's thirteen N60 runs.

Replay actuator commands, including original waits, from a fresh initialization.
Never restore a MuJoCo state. No model inference, training or new test evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict,replace
from functools import partial
import json
import multiprocessing
from pathlib import Path
import time

from .l4_lift import LiftConfig,run_trial
from .duration_pilot import STATE_KEYS,file_hash


def boundary_prefix(source,boundary='relevel30'):
    if boundary=='search_exhausted':
        prefix=[f for f in source['trace'] if f['phase']!='final_hold']
        if (not prefix or prefix[-1]['stage'] not in ('local_arc_close','controller_compute_wait') or
                any(f['stage']=='local_relevel' for f in prefix)):
            raise ValueError('source did not exhaust its local search')
        return prefix
    indices=[i for i,f in enumerate(source['trace']) if f['stage']=='local_relevel']
    if boundary not in ('relevel30','relevel40'):raise ValueError('unknown replay boundary')
    step=40 if boundary=='relevel40' else 30
    if len(indices)<step:raise ValueError(f'source has not reached relevel{step}')
    saved=source.get('continuation_diagnostic',{}).get('boundary_trace_length')
    if saved is not None:
        prefix=source['trace'][:saved]
        if sum(f['stage']=='local_relevel' for f in prefix)==step:return prefix
    end=indices[step-1]+1
    while end<len(source['trace']) and source['trace'][end]['stage']=='controller_compute_wait':end+=1
    return source['trace'][:end]


def run_continuation(spec,transition='hard',account_control_latency=False,observer=None,
                     boundary='relevel30',pace_recovery=False,table_assist_template=None):
    source,branch=spec
    warmup_seconds=0.
    if account_control_latency:
        from robosuite.utils import transform_utils as transforms
        import numpy as np
        warmed=time.perf_counter()
        transforms.quat2axisangle(transforms.mat2quat(np.eye(3)))
        warmup_seconds=time.perf_counter()-warmed
    started=time.perf_counter()
    role=source.get('source_candidate',source['candidate']);prefix=boundary_prefix(source,boundary)
    scope_steps=sum(f['stage']=='inference_wait' for f in source['trace'])
    cfg=replace(LiftConfig(),local_relevel_steps=60,
        fault_open_steps=sum(f['stage']=='fault_open' for f in source['trace']),
        fault_withdraw_steps=sum(f['stage']=='fault_withdraw' for f in source['trace']))
    row=run_trial(source['seed'],source['scenario'],config=cfg,
        selector=lambda port:{'candidate':role,'local_relevel_steps':60},fixed_selector_wait_steps=scope_steps,
        continuation_diagnostic={'trace_prefix':prefix,'branch':branch,'transition':transition,
            'boundary':boundary,'account_control_latency':account_control_latency},
        continuation_selector=observer,pace_recovery=pace_recovery,
        search_exhaustion_recovery=boundary=='search_exhausted' and branch in ('R_global','R_table_assist'),
        search_exhaustion_skill=branch if branch in ('R_global','R_table_assist') else 'R_global',
        table_assist_template=table_assist_template,
        continuation_windows=tuple(source.get('continuation_windows',[30])),
        continuation_features=source.get('continuation_features',False))
    selected=row.get('continuation_diagnostic',{}).get('branch',branch)
    row.update(method='continue60' if selected=='continue' else 'reject_after30' if selected=='reject' else 'global_after30',diagnostic_only=True,
        source_candidate=role,source_safe_success=source['success'] and not source['protection_violation'],
        source_selector_wall_seconds=source['selector_wall_seconds'],model_inference_rerun=branch=='model',
        controller_math_warmup_before_initialization_seconds=warmup_seconds,
        episode_wall_seconds=time.perf_counter()-started)
    if boundary=='relevel40' and selected=='R_global':row['method']='global_after40'
    if selected=='R_table_assist':row['method']='table_after_search'
    row['diagnostic_errors']=audit(row,source)
    return row


def audit(row,source):
    errors=[]
    if row.get('error'):return [row['error']]
    boundary=row.get('continuation_diagnostic',{})
    search_boundary=boundary.get('boundary')=='search_exhausted'
    replay_boundary='search_exhausted' if search_boundary else 'relevel40' if boundary.get('relevel_step')==40 else 'relevel30'
    prefix=boundary_prefix(source,replay_boundary)
    if row['trace'][:len(prefix)]!=prefix:errors.append('source_prefix_mismatch')
    if boundary.get('boundary_trace_length')!=len(prefix):errors.append('wrong_intervention_boundary')
    for name in STATE_KEYS:
        if row.get(name)!=source.get(name):errors.append(name+'_mismatch')
    wait_names=('inference_wait','controller_compute_wait','prefix_observation_wait','duration_budget_wait','duration_deadline_wait')
    original_replay_waits=[f for f in row['trace'] if f['stage'] in wait_names and f['time']<=prefix[-1]['time']+1e-8]
    waits=[f for f in row['trace'] if f['stage'] in (*wait_names,'transition_compute_wait')]
    original_waits=[f for f in prefix if f['stage'] in wait_names]
    if original_replay_waits!=original_waits:errors.append('original_waits_not_replayed')
    if len(waits)!=row['inference_wait_steps'] or abs(len(waits)/20-row['inference_wait_seconds'])>1e-8:
        errors.append('wait_accounting')
    if abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])>1e-8:
        errors.append('total_cost_accounting')
    if row['blocked_calls'] or any(row[k]!=0 for k in ('state_edit_count_after_initialization','contact_anchor_count','external_wrench_at_end')):
        errors.append('forbidden_state_edit_or_force')
    before=boundary['protection_before'];after=row['protection']
    if any(after[k]<before[k] for k in before):errors.append('protection_counter_reset')
    keeper=1 if row['source_candidate']=='R_left' else 0
    if row['method'] in ('continue60','reject_after30'):
        protected=[f for f in row['trace'] if f['phase']=='recovery']
        if row['method']=='continue60' and source['candidate'] in ('R_left','R_right') and (
                search_boundary or sum(f['stage']=='local_relevel' for f in source['trace'])==60):
            if row['trace']!=source['trace']:errors.append('continued_trajectory_mismatch')
            for k in ('success','protection','protection_violation','failure_reason','total_post_fault_seconds'):
                if row[k]!=source[k]:errors.append('continued_'+k+'_mismatch')
    else:
        handoff=next((i for i,f in enumerate(row['trace']) if f['stage'] in ('global_table_loaded','assist_table_loaded') and f['evaluator']['table_contact']),len(row['trace']))
        protected=[f for f in row['trace'][:handoff] if f['phase']=='recovery']
        if sum(f['stage']=='local_relevel' for f in row['trace'])!=(0 if search_boundary else boundary.get('relevel_step',30)):
            errors.append('wrong_local_steps_before_global')
        if row['success']:
            if row['method']=='table_after_search':
                if not row['table_assist_support_verified'] or not row['table_assist_lift_verified']:errors.append('unverified_table_assist')
            elif not row['table_cycle_verified']:errors.append('unverified_table_cycle')
    losses=sum(not f['evaluator']['holds'][keeper] for f in protected)
    if losses!=row['protection']['keeper_grasp_lost_steps']:errors.append('keeper_guard_not_carried_to_table')
    return errors


def analyze(rows,sources):
    pairs={}
    for r in rows:pairs.setdefault((r['seed'],r['scenario']),{})[r['method']]=r
    safe=lambda r:r['success'] and not r['protection_violation']
    mean=lambda a:sum(a)/len(a) if a else None
    cases=[];errors=[]
    for key,pair in sorted(pairs.items()):
        if set(pair)!={'continue60','global_after30'}:errors.append([*key,'incomplete_pair']);continue
        a,b=pair['continue60'],pair['global_after30']
        for name in ('boundary_state','boundary_robot','boundary_position_targets','boundary_rotation_targets',
                'protection_before','boundary_wait_steps'):
            if a['continuation_diagnostic'][name]!=b['continuation_diagnostic'][name]:errors.append([*key,name+'_mismatch'])
        for r in pair.values():errors.extend([*key,r['method'],e] for e in r['diagnostic_errors'])
        cut=b['continuation_diagnostic']['boundary_trace_length'];tail=b['trace'][cut:]
        keeper=1 if b['source_candidate']=='R_left' else 0
        loss=next((f for f in tail if f['phase']=='recovery' and not f['evaluator']['holds'][keeper]),None)
        contact=next((f for f in tail if f['evaluator']['table_contact']),None)
        boundary_time=b['continuation_diagnostic']['boundary_state']['time']
        pre_contact_losses=sum(not f['evaluator']['holds'][keeper] for f in tail if f['phase']=='recovery' and
            (contact is None or f['time']<contact['time']))
        cases.append({'seed':key[0],'scenario':key[1],'source_safe':a['source_safe_success'],
            'continue_label':('A' if safe(a) else 'B') if a['success'] else ('D' if a['protection_violation'] else 'C'),
            'global_label':('A' if safe(b) else 'B') if b['success'] else ('D' if b['protection_violation'] else 'C'),
            'safe_rescued':not safe(a) and safe(b),'safe_lost':safe(a) and not safe(b),
            'continue_total_seconds':a['total_post_fault_seconds'],'global_total_seconds':b['total_post_fault_seconds'],
            'cost_delta_seconds':b['total_post_fault_seconds']-a['total_post_fault_seconds'],
            'original_wait_seconds':a['inference_wait_seconds'],
            'prefix_seconds':a['continuation_diagnostic']['post_fault_prefix_seconds'],
            'continue_keeper_lost_steps':a['protection']['keeper_grasp_lost_steps'],
            'global_keeper_lost_steps':b['protection']['keeper_grasp_lost_steps'],
            'global_keeper_lost_before_first_table_contact_steps':pre_contact_losses,
            'global_first_keeper_loss_stage':loss['stage'] if loss else None,
            'global_first_keeper_loss_after_switch_seconds':loss['time']-boundary_time if loss else None,
            'global_first_table_contact_after_switch_seconds':contact['time']-boundary_time if contact else None,
            'table_cycle_verified':b['table_cycle_verified'],'global_failure_reason':b['failure_reason']})
    methods={}
    for method in ('continue60','global_after30'):
        group=[r for r in rows if r['method']==method]
        methods[method]={'episodes':len(group),'task_successes':sum(r['success'] for r in group),
            'safe_successes':sum(safe(r) for r in group),'process_violations':sum(r['protection_violation'] for r in group),
            'unsupported_over_100ms':sum(r['protection']['max_unsupported_seconds']>.1 for r in group),
            'mean_total_post_fault_seconds':mean([r['total_post_fault_seconds'] for r in group]),
            'mean_remaining_seconds_after_relevel30':mean([r['total_post_fault_seconds']-r['continuation_diagnostic']['post_fault_prefix_seconds'] for r in group]),
            'mean_wait_seconds':mean([r['inference_wait_seconds'] for r in group]),
            'new_control_wait_steps':sum(r.get('transition_compute_wait_steps',0) for r in group),
            'verified_table_cycles':sum(r['table_cycle_verified'] for r in group)}
    assert len(sources)==13 and len(rows)==26 and len(pairs)==13
    return {'diagnostic_only':True,'physical_states':13,'physical_episodes':26,'valid_execution':not errors,
        'errors':errors,'methods':methods,'safe_rescued_failure_states':sum(r['safe_rescued'] for r in cases),
        'safe_lost_control_states':sum(r['safe_lost'] for r in cases),
        'original_failed_states':sum(not r['source_safe'] for r in cases),
        'original_safe_states':sum(r['source_safe'] for r in cases),
        'original_safe_states_preserved':sum(r['source_safe'] and r['global_label']=='A' for r in cases),
        'global_keeper_interruptions_before_first_table_contact':sum(r['global_keeper_lost_before_first_table_contact_steps']>0 for r in cases),
        'mean_global_minus_continue_seconds':mean([r['cost_delta_seconds'] for r in cases]),'cases':cases,
        'scope':'All13 previously observed family_set N60 states;7 unsafe and6 safe controls. Fixed intervention at relevel30, no model calls or fitting, no new independent-test claim. Source actuator commands and all original physical waits replayed from fresh initialization, without state restoration. Cumulative protection includes original keeper until actual table-loaded contact. New branch inference latency is not estimated by this predetermined physical diagnostic.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',default='outputs/vlm-duration-family-stream-20261109/closed-loop')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--transition',choices=('hard','defer_rotation','rigid'),default='hard')
    parser.add_argument('--account-control-latency',action='store_true')
    parser.add_argument('--reference-pilot',help='Require the validated controller and physical configuration to match this pilot')
    args=parser.parse_args();source_dir=Path(args.source_dir).resolve();output=Path(args.output_dir).resolve()
    if (output/'trials.jsonl').exists():raise ValueError('diagnostic results already exist')
    output.mkdir(parents=True,exist_ok=True)
    source_file=source_dir/'trials.jsonl'
    sources=[r for r in map(json.loads,source_file.read_text().splitlines()) if
        r['method']=='family_set' and r.get('local_steps',{}).get('local_relevel_steps')==60]
    assert len(sources)==13 and sum(r['success'] and not r['protection_violation'] for r in sources)==6
    old_protocol=json.loads((source_dir/'protocol.json').read_text())
    frozen={p:file_hash(p) for p in old_protocol['frozen']}
    assert frozen==old_protocol['frozen'],'a frozen model or threshold changed before the diagnostic'
    reference=None
    if args.reference_pilot:
        pilot=Path(args.reference_pilot).resolve()
        reference=json.loads((pilot/'protocol.json').read_text())
        assert reference['frozen_models_and_threshold']==frozen
        assert args.transition in reference['variants']
        assert file_hash(Path(__file__).resolve().parent/'l4_lift.py')==file_hash(pilot/'source/l4_lift.py')
        assert json.loads(json.dumps(asdict(replace(LiftConfig(),local_relevel_steps=60))))==reference['config']
    protocol={'diagnostic_only':True,'use_for_training':False,'use_as_new_independent_test':False,
        'source':str(source_file),'source_sha256':file_hash(source_file),'frozen_models_and_threshold':frozen,
        'threshold':old_protocol['threshold'],'states':[{'seed':r['seed'],'scenario':r['scenario']} for r in sources],
            'transition':args.transition,'account_control_latency':args.account_control_latency,
            'reference_pilot':str(Path(args.reference_pilot).resolve()) if args.reference_pilot else None,
            'validated_controller_sha256':file_hash(Path(__file__).resolve().parent/'l4_lift.py'),
        'branches':['continue60','global_after30'],'intervention_relevel_step':30,'planned_physical_episodes':26,
        'config':asdict(replace(LiftConfig(),local_relevel_steps=60)),
        'protection':'Counters accumulate from fault recovery without reset; original keeper protected until table-loaded native contact; then original global support/collision guards and table cycle verification remain active.',
        'waiting':'Replay original inference_wait and controller_compute_wait actions at their original positions. No model inference rerun. Full prefix and final verification included in post-fault cost.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (output/'source').mkdir(exist_ok=True)
    for name in ('l4_lift.py','duration_continuation.py','l0.py','continuous.py','recovery_skills.py'):
        path=Path(__file__).resolve().parent/name
        (output/'source'/name).write_bytes(path.read_bytes())
    specs=[]
    for i,source in enumerate(sources):
        for branch in (('continue','R_global') if i%2==0 else ('R_global','continue')):specs.append((source,branch))
    rows=[];started=time.perf_counter()
    worker=partial(run_continuation,transition=args.transition,account_control_latency=args.account_control_latency)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (output/'trials.jsonl').open('w') as stream:
            for row in pool.map(worker,specs):
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','total_post_fault_seconds','diagnostic_errors')}),flush=True)
                if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
    report=analyze(rows,sources)
    report['transition']=args.transition
    report['frozen_models_and_threshold_unchanged']=all(file_hash(p)==digest for p,digest in frozen.items())
    report['source_data_unchanged']=file_hash(source_file)==protocol['source_sha256']
    report['campaign_wall_seconds']=time.perf_counter()-started
    assert report['valid_execution'] and report['frozen_models_and_threshold_unchanged'] and report['source_data_unchanged']
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['cases'][0]));writer.writeheader();writer.writerows(report['cases'])
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
