"""Twelve fixed-prefix physical trials of three loaded global-entry controls."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict,replace
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .duration_continuation import run_continuation,boundary_prefix
from .duration_pilot import file_hash
from .l4_lift import LiftConfig


CASES=((24000,'right_loss'),(24003,'left_loss'),(24001,'left_loss'),(24007,'right_loss'))
VARIANTS=('hard','defer_rotation','rigid')


def rotation_angle(matrix):
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix)-1)/2,-1.,1.))))


def run_case(spec):
    source,variant,legacy=spec
    # Compile the already-used controller math before any environment exists;
    # cold numerical setup must not become a method-order-dependent robot hold.
    from robosuite.utils import transform_utils as transforms
    warmed=time.perf_counter()
    transforms.quat2axisangle(transforms.mat2quat(np.eye(3)))
    warmup_seconds=time.perf_counter()-warmed
    row=run_continuation((source,'R_global'),transition=variant,account_control_latency=True)
    row['method']=variant
    row['controller_math_warmup_before_initialization_seconds']=warmup_seconds
    if not row.get('error'):
        row['source_prefix_exact']=row['trace'][:len(boundary_prefix(source))]==boundary_prefix(source)
        row['legacy_hard_full_trace_exact']=(row['trace']==legacy['trace']) if variant=='hard' else None
        if variant=='hard' and not row.get('transition_compute_wait_steps') and not row['legacy_hard_full_trace_exact']:
            row['diagnostic_errors'].append('legacy_hard_trajectory_mismatch_without_extra_wait')
    return row


def case_metrics(row):
    d=row['continuation_diagnostic'];start=d['boundary_state']['time'];cut=d['boundary_trace_length']
    keeper=1 if row['source_candidate']=='R_left' else 0;repair=1-keeper
    frames=[f for f in row['trace'][cut:] if f['phase']=='recovery' and f['time']-start<=.5+1e-8]
    continuation=[f for f in row['trace'][cut:] if f['phase']=='recovery']
    handoff=next((i for i,f in enumerate(continuation) if f['stage']=='global_table_loaded' and
        f['evaluator']['table_contact']),len(continuation))
    guarded_loss=next((f for f in continuation[:handoff] if not f['evaluator']['holds'][keeper]),None)
    first_contact=next((f for f in continuation if f['evaluator']['table_contact']),None)
    flight=maximum=0
    for f in frames:
        e=f['evaluator'];flight=flight+1 if not(any(e['holds']) or e['table_contact']) else 0;maximum=max(maximum,flight)
    commands=[c for c in row['continuation_commands'] if c['stage']=='global_lower_to_table']
    first=commands[0];p0=np.asarray(d['boundary_position_targets']);r0=np.asarray(d['boundary_rotation_targets'])
    jump=np.linalg.norm(np.asarray(first['position_targets'])-p0,axis=1)*1000
    rot_jump=[rotation_angle(np.asarray(first['rotation_targets'])[i]@r0[i].T) for i in range(2)]
    span=p0[1]-p0[0];relative_rotation=r0[0].T@r0[1];relative_position=r0[0].T@span
    distance_errors=[];orientation_errors=[];grasp_errors=[]
    for c in commands:
        p=np.asarray(c['position_targets']);r=np.asarray(c['rotation_targets']);delta=p[1]-p[0]
        distance_errors.append(abs(np.linalg.norm(delta)-np.linalg.norm(span))*1000)
        orientation_errors.append(rotation_angle((r[0].T@r[1])@relative_rotation.T))
        grasp_errors.append(float(np.linalg.norm(r[0].T@delta-relative_position))*1000)
    safe=row['success'] and not row['protection_violation']
    return {'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
        'label':('A' if safe else 'B') if row['success'] else ('D' if row['protection_violation'] else 'C'),
        'task_success':row['success'],'safe_success':safe,'process_violation':row['protection_violation'],
        'keeper_loss_first500ms_steps':sum(not f['evaluator']['holds'][keeper] for f in frames),
        'max_unsupported_first500ms_seconds':maximum/20,
        'max_unsupported_whole_seconds':row['protection']['max_unsupported_seconds'],
        'first_guarded_keeper_loss_after_switch_seconds':guarded_loss['time']-start if guarded_loss else None,
        'first_guarded_keeper_loss_stage':guarded_loss['stage'] if guarded_loss else None,
        'table_contact_at_first_guarded_keeper_loss':guarded_loss['evaluator']['table_contact'] if guarded_loss else None,
        'first_table_contact_after_switch_seconds':first_contact['time']-start if first_contact else None,
        'keeper_loss_whole_steps':row['protection']['keeper_grasp_lost_steps'],
        'deep_interarm_contact_steps':row['protection']['deep_interarm_contact_steps'],
        'repair_reference_jump_mm':float(jump[repair]),'keeper_reference_jump_mm':float(jump[keeper]),
        'repair_reference_rotation_jump_deg':rot_jump[repair],'keeper_reference_rotation_jump_deg':rot_jump[keeper],
        'max_reference_span_length_change_mm':max(distance_errors),
        'max_reference_relative_rotation_change_deg':max(orientation_errors),
        'max_reference_grasp_translation_change_mm':max(grasp_errors),
        'original_wait_steps':d['boundary_wait_steps'],'new_control_wait_steps':row.get('transition_compute_wait_steps',0),
        'total_wait_seconds':row['inference_wait_seconds'],'total_post_fault_seconds':row['total_post_fault_seconds'],
        'remaining_seconds_after_switch':row['total_post_fault_seconds']-d['post_fault_prefix_seconds'],
        'max_control_wall_seconds':row.get('max_recovery_control_wall_seconds',0.),
        'table_cycle_verified':row['table_cycle_verified'],'failure_reason':row['failure_reason']}


def analyze(rows):
    assert len(rows)==12
    errors=[];groups={}
    for r in rows:
        errors.extend([r['seed'],r['scenario'],r['method'],x] for x in r['diagnostic_errors'])
        groups.setdefault((r['seed'],r['scenario']),{})[r['method']]=r
    assert set(groups)==set(CASES)
    for key,group in groups.items():
        assert set(group)==set(VARIANTS)
        reference=group['hard']['continuation_diagnostic']
        for method,row in group.items():
            for name in ('boundary_state','boundary_robot','boundary_position_targets','boundary_rotation_targets',
                    'boundary_wait_steps','protection_before'):
                if row['continuation_diagnostic'][name]!=reference[name]:errors.append([*key,method,name+'_mismatch'])
    cases=[case_metrics(r) for r in rows];methods={}
    for variant in VARIANTS:
        a=[r for r in cases if r['method']==variant];mean=lambda key:float(np.mean([r[key] for r in a]))
        methods[variant]={'episodes':len(a),'task_successes':sum(r['task_success'] for r in a),
            'safe_successes':sum(r['safe_success'] for r in a),'process_violations':sum(r['process_violation'] for r in a),
            'early_keeper_loss_states':sum(r['keeper_loss_first500ms_steps']>0 for r in a),
            'unsupported_over100ms_states':sum(r['max_unsupported_whole_seconds']>.1 for r in a),
            'mean_total_post_fault_seconds':mean('total_post_fault_seconds'),'mean_wait_seconds':mean('total_wait_seconds'),
            'new_control_wait_steps':sum(r['new_control_wait_steps'] for r in a),
            'repair_reference_jump_mm_range':[min(r['repair_reference_jump_mm'] for r in a),max(r['repair_reference_jump_mm'] for r in a)],
            'repair_reference_rotation_jump_deg_range':[min(r['repair_reference_rotation_jump_deg'] for r in a),max(r['repair_reference_rotation_jump_deg'] for r in a)]}
    return {'diagnostic_only':True,'use_for_training':False,'use_as_new_independent_test':False,
        'physical_states':4,'physical_episodes':12,'valid_execution':not errors,'errors':errors,'methods':methods,'cases':cases,
        'scope':'Predetermined physical control interventions at the exact replayed relevel30 boundary. No inference rerun or fine tuning. Source waits replayed; new control compute overruns physically held and charged. No new independent-policy or population-safety claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',default='outputs/vlm-duration-family-stream-20261109/closed-loop')
    parser.add_argument('--legacy-dir',default='outputs/vlm-duration-continuation-diagnostic-20261109')
    parser.add_argument('--output-dir',required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();output=Path(args.output_dir).resolve()
    if (output/'trials.jsonl').exists():raise ValueError('existing pilot must not be overwritten')
    output.mkdir(parents=True,exist_ok=True)
    source_file=Path(args.source_dir).resolve()/'trials.jsonl';legacy_dir=Path(args.legacy_dir).resolve()
    sources={(r['seed'],r['scenario']):r for r in map(json.loads,source_file.read_text().splitlines()) if r['method']=='family_set'}
    legacy={(r['seed'],r['scenario']):r for r in map(json.loads,(legacy_dir/'trials.jsonl').read_text().splitlines()) if r['method']=='global_after30'}
    previous=json.loads((legacy_dir/'protocol.json').read_text())
    frozen=previous['frozen_models_and_threshold'];assert all(file_hash(p)==d for p,d in frozen.items())
    config=json.loads(json.dumps(asdict(replace(LiftConfig(),local_relevel_steps=60))))
    assert config==previous['config']
    protocol={'diagnostic_only':True,'use_for_training':False,'use_as_new_independent_test':False,'fine_tuning_started':False,
        'cases':CASES,'variants':VARIANTS,'intervention_relevel_step':30,'planned_episodes':12,
        'source':str(source_file),'source_sha256':file_hash(source_file),'legacy_trials':str(legacy_dir/'trials.jsonl'),
        'legacy_sha256':file_hash(legacy_dir/'trials.jsonl'),'frozen_models_and_threshold':frozen,'threshold':previous['threshold'],
        'config':config,'action_position_gain':.8,'rotation_error_scale':.5,
        'motion_budget':'Same60 lower,8 loaded,14 release,6 settle,16 clear,24 hover,20 descend,22 close,50 lift,12 settle and20 final hold. All overruns add audited physical holds.',
        'control':'hard=legacy;defer_rotation=retain last commanded orientation through placement and clear, interpolate to registered in hover;rigid=also retain last position command and use a common rigid transform with quintic progress to the registered table center/span. Contact truth is audit only.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(output/'source').mkdir(exist_ok=True)
    for name in ('l4_lift.py','duration_continuation.py','duration_transition_pilot.py','continuous.py','l0.py','recovery_skills.py'):
        path=Path(__file__).resolve().parent/name;(output/'source'/name).write_bytes(path.read_bytes())
    specs=[]
    for i,key in enumerate(CASES):
        for variant in VARIANTS[i%3:]+VARIANTS[:i%3]:specs.append((sources[key],variant,legacy[key]))
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (output/'trials.jsonl').open('w') as stream:
            for row in pool.map(run_case,specs):
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','total_post_fault_seconds','diagnostic_errors')}),flush=True)
                if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
    report=analyze(rows)
    report.update(frozen_models_and_threshold_unchanged=all(file_hash(p)==d for p,d in frozen.items()),
        source_data_unchanged=file_hash(source_file)==protocol['source_sha256'] and file_hash(legacy_dir/'trials.jsonl')==protocol['legacy_sha256'],
        legacy_hard_all_trajectories_exact=all(r['legacy_hard_full_trace_exact'] for r in rows if r['method']=='hard'),
        campaign_wall_seconds=time.perf_counter()-started)
    assert report['valid_execution'] and report['frozen_models_and_threshold_unchanged'] and report['source_data_unchanged']
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(report['cases'][0]));writer.writeheader();writer.writerows(report['cases'])
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
