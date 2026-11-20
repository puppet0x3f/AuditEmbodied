"""Export unchanged logged observations for the small, offline review package.

Author-side operation only: run in the original workspace. The distributed
recompute.py never imports this file or reads the workspace's outputs directory.
"""
import gzip
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent
SOURCES = {
    'handover': 'outputs/handover-audit-20261109/trials.jsonl',
    'scope': 'outputs/agentchord-style-20261109/trials.jsonl',
    'supervision': 'outputs/supervision-controls-20261109/closed-loop/trials.jsonl',
    'load': 'outputs/p1-load-transfer-20261109/trials.jsonl',
    'waiting': 'outputs/p0-20261109/intervention-trials.jsonl',
    'wait_decision': 'outputs/wait-decision-20261110/trials.jsonl',
    'interface': 'outputs/interface-retarget-20261110/trials.jsonl',
}
CONTROLLER_SOURCE = 'outputs/closed-loop-recovery-20261111/trials.jsonl'
if (ROOT / CONTROLLER_SOURCE).exists():
    SOURCES['controller'] = CONTROLLER_SOURCE
PILOT_SOURCE='outputs/cross-controller-20261110/trials.jsonl'
if (ROOT/PILOT_SOURCE).exists():
    SOURCES['controller_pilot']=PILOT_SOURCE
PROSPECTIVE=ROOT/'outputs/wait-hold-geometry-20261111'
for name,source in [('wait_hold','wait-trials.jsonl'),('geometry','geometry-trials.jsonl')]:
    if (PROSPECTIVE/source).exists():
        SOURCES[name]=str((PROSPECTIVE/source).relative_to(ROOT))
FOLLOWUP=ROOT/'outputs/hold-components-vision-20261111'
for name,source in [('holding_components','mechanism-trials.jsonl'),('independent_visual','independent-trials.jsonl')]:
    if (FOLLOWUP/source).exists():SOURCES[name]=str((FOLLOWUP/source).relative_to(ROOT))
PAIRED=ROOT/'outputs/paired-executor-holding-20261111'
for name,source in [('paired_holding','paired-trials.jsonl'),('paired_placement','placement-trials.jsonl')]:
    if (PAIRED/source).exists():SOURCES[name]=str((PAIRED/source).relative_to(ROOT))
HANDOVER_HOLDING=ROOT/'outputs/handover-holding-20261111'
if (HANDOVER_HOLDING/'formal-trials.jsonl').exists():
    SOURCES['handover_holding']=str((HANDOVER_HOLDING/'formal-trials.jsonl').relative_to(ROOT))
VISUAL_CORE=ROOT/'outputs/visual-execution-core-20261111'
if (VISUAL_CORE/'lifting/formal-trials.jsonl').exists():
    SOURCES['lifting_smooth']=str((VISUAL_CORE/'lifting/formal-trials.jsonl').relative_to(ROOT))
if (VISUAL_CORE/'public-policy/formal-trials.jsonl').exists():
    SOURCES['public_transport']=str((VISUAL_CORE/'public-policy/formal-trials.jsonl').relative_to(ROOT))
TABLES = ('handover-audit', 'scope', 'load-transfer', 'protection-validity',
          'supervision', 'selection-change', 'wait-decisions', 'interface-retarget', 'interface-full',
          'evaluation-layers')
FIELDS = (
    'seed', 'scenario', 'method', 'candidate', 'window', 'load', 'success',
    'task_complete', 'handover_completed', 'failure_reason', 'blocked_calls',
    'abstained', 'used_global_recovery', 'table_cycle_verified',
    'table_setdown_verified', 'table_regrasp_and_lift_verified',
    'giver_fully_released', 'receiver_stable_for_1s', 'bad_state_warnings',
    'post_fault', 'post_wait_evaluator', 'post_fault_predicate',
    'protection', 'protection_violation', 'recovery_path_m',
    'inference_wait_steps', 'inference_wait_seconds', 'skill_seconds',
    'hold_seconds', 'total_post_fault_seconds', 'planned_wait_steps',
    'planned_duration_steps', 'selection',
    'contact_anchor_count', 'state_edit_count_after_initialization',
    'wait_decision', 'wait_initial_decision',
    'regrasp_decision',
    'global_execution', 'native_file', 'native_samples', 'uniform_audit',
    'controller_services','controller_events',
)
EXAMPLES = {
    ('handover', 40000, 'object_rotation', 'rule', 'actual'): 'handover-retry-A',
    ('handover', 40001, 'object_rotation', 'rule', '1000ms'): 'handover-completed-B',
    ('handover', 40017, 'object_rotation', 'proprio', '2000ms'): 'handover-invalid-E',
    ('scope', 12006, 'right_loss', 'protected', None): 'lift-protected-A',
    ('scope', 12006, 'right_loss', 'agentchord_style', None): 'lift-repaired-B',
    ('load', 50000, 'left_loss', 'global', 'actual'): 'lift-heavy-pregrasp-D',
}


def project(row, cohort):
    if cohort=='public_transport':return dict(row)
    if cohort=='handover_holding':
        if row['holding_input'] is None:
            assert row['failure_reason']=='initialization_error' and not row['trace']
            return dict(row)
        result=project(row,'handover')
        for key in ('hold_mode','wait_steps','native_file','native_samples','holding_input','hold_tracking',
                    'controller_overrun_steps','passive_recording_wall_seconds','initialization_state',
                    'initialization_end_state','prefix_state','registered_robot_poses','initialization_supported',
                    'external_wrench_at_recovery_start'):
            result[key]=row[key]
        result['decision_events']=row['decision_events']
        return result
    if cohort in ('wait_hold','geometry','holding_components','independent_visual','paired_holding','paired_placement','lifting_smooth'):
        result=project(row,'controller')
        result.update({k:row[k] for k in ('geometry','hold_mode','wait_steps','registered_mass_kg','holding_input')})
        if cohort in ('holding_components','independent_visual','paired_holding','paired_placement','lifting_smooth'):result['hold_tracking']=row['hold_tracking']
        if cohort in ('holding_components','independent_visual'):
            frames={round(f['time'],8):f for f in row['trace']}
            for d in result['global_execution']['observations']:
                f=frames[round(d['observed_time'],8)]
                d['observation_truth']={'time':f['time'],'stage':f['stage'],
                    'handle_positions':f['evaluator']['handle_positions'],'robot_positions':f['robot']['positions']}
        if cohort in ('paired_holding','paired_placement','lifting_smooth'):
            result.update({k:row[k] for k in ('initialization_state','preclose_state',
                'initialization_end_state','prefix_state','post_wait_state')})
        if cohort=='paired_holding':
            closing=next(a['close_time'] for a in row['global_execution']['pickup_attempts'] if a.get('close_time') is not None)
            observed=max(d['observed_time'] for d in row['global_execution']['observations']
                if d['estimate']['valid'] and d['observed_time']<closing+1e-8)
            frames={round(f['time'],8):f for f in row['trace']}
            for f in result['trace']:
                if observed<f['time']<=closing+1e-8:
                    f['robot']['action']=frames[round(f['time'],8)]['robot']['action']
            actual=next(f for f in row['trace'] if abs(f['time']-closing)<1e-8)
            projected=next(f for f in result['trace'] if abs(f['time']-closing)<1e-8)
            projected['robot'].update({k:actual['robot'][k] for k in ('rotations','action')})
        # New uniform labels use raw terminal C; legacy scoring separately checks
        # the recorded success/interface flags before uniform C/P are reconstructed.
        result['recorded'].pop('label',None)
        return result
    result = {k: row[k] for k in FIELDS if k in row}
    result['prefix_state'] = {'time': row['prefix_state']['time']}
    # Values are copied without rounding. Prefix history is unnecessary for
    # scoring; retain the last fault sample for admissions at observation time.
    prefix = [f for f in row['trace'] if f['time'] <= row['prefix_state']['time'] + 1e-8][-1:]
    after = [f for f in row['trace'] if f['time'] > row['prefix_state']['time'] + 1e-8]
    result['trace'] = []
    for frame in prefix + after:
        sample = {k: frame[k] for k in ('time', 'phase', 'stage', 'skill') if k in frame}
        if cohort == 'handover':
            keys = ('giver_holds', 'receiver_holds', 'receiver_supports', 'table_contact', 'clearance', 'relative_speed')
            sample['evaluator'] = {k: frame['evaluator'][k] for k in keys}
            sample['robot'] = {'gripper_commands': [frame['robot']['action14'][i] for i in (6, 13)],
                               'actual_finger_qpos4': frame['robot']['actual_finger_qpos4']}
        else:
            # Contact atoms are retained at every step. Kinematics are needed
            # for final completion, global lift acceptance and stage diagnosis.
            keys = ['holds', 'table_contact']
            if frame['stage'] == 'final_stable_hold':
                keys += ['supports', 'relative_speeds', 'clearance', 'tilt_deg', 'handle_height_difference', 'ready']
            elif frame['stage'] in ('global_lift', 'global_lift_settle'):
                keys += ['clearance']
            if frame['stage'] in ('global_table_settle', 'global_close'):
                keys += ['tilt_deg', 'handle_positions']
                sample['robot'] = {'positions': frame['robot']['positions']}
            sample['evaluator'] = {k: frame['evaluator'][k] for k in dict.fromkeys(keys)}
            if cohort in ('interface', 'controller','controller_pilot'):
                sample['robot'] = {'positions':frame['robot']['positions']}
                if frame['stage']=='global_close':
                    sample['evaluator'].update({k:frame['evaluator'][k] for k in ('handle_positions','tilt_deg')})
                if frame['stage']=='regrasp_observation_wait':
                    sample['evaluator']['handle_positions']=frame['evaluator']['handle_positions']
        result['trace'].append(sample)
    if 'duration_selection' in row:
        result['duration_selection'] = {k: row['duration_selection'][k] for k in ('decision', 'baseline_fallback')}
    if cohort=='controller_pilot':
        name='-'.join(str(row[k]) for k in ('seed','scenario','load','window','method'))+'-native.npz'
        result['native_file']=name
        result['skill_events']=[{k:e[k] for k in ('candidate','start_time','end_time')} for e in row['skill_events']]
    if cohort == 'handover':
        event_keys = ('kind', 'plan', 'start_time', 'end_time', 'observed_time',
                      'required_entry_predicate', 'prerequisite_at_observation',
                      'prerequisite_at_entry', 'physical_wait_steps')
        call_keys = ('index', 'skill', 'start_time', 'end_time', 'prerequisite',
                     'start_evaluator', 'valid_at_entry')
        result['decision_events'] = [{k: e[k] for k in event_keys} for e in row['decision_events']]
        result['skill_calls'] = [{k: c[k] for k in call_keys} for c in row['skill_calls']]
        result['recovery'] = {'sequence': row.get('recovery', {}).get('sequence', [])}
        result['recorded'] = row['audit']
    else:
        result['recorded'] = {'success': row['success'], 'task_complete': row['task_complete'],
                              'protection_violation': row['protection_violation']}
        if 'audit' in row:
            result['recorded'].update(row['audit'])
        if cohort == 'controller':
            result['audit'] = row['audit']
    return result


def main():
    for name in ('data', 'examples', 'reference', 'reference/tables', 'models'):
        (DEST / name).mkdir(parents=True, exist_ok=True)
    counts = {}
    for cohort, source in SOURCES.items():
        count = 0
        with (ROOT / source).open() as stream, gzip.open(DEST / 'data' / f'{cohort}.jsonl.gz', 'wt') as target:
            for line in stream:
                row = json.loads(line)
                small = project(row, cohort)
                target.write(json.dumps(small, separators=(',', ':')) + '\n')
                key = (cohort, row['seed'], row['scenario'], row['method'], row.get('window'))
                # Select the actual-window heavy case, rather than its native counterpart.
                if key in EXAMPLES and (cohort != 'load' or row['load'] == 'density150'):
                    raw = {k: v for k, v in row.items() if k not in
                           ('policy_observations', 'recorded_rgb_bundles', 'policy_input_file', 'acquisition')}
                    with gzip.open(DEST / 'examples' / (EXAMPLES[key] + '.json.gz'), 'wt') as f:
                        json.dump(raw, f, separators=(',', ':'))
                count += 1
        counts[cohort] = count
    expected = {'handover': 720, 'scope': 300, 'supervision': 160, 'load': 960, 'waiting': 96, 'wait_decision': 480, 'interface':720}
    if 'controller' in SOURCES:expected['controller']=480
    if 'controller_pilot' in SOURCES:expected['controller_pilot']=480
    if 'wait_hold' in SOURCES:expected['wait_hold']=800
    if 'geometry' in SOURCES:expected['geometry']=240
    if 'holding_components' in SOURCES:expected['holding_components']=480
    if 'independent_visual' in SOURCES:expected['independent_visual']=240
    if 'paired_holding' in SOURCES:expected['paired_holding']=960
    if 'paired_placement' in SOURCES:expected['paired_placement']=240
    if 'handover_holding' in SOURCES:expected['handover_holding']=720
    if 'lifting_smooth' in SOURCES:expected['lifting_smooth']=360
    if 'public_transport' in SOURCES:expected['public_transport']=360
    assert counts == expected
    for name in TABLES:
        shutil.copyfile(ROOT / 'paper/tables' / f'{name}.tex', DEST / 'reference/tables' / f'{name}.tex')
    graph_calls = []
    with (ROOT / SOURCES['scope']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['method'] == 'agentchord_style' and row['scenario'] != 'sham':
                graph_calls.append({'seed': row['seed'], 'scenario': row['scenario'],
                    'calls': [{k: c[k] for k in ('candidate', 'start_time', 'end_time')}
                              for c in row['skill_events']]})
    (DEST / 'data/scope-calls.json').write_text(json.dumps(graph_calls, separators=(',', ':')) + '\n')
    # Reference reports are used only after recomputation, as comparison targets.
    reports = {
        'handover': 'outputs/handover-audit-20261109/report.json',
        'scope': 'outputs/agentchord-style-20261109/report.json',
        'supervision': 'outputs/supervision-controls-20261109/report.json',
        'load': 'outputs/p1-load-transfer-20261109/report.json',
    }
    references = {}
    for cohort, path in reports.items():
        r = json.loads((ROOT / path).read_text())
        keys = {
            'handover': ('methods', 'baseline_paired_fault_comparisons'),
            'scope': ('fault_results', 'healthy_controls', 'all_four_common_safe'),
            'supervision': ('methods', 'paired'),
            'load': ('fault_results', 'healthy_controls', 'paired_methods'),
        }[cohort]
        references[cohort] = {k: r[k] for k in keys}
    (DEST / 'reference/frozen-results.json').write_text(json.dumps(references, indent=2) + '\n')
    shutil.copyfile(ROOT / 'outputs/p1-regrasp-attribution-20261109/raw-attribution.json',
                    DEST / 'reference/regrasp-attribution.json')
    for source,target in [('report.json','reference/wait-decisions.json'),('protocol.json','data/wait-decision-protocol.json')]:
        shutil.copyfile(ROOT / 'outputs/wait-decision-20261110' / source, DEST / target)
    frame_dir=DEST/'figures/execution';frame_dir.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'outputs/wait-decision-20261110/keyframes').iterdir():
        if source.suffix in ('.png','.json'):shutil.copyfile(source,frame_dir/source.name)
    interface=ROOT/'outputs/interface-retarget-20261110'
    shutil.copyfile(interface/'geometry.py',DEST/'interface_geometry.py')
    shutil.copyfile(interface/'protocol.json',DEST/'data/interface-protocol.json')
    shutil.copyfile(interface/'report.json',DEST/'reference/interface-results.json')
    image_dir=DEST/'figures/interface';image_dir.mkdir(parents=True,exist_ok=True)
    for source in (interface/'paper-keyframes').iterdir():
        if source.suffix in ('.png','.json','.npz'):shutil.copyfile(source,image_dir/source.name)
    shutil.copyfile(ROOT / 'outputs/p0-20261109/direct-selected.json', DEST / 'models/direct-selected.json')
    # Model payloads are exact copies. Offline trajectory scoring does not need
    # their image inputs, inference stack, or the separate 8B VLM checkpoints.
    for name in ('support-robot.npz', 'visual-support.npz'):
        shutil.copyfile(ROOT / 'outputs/handover-audit-20261109/models' / name, DEST / 'models' / name)
    p = json.loads((ROOT / 'outputs/handover-audit-20261109/protocol.json').read_text())
    q = json.loads((ROOT / 'outputs/p1-load-transfer-20261109/protocol.json').read_text())
    config = {'source_files': SOURCES, 'episodes': counts,
              'handover': {k: p[k] for k in ('thresholds', 'physical_config') if k in p},
              'lift': {k: q[k] for k in ('threshold', 'rule', 'test_config', 'loads', 'windows')},
              'bootstrap': {'draws': 2000, 'seed': 20261109, 'unit': 'initialization seed'},
              'waiting_policy_bootstrap': {'draws': 2000, 'seed': 20261110, 'unit': 'initialization seed'},
              'interface_bootstrap': {'draws': 2000, 'seed': 20261111, 'unit': 'initialization seed'},
              'projection': 'Unrounded logged support atoms/stages/times along the full path; terminal kinematics, necessary robot fields and global stage endpoints. Not images or a simulator replay.'}
    (DEST / 'data/config.json').write_text(json.dumps(config, indent=2) + '\n')
    if 'wait_hold' in SOURCES:
        shutil.copyfile(PROSPECTIVE/'geometry.py',DEST/'prospective_geometry.py')
        shutil.copyfile(PROSPECTIVE/'protocol.json',DEST/'data/wait-geometry-protocol.json')
        images=DEST/'figures/prospective';images.mkdir(parents=True,exist_ok=True)
        for path in (PROSPECTIVE/'rgb-examples').iterdir():
            shutil.copyfile(path,images/path.name)
        for cohort,phase in [('wait_hold','wait'),('geometry','geometry')]:
            if cohort not in SOURCES:continue
            native=DEST/'data'/(cohort+'-native');native.mkdir(exist_ok=True)
            for path in (PROSPECTIVE/phase).glob('*-native.npz'):
                # Keep XYZ/tilt here as measured physical diagnostics, alongside guards.
                shutil.copyfile(path,native/path.name)
            shutil.copyfile(PROSPECTIVE/(phase+'-analysis/report.json'),DEST/'reference'/(cohort+'-results.json'))
            for path in (PROSPECTIVE/(phase+'-analysis')).glob('*.tex'):
                shutil.copyfile(path,DEST/'reference/tables'/path.name)
    if 'holding_components' in SOURCES:
        for cohort,phase in [('holding_components','mechanism'),('independent_visual','independent')]:
            native=DEST/'data'/(cohort+'-native');native.mkdir(exist_ok=True)
            for path in (FOLLOWUP/phase).glob('*-native.npz'):shutil.copyfile(path,native/path.name)
            shutil.copyfile(FOLLOWUP/(phase+'-analysis/report.json'),DEST/'reference'/(cohort+'-results.json'))
            for path in (FOLLOWUP/(phase+'-analysis')).glob('*.tex'):
                shutil.copyfile(path,DEST/'reference/tables'/path.name)
        for name in ('protocol.json','independent-protocol.json'):
            shutil.copyfile(FOLLOWUP/name,DEST/'data'/('holding-'+name))
        native=DEST/'data/handover-native';native.mkdir(exist_ok=True)
        for path in (FOLLOWUP/'handover-native').glob('*.npz'):shutil.copyfile(path,native/path.name)
        shutil.copyfile(FOLLOWUP/'handover-replay-report.json',DEST/'reference/handover-native.json')
        images=DEST/'figures/landmark';images.mkdir(parents=True,exist_ok=True)
        import hashlib
        import imageio.v3 as iio
        import numpy as np
        for seed in (90,93):
            stem=f'{seed}-sham-01-measured_pose-visual_servo'
            row=json.loads((FOLLOWUP/'development-independent-final'/(stem+'.json')).read_text())
            observation=row['global_execution']['observations'][-1]
            arrays={camera:iio.imread(FOLLOWUP/'development-independent-final'/(stem+'-'+camera+'.png'))
                    for camera in ('agentview','frontview','sideview','birdview')}
            digests=[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays.values()]
            assert digests==observation['rgb_sha256']
            name=f'input-landmark-{seed}'
            np.savez_compressed(images/(name+'.npz'),**arrays)
            (images/(name+'.json')).write_text(json.dumps({'development_only':True,'data_file':name+'.npz',
                'port':observation['input'],'rgb_sha256':digests},separators=(',',':'))+'\n')
    if 'paired_holding' in SOURCES:
        shutil.copyfile(PAIRED/'prior-mechanism.json',DEST/'reference/prior-mechanism.json')
        for cohort,phase in [('paired_holding','paired'),('paired_placement','placement')]:
            if cohort not in SOURCES:continue
            native=DEST/'data'/(cohort+'-native');native.mkdir(exist_ok=True)
            for path in (PAIRED/phase).glob('*-native.npz'):shutil.copyfile(path,native/path.name)
            shutil.copyfile(PAIRED/(phase+'-analysis/report.json'),DEST/'reference'/(cohort+'-results.json'))
            for path in (PAIRED/(phase+'-analysis')).glob('*.tex'):
                shutil.copyfile(path,DEST/'reference/tables'/path.name)
        for name in ('protocol.json','paired-freeze.json','placement-freeze.json'):
            if (PAIRED/name).exists():shutil.copyfile(PAIRED/name,DEST/'data'/('paired-'+name))
        shutil.copyfile(ROOT/'paper/tables/paired-holding-main.tex',DEST/'reference/tables/paired-holding-main.tex')
        shutil.copyfile(ROOT/'paper/tables/handover-focus.tex',DEST/'reference/tables/handover-focus.tex')
    if 'handover_holding' in SOURCES:
        native=DEST/'data/handover_holding-native';native.mkdir(exist_ok=True)
        for path in (HANDOVER_HOLDING/'formal').glob('*-native.npz'):
            shutil.copyfile(path,native/path.name)
        shutil.copyfile(HANDOVER_HOLDING/'formal-analysis/report.json',DEST/'reference/handover-holding-results.json')
        shutil.copyfile(HANDOVER_HOLDING/'control-reference-diagnostics.json',DEST/'reference/control-reference-diagnostics.json')
        for name in ('protocol.json','formal-freeze.json'):
            shutil.copyfile(HANDOVER_HOLDING/name,DEST/'data'/('handover-holding-'+name))
        for name in ('timing-analysis-note.json','initialization-accounting.json'):
            shutil.copyfile(HANDOVER_HOLDING/name,DEST/'data'/('handover-holding-'+name))
        for name in ('handover-holding','handover-holding-main'):
            shutil.copyfile(ROOT/'paper/tables'/(name+'.tex'),DEST/'reference/tables'/(name+'.tex'))
        images=DEST/'figures/waiting';images.mkdir(parents=True,exist_ok=True)
        for path in (HANDOVER_HOLDING/'waiting-keyframes').glob('*.png'):
            shutil.copyfile(path,images/path.name)
        shutil.copyfile(HANDOVER_HOLDING/'waiting-replay-report.json',images/'replay-report.json')
    if 'lifting_smooth' in SOURCES:
        native=DEST/'data/lifting_smooth-native';native.mkdir(exist_ok=True)
        for path in (VISUAL_CORE/'lifting/formal').glob('*-native.npz'):shutil.copyfile(path,native/path.name)
        shutil.copyfile(VISUAL_CORE/'lifting/formal-analysis/report.json',DEST/'reference/lifting-smooth-results.json')
        for name in ('protocol.json','formal-freeze.json'):
            shutil.copyfile(VISUAL_CORE/'lifting'/name,DEST/'data'/('lifting-smooth-'+name))
        for name in ('execution-error','execution-stages','lifting-smooth'):
            shutil.copyfile(ROOT/'paper/tables'/(name+'.tex'),DEST/'reference/tables'/(name+'.tex'))
        shutil.copyfile(VISUAL_CORE/'existing-analysis/report.json',DEST/'reference/execution-error-results.json')
        shutil.copyfile(VISUAL_CORE/'analysis-protocol.json',DEST/'data/execution-error-protocol.json')
    if 'public_transport' in SOURCES:
        public=VISUAL_CORE/'public-policy'
        shutil.copyfile(public/'formal-analysis/report.json',DEST/'reference/public-policy-results.json')
        for name in ('protocol.json','formal-freeze.json','diffusion-continuation-freeze.json',
                     'lstm-continuation-freeze.json','gate.json','timing-and-prefix-note.json','prefix-interruptions-all.json'):
            shutil.copyfile(public/name,DEST/'data'/('public-policy-'+name))
        for name in ('public-policy-main','public-policy-full'):
            shutil.copyfile(ROOT/'paper/tables'/(name+'.tex'),DEST/'reference/tables'/(name+'.tex'))
        target=DEST/'figures/public-policy';target.mkdir(parents=True,exist_ok=True)
        metadata=[]
        for method in ('lstm_gmm','diffusion_policy_cnn'):
            for boundary in (32,272):
                name=f'{method}-{boundary}-policy-input.npz'
                shutil.copyfile(public/'formal'/method/f'boundary-{boundary:03d}-policy-input.npz',target/name)
                metadata.append({'method':method,'boundary':boundary,'file':name})
        (target/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
