"""Check published admission/protection semantics against the unchanged handover traces."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'outputs/handover-audit-20261109'
OUT=Path(__file__).resolve().parent
WAIT=('decision_wait','controller_compute_wait')
BEFORE_TABLE=('r3_free_other_gripper','r3_park_other_arm','r3_recenter_carrier',
    'r3_transport_above_real_table','r3_lower_to_table','r3_table_loaded','r3_table_release','r3_table_settle')


def inside(event, t):
    # A sample is recorded after its physical step. The start belongs to the
    # preceding stage, whereas the end belongs to this executed interval.
    return event['start_time']+1e-8<t<=event['end_time']+1e-8


def reconstruct_roles(row, frames):
    """Replay the published obligations from actual call intervals and truth.

    This changes no saved state, action, decision, or raw file. In particular,
    do not infer the final sample's role from an observer whose skill returned.
    """
    carrier={};handoff={};previous=None
    for f in frames:
        role=None;stage=f['stage'];truth=f['evaluator'];t=f['time']
        if f['phase'] in ('release','hold'):role='receiver_supports'
        elif stage in WAIT:
            decisions=[d for d in row['decision_events'] if inside(d,t)]
            if decisions:
                d=decisions[0]
                role=('receiver_supports' if d['plan']=='continue' else 'giver_holds'
                    if d['plan']=='R1_retarget' or d['kind']=='post_local_check' else None)
            elif stage=='controller_compute_wait':role=previous
            else:raise AssertionError(('unmatched decision wait',t))
        else:
            calls=[c for c in row['skill_calls'] if inside(c,t)]
            call=min(calls,key=lambda c:c['end_time']-c['start_time']) if calls else None
            if call:
                skill=call['skill']
                if skill in ('R1_retarget','R3_receiver_catch'):role='giver_holds'
                elif skill=='R3_transfer_R2':role='receiver_holds'
                elif skill=='R3':
                    idx=call['index']
                    if stage in ('r3_table_loaded','r3_table_release','r3_table_settle') and truth['table_contact']:
                        handoff[idx]=True
                    if stage in BEFORE_TABLE:
                        if idx not in carrier:
                            a=f['robot']['action14'];positive=[i for i,k in enumerate((6,13)) if a[k]>0]
                            if len(positive)==1:carrier[idx]=positive[0]
                        if not handoff.get(idx,False) and idx in carrier:
                            role='giver_holds' if carrier[idx]==0 else 'receiver_holds'
                    elif stage in ('r3_return_to_handover','r3_handover_settle'):role='giver_holds'
        yield f,role
        previous=role


def longest(values):
    run=maximum=0
    for value in values:run=run+1 if value else 0;maximum=max(maximum,run)
    return maximum


def main():
    count=0;changes=[];invalid=[];thresholds=set();labels=Counter();roles=Counter()
    boundary=[];counter_changes=[];total_old_bad=total_new_bad=0
    for line in (STUDY/'trials.jsonl').open():
        row=json.loads(line);a=row['audit'];assert a['complete']
        frames=[f for f in row['trace'] if f['phase'] in ('recovery','release','hold')]
        by_time={f['time']:f['evaluator'] for f in row['trace']}
        admission=[]
        for d in row['decision_events']:
            required={'continue':'receiver_supports','R1_retarget':'giver_holds','R3':None}[d['plan']]
            assert required==d['required_entry_predicate']
            observed=bool(by_time[d['start_time']][required]) if required else True
            entry=bool(by_time[d['end_time']][required]) if required else True
            assert observed==d['prerequisite_at_observation'] and entry==d['prerequisite_at_entry']
            admission.append(entry)
            if not entry:invalid.append({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
                'window':row['window'],'kind':d['kind'],'plan':d['plan'],'required':required,
                'valid_at_observation':observed,'valid_at_entry':entry})
        for c in row['skill_calls']:
            required={'R1_retarget':'giver_holds','R3_receiver_catch':'giver_holds',
                      'R3_transfer_R2':'receiver_holds','R3':None}[c['skill']]
            assert required==c['prerequisite']
            entry=bool(c['start_evaluator'][required]) if required else True
            assert entry==c['valid_at_entry'];admission.append(entry)
        v=all(admission)
        gap=longest([not(f['evaluator']['giver_holds'] or f['evaluator']['receiver_holds'] or f['evaluator']['table_contact']) for f in frames])*50
        shared=gap<=100 and not row['protection']['deep_robot_robot_contact_steps']
        obligation=True;bad=wait_bad=0
        final_recovery=max(f['time'] for f in frames if f['phase']=='recovery')
        for f,role in reconstruct_roles(row,frames):
            truth=f['evaluator'];old=f['audit']['role_obligation']
            ok=bool(truth[role]) if role else True
            assert (bool(truth[old]) if old else True)==f['audit']['role_obligation_satisfied']
            if role!=old:
                assert f['time']==final_recovery and old is None
                boundary.append({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
                    'window':row['window'],'time':f['time'],'stage':f['stage'],
                    'recorded_role':old,'reconstructed_role':role,'obligation_satisfied':ok})
            obligation&=ok;bad+=not ok;wait_bad+=not ok and f['stage'] in WAIT
            if role:roles[role]+=1
            thresholds.add((truth['height_threshold'],truth['relative_speed_threshold']))
        total_old_bad+=a['role_loss_samples'];total_new_bad+=bad
        if bad!=a['role_loss_samples'] or wait_bad!=a['role_violations_during_wait']:
            counter_changes.append({'seed':row['seed'],'scenario':row['scenario'],'method':row['method'],
                'window':row['window'],'label':a['label'],
                'old_role_loss_samples':a['role_loss_samples'],'corrected_role_loss_samples':bad,
                'old_wait_role_loss_samples':a['role_violations_during_wait'],
                'corrected_wait_role_loss_samples':wait_bad})
        interface=not row.get('used_global_recovery') or bool(row.get('table_cycle_verified'))
        hard=bool(row['blocked_calls'] or row['failure_reason']=='numerical_instability')
        c=bool(row['handover_completed']);p=bool(shared and obligation and interface and not hard)
        label='E' if not v else 'A' if c and p else 'B' if c else 'C' if p else 'D'
        if label!=a['label'] or bool(v and c and p)!=a['protected_completion']:
            changes.append({'seed':row['seed'],'method':row['method'],'window':row['window']})
        assert gap==a['maximum_registered_support_gap_ms']
        assert abs(a['total_seconds']-sum(a[k] for k in ['wait_seconds','recovery_motion_seconds','release_seconds','verification_seconds']))<1e-7
        assert not row['state_edit_count_after_initialization'] and not row['contact_anchor_count'] and not row['external_wrench_leak_steps']
        count+=1;labels[label]+=1
    assert count==720 and not changes and thresholds=={(.1,.2)}
    protocol=json.loads((STUDY/'protocol.json').read_text())
    hashes={name:hashlib.sha256((STUDY/'models'/name).read_bytes()).hexdigest() for name in protocol['model_hashes']}
    assert hashes==protocol['model_hashes']
    result={'episodes':count,'verdict_changes':changes,'labels':dict(labels),
        'boundary_obligation_corrections':len(boundary),'additional_violated_boundary_samples':total_new_bad-total_old_bad,
        'old_role_loss_samples':total_old_bad,'corrected_role_loss_samples':total_new_bad,
        'counter_changes':counter_changes,'corrected_boundaries':boundary,
        'verified_handover_clearance_threshold_m':.1,'verified_relative_speed_limit_m_per_s':.2,
        'native_recording_hz':20,'shared_max_allowed_registered_support_gap_ms':100,
        'shared_deep_interarm_penetration_limit_m':.01,'primary_role_gap_tolerance_ms':0,
        'invalid_admissions':invalid,'role_obligation_samples':dict(roles),
        'model_hashes_unchanged':hashes,
        'semantics':'E means any failed required admission, including an approved handover continuation. Legacy lifting local-prefix checks remain separately identified; no historical labels are replaced.'}
    (OUT/'contract-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='corrected_boundaries'},indent=2))


if __name__=='__main__':main()
