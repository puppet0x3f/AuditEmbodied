"""Summarize exact replay contacts, keeping finger bodies distinct from native pads."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import numpy as np

OUT=Path(__file__).resolve().parent
STAGES=('global_lower_to_table','global_table_loaded','global_release','global_table_settle',
    'global_clear','global_hover','global_descend','global_close','global_lift','global_lift_settle','final_stable_hold')


def main():
    all_cases={};endpoints=[]
    for scenario in ('left_loss','right_loss','sham'):
        case=f'50000-{scenario}-density150-actual-global'
        base=OUT/'contacts';meta=json.loads((base/(case+'.json')).read_text())
        assert meta['all_original_20hz_observations_match'] and meta['maximum_original_observation_error']==0
        samples=[json.loads(x) for x in (base/(case+'-samples.jsonl')).open()]
        contacts=defaultdict(list)
        for line in (base/(case+'-contacts.jsonl')).open():
            x=json.loads(line);contacts[x['time']].append(x)
        summary={};total_pad_samples=np.zeros(2,dtype=int)
        for stage in STAGES:
            frames=[x for x in samples if x['stage']==stage]
            for frame in frames:
                pads=np.zeros((2,2),dtype=int);pad_force=np.zeros((2,2));palm=0.;finger_body=0.
                for contact in contacts[frame['time']]:
                    other=contact['other_geom'];obj=contact['object_geom'];force=contact['force_norm_N']
                    match=re.fullmatch(r'gripper([01])_right_finger([12])_pad_collision',other)
                    if match:
                        arm,finger=int(match[1]),int(match[2])-1
                        if obj.startswith(f'pot_handle{arm}'):
                            pads[arm,finger]+=1;pad_force[arm,finger]+=force
                    elif other.endswith('_hand_collision'):palm+=force
                    elif re.fullmatch(r'gripper[01]_right_finger[12]_collision',other):finger_body+=force
                registered=(pads>0).all(1)
                assert registered.tolist()==frame['holds'],(case,stage,registered,frame['holds'])
                frame['assigned_handle_pad_contacts']=pads.tolist()
                frame['assigned_handle_pad_force_norm_N']=pad_force.tolist()
                frame['palm_contact_force_norm_N']=palm;frame['finger_body_contact_force_norm_N']=finger_body
                total_pad_samples+=registered
            last=frames[-1]
            summary[stage]={'samples':len(frames),'end_tilt_deg':last['tilt_deg'],
                'end_holds':last['holds'],'end_handle_pad_contacts':last['assigned_handle_pad_contacts'],
                'both_native_grasp_samples':sum(all(x['holds']) for x in frames),
                'end_table_force_norm_sum_N':last['contact_force_norm_sums_N']['table'],
                'end_palm_force_norm_sum_N':last['palm_contact_force_norm_N'],
                'end_finger_body_force_norm_sum_N':last['finger_body_contact_force_norm_N'],
                'end_assigned_pad_force_norm_N':last['assigned_handle_pad_force_norm_N'],
                'palm_contact_samples':sum(x['palm_contact_force_norm_N']>1e-6 for x in frames),
                'end_gripper_actuator_forces_N':last['gripper_actuator_force_N']}
            endpoints.append({'case':case,'stage':stage,**summary[stage]})
        all_cases[case]={'metadata':meta,'stage_summary':summary,
            'native_grasp_reconstructed_from_assigned_fingerpads':True,
            'native_grasp_samples_by_arm':total_pad_samples.tolist()}
    result={'known_replays':3,'new_independent_policy_tests':0,'physics_steps_or_actions_changed':False,
        'original_20hz_observations_reproduced_exactly':True,'postfault_recorded_samples':759,
        'native_definition':'Both left/right fingerpad groups must contact the assigned handle. Finger bodies, palms, and force magnitudes alone do not establish native grasp.',
        'causal_scope':'The observed failures arise before a registered two-handle regrasp. These logs identify pose/registration mismatch and incidental contacts, not an isolated mass-capacity law or a tested controller repair.',
        'cases':all_cases}
    (OUT/'contact-attribution.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'contact-stage-endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(endpoints[0]));w.writeheader();w.writerows(endpoints)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
