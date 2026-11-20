"""Two native outcome readouts for the remaining local/global physical suffixes."""
from __future__ import annotations

import json
import numpy as np

from .vlm_duration_family import compact_state

KEYS=('local','global')


def candidate_keys(row):
    return ('local','global','defer') if row.get('two_window') and row['query']['relevel_step']==20 else KEYS
INSTRUCTION="""Predict two physical continuations of a bimanual pot lift.
Images: external front, external agent, REPAIR wrist, KEEPER wrist. Images are old
h0 observations. Robot measurements are timestamped. A60-step local relevel has
started; after observation at step20 the robot will execute10 committed steps.
At step30 choose a suffix. LOCAL: finish30 relevel steps,8 settle steps,20 final hold.
GLOBAL: keep closing the grippers, lower60,loaded8,release14,settle6,clear16,hover24,
descend20,close22,lift50,settle12,final hold20. Preserve the entry wrist commands
through placement and clear; restore registered wrist commands smoothly in hover.
Global is not assumed safe. Predict each WHOLE episode including its executed prefix.
Goal: both handles grasped, clearance>=0.10m, tilt<=15deg, height difference<=0.03m,
stable for1 second. Keep the original keeper grasp until verified table support;
avoid deep interarm collision and unsupported intervals longer than0.10s.
Condition on the h0 keeper prerequisite. Past violations cannot be erased.
A: task complete, all process guards pass. B: complete, a guard violated.
C: incomplete, guards pass. D: incomplete, a guard violated.
Other assistant answers are empty; no other candidate label is supplied."""


def prefix_text(row):
    entry=row['entry']
    instruction=INSTRUCTION
    if row.get('two_window'):
        instruction=instruction.replace('Predict two physical continuations','Predict action- and time-conditioned physical continuations')
        instruction=instruction.replace('Images are old\nh0 observations.','Images are measured at relevel entry, AFTER the attempted regrasp.')
        instruction=instruction.replace('after observation at step20 the robot will execute10 committed steps.\nAt step30 choose a suffix. LOCAL: finish30 relevel steps,8 settle steps,20 final hold.',
            'each query specifies its observation step and decision step. Execute10 committed steps before choosing. LOCAL: finish the remaining relevel steps,8 settle steps,20 final hold.')
        instruction+='\nAt the step30 boundary DEFER means execute10 further local relevel steps then the SAME GLOBAL at step40. Predict the actual entire GLOBAL40 episode, including those extra10 steps. A later query may revise the deferred plan from new measurements.'
    return instruction+'\nRoles:'+json.dumps({k:row['plan'][k] for k in ('repair_arm','keeper_arm')})+\
        '\nMeasured at relevel start:'+json.dumps(compact_state(entry,row['before_robot'],
            row['plan']['repair_arm'],entry['seconds_after_h0']),separators=(',',':'))


def tail_text(row):
    q=row['query'];repair=0 if row['plan']['repair_arm']=='left' else 1;roles=[repair,1-repair]
    error=(np.asarray(q['command_positions'])-np.asarray(q['positions']))[roles]
    rotation_errors=[]
    for i in roles:
        delta=np.asarray(q['command_rotations'])[i]@np.asarray(q['rotations'])[i].T
        rotation_errors.append(round(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))),2))
    state=compact_state(q,q['previous_robot'],row['plan']['repair_arm'],q['delta_seconds'])
    observed=int(q.get('relevel_step',20)) if row.get('two_window') else 20
    decision=observed+10
    text=('Measured at relevel'+str(observed)+':'+json.dumps(state,separators=(',',':'))+
        '\nMeasured command-following error:'+json.dumps({'repair_keeper_xyz_mm':np.round(error*1000,1).tolist(),
            'repair_keeper_rotation_deg':rotation_errors},separators=(',',':'))+
        '\nKnown continuation:'+json.dumps({'observed_step':observed,**({'decision_step':decision} if row.get('two_window') else {}),'committed_steps_before_decision':10,
            'image_age_s':round(q['seconds_after_h0']-(row['entry']['seconds_after_h0'] if row.get('two_window') else 0),3),
            'local_remaining_s_at_decision':(60-decision+8+20)/20,
            'global_remaining_s_at_decision':12.6,
            **({'defer_global40_remaining_s_at_decision':13.1} if row.get('two_window') and observed==20 else {})},separators=(',',':')))
    if row.get('two_window'):
        from scipy.spatial.transform import Rotation
        current=np.asarray(q['wrench'])[roles];entry=np.asarray(row['entry']['wrench'])[roles]
        history=q['sensor_history'];first=np.asarray(history[0]['wrench'])[roles]
        age=max(q['sim_time']-history[0]['time'],.05)
        ref=np.asarray(q['reference_positions']);horizontal=(ref[roles[0]]-ref[roles[1]])[:2]
        horizontal=horizontal/np.linalg.norm(horizontal)
        basis=np.array([[*horizontal,0.],[-horizontal[1],horizontal[0],0.],[0.,0.,1.]])
        def canonical(w):return np.concatenate([w[:,:3]@basis.T,w[:,3:]@basis.T],axis=1)
        current,entry,first=map(canonical,(current,entry,first))
        drift=[];signed_error=[]
        for i in roles:
            drift.append(np.round(basis@Rotation.from_matrix(np.asarray(q['rotations'])[i]@np.asarray(row['entry']['rotations'])[i].T).as_rotvec()*180/np.pi,1).tolist())
            signed_error.append(np.round(basis@Rotation.from_matrix(np.asarray(q['command_rotations'])[i]@np.asarray(q['rotations'])[i].T).as_rotvec()*180/np.pi,1).tolist())
        text+='\nMeasured repair/keeper sensors:'+json.dumps({
            'wrench_N_Nm':np.round(current,2).tolist(),'wrench_change_from_entry':np.round(current-entry,2).tolist(),
            'wrench_slope_per_s':np.round((current-first)/age,2).tolist(),
            'rotation_change_deg':drift,'signed_command_rotation_error_deg':signed_error},separators=(',',':'))
    return text


def encode_prefix(processor,row,device=None):
    from PIL import Image
    with np.load(row.get('entry_image',row['before_image'])) as data:
        order=[0,1,2,3] if row['plan']['repair_arm']=='left' else [0,1,3,2]
        images=[Image.fromarray(data['rgb'][i].copy()) for i in order]
    content=[{'type':'image'} for _ in images]+[{'type':'text','text':prefix_text(row)}]
    text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=False)
    batch=processor(text=[text],images=images,return_tensors='pt')
    return batch.to(device) if device else batch


def encode_tail(processor,row,device=None):
    import torch
    tokens=processor.tokenizer.encode('<|im_start|>user\n'+tail_text(row)+'<|im_end|>\n',add_special_tokens=False)
    positions=[]
    decision=int(row['query'].get('relevel_step',20))+10 if row.get('two_window') else 30
    for key in candidate_keys(row):
        tokens+=processor.tokenizer.encode(f'<|im_start|>user\nPredict {key.upper()} at the step{decision} boundary. Answer A/B/C/D.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
        positions.append(len(tokens)-1)
        tokens+=processor.tokenizer.encode('<|im_end|>\n',add_special_tokens=False)
    ids=torch.tensor([tokens],dtype=torch.long,device=device)
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'mm_token_type_ids':torch.zeros_like(ids)},positions


def prefill(model,processor,row):
    from .vlm_duration_family import prefill as cached_prefix
    return cached_prefix(model,processor,row,prefix_encoder=encode_prefix,prefix_renderer=prefix_text)


def cached_logits(model,processor,row,cache):
    from .vlm_duration_family import cached_logits as tail
    return tail(model,processor,row,cache,tail_encoder=encode_tail,prefix_renderer=prefix_text,keys=candidate_keys(row))


def stream_logits(model,processor,row):
    import torch
    cache=prefill(model,processor,row)
    if model.training and torch.is_grad_enabled() and not hasattr(model,'mid_prefix_gradient_norm'):
        key=cache['past_key_values'].layers[0].keys
        if not key.requires_grad:raise ValueError('mid prefix detached')
        key.register_hook(lambda g:setattr(model,'mid_prefix_gradient_norm',float(g.float().norm())))
    return cached_logits(model,processor,row,cache)


def select_forecast(records,threshold,policy='consistent'):
    if policy not in ('consistent','legacy'):raise ValueError('unknown mid admission policy')
    records=[{**r,'accepted':r['class_probabilities'][0]>=threshold} for r in records]
    accepted=next((r for r in records if r['accepted']),None)
    rejected=accepted is None and all(1-r['class_probabilities'][0]>=threshold for r in records)
    no_plan=accepted is None
    return {'records':records,'action':({'local':'continue','global':'R_global','defer':'defer'}[accepted['key']]) if accepted else
            ('reject' if policy=='consistent' or rejected else 'R_global'),
        'model_admitted':accepted is not None,'baseline_fallback':policy=='legacy' and no_plan and not rejected,
        'explicit_rejection':rejected,'no_admissible_plan':no_plan,
        'admission_policy':policy,'uncertainty_exit':policy=='consistent' and no_plan and not rejected}


def forecast(values,ids,start_score,threshold,policy='consistent'):
    import torch
    records=[]
    for key in values:
        conditional=torch.softmax(values[key][ids[:4]],0).detach().cpu().tolist()
        joint=[start_score*p for p in conditional]+[1-start_score]
        records.append({'key':key,'conditional_probabilities':conditional,'class_probabilities':joint,
            'accepted':joint[0]>=threshold})
    return select_forecast(records,threshold,policy)


def select_at_step(records, threshold, query_step, decision_step, prefer_defer=False):
    """Use only candidates whose original activation time is still reachable."""
    rows=[]
    for record in records:
        activation = 60 if record['key']=='local' else 40 if record['key']=='defer' else query_step+10
        feasible = record['key']=='local' or activation==decision_step or activation==40 and decision_step==30
        rows.append({**record,'activation_step':activation,'feasible_now':feasible,
            'accepted':record['class_probabilities'][0]>=threshold})
    allowed={r['key']:r for r in rows if r['accepted'] and r['feasible_now']}
    key=next((k for k in ('local','global','defer') if k in allowed),None)
    if prefer_defer and decision_step==30 and key=='global' and 'defer' in allowed:key='defer'
    selected=allowed.get(key)
    action='reject' if selected is None else 'continue' if key=='local' else 'defer' if selected['activation_step']>decision_step else 'R_global'
    return {'records':rows,'action':action,'model_admitted':selected is not None,
        'selected_key':key,'activation_step':None if selected is None else selected['activation_step'],
        'forecast_query_step':query_step,'baseline_fallback':False,'admission_policy':'consistent',
        'no_admissible_plan':selected is None,'uncertainty_exit':selected is None}
