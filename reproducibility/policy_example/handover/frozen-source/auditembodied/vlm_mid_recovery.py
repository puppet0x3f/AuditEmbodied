"""Two native outcome readouts for the remaining local/global physical suffixes."""
from __future__ import annotations

import json
import numpy as np

from .vlm_duration_family import compact_state

KEYS=('local','global')
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
    return INSTRUCTION+'\nRoles:'+json.dumps({k:row['plan'][k] for k in ('repair_arm','keeper_arm')})+\
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
    return ('Measured at relevel20:'+json.dumps(state,separators=(',',':'))+
        '\nMeasured command-following error:'+json.dumps({'repair_keeper_xyz_mm':np.round(error*1000,1).tolist(),
            'repair_keeper_rotation_deg':rotation_errors},separators=(',',':'))+
        '\nKnown continuation:'+json.dumps({'observed_step':20,'committed_steps_before_decision':10,
            'image_age_s':round(q['seconds_after_h0'],3),'local_remaining_s_at_decision':2.9,
            'global_remaining_s_at_decision':12.6},separators=(',',':')))


def encode_prefix(processor,row,device=None):
    from PIL import Image
    with np.load(row['before_image']) as data:
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
    for key in KEYS:
        tokens+=processor.tokenizer.encode(f'<|im_start|>user\nPredict {key.upper()} at the step30 boundary. Answer A/B/C/D.<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
        positions.append(len(tokens)-1)
        tokens+=processor.tokenizer.encode('<|im_end|>\n',add_special_tokens=False)
    ids=torch.tensor([tokens],dtype=torch.long,device=device)
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'mm_token_type_ids':torch.zeros_like(ids)},positions


def prefill(model,processor,row):
    from .vlm_duration_family import prefill as cached_prefix
    return cached_prefix(model,processor,row,prefix_encoder=encode_prefix,prefix_renderer=prefix_text)


def cached_logits(model,processor,row,cache):
    from .vlm_duration_family import cached_logits as tail
    return tail(model,processor,row,cache,tail_encoder=encode_tail,prefix_renderer=prefix_text,keys=KEYS)


def stream_logits(model,processor,row):
    import torch
    cache=prefill(model,processor,row)
    if model.training and torch.is_grad_enabled() and not hasattr(model,'mid_prefix_gradient_norm'):
        key=cache['past_key_values'].layers[0].keys
        if not key.requires_grad:raise ValueError('mid prefix detached')
        key.register_hook(lambda g:setattr(model,'mid_prefix_gradient_norm',float(g.float().norm())))
    return cached_logits(model,processor,row,cache)


def forecast(values,ids,start_score,threshold):
    import torch
    records=[]
    for key in KEYS:
        conditional=torch.softmax(values[key][ids[:4]],0).detach().cpu().tolist()
        joint=[start_score*p for p in conditional]+[1-start_score]
        records.append({'key':key,'conditional_probabilities':conditional,'class_probabilities':joint,
            'accepted':joint[0]>=threshold})
    accepted=next((r for r in records if r['accepted']),None)
    rejected=accepted is None and all(1-r['class_probabilities'][0]>=threshold for r in records)
    return {'records':records,'action':('continue' if accepted['key']=='local' else 'R_global') if accepted else
            ('reject' if rejected else 'R_global'),
        'model_admitted':accepted is not None,'baseline_fallback':accepted is None and not rejected,
        'explicit_rejection':rejected}
