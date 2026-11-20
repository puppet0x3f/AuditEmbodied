"""One causal context, three native-token outcome readouts, and reusable KV.

Only h0 RGB, measured robot states and known controller actions are encoded.
Training labels never appear in the context or in an earlier readout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DURATIONS = (60, 40, 30)
INSTRUCTION = """Predict executed recovery outcomes for two arms lifting a handled pot.
Camera order: external front, external agent, REPAIR wrist, KEEPER wrist.
The images were acquired at h0, before the initial decision hold. Later robot measurements
are explicitly timestamped; the images are not a current observation of contact.
Goal: both handles grasped, clearance >=0.10m, tilt <=15deg, handle height difference
<=0.03m, stable for one second. Preserve the keeper grasp throughout local recovery;
avoid deep inter-arm collision and unsupported intervals longer than0.10s.
Predict the WHOLE post-fault episode, including earlier holds and the remaining
committed closing actions, the listed post-prefix hold, relevel and final verification.
Condition on the keeper prerequisite being present at h0. Its later survival is uncertain.
A: completed and all guards passed. B: completed but a guard was violated.
C: incomplete and guards passed. D: incomplete and a guard was violated.
All candidates use the same geometry, controller gains and force limits at20Hz.
Read each candidate independently at its assistant query position using A/B/C/D.
Other assistant turns are intentionally empty: no other candidate answer is supplied."""


def entry_example(row):
    """The unchanged prerequisite head is evaluated at h0, not at a later phase."""
    return {**row, 'image': row['before_image'], 'robot': row['before_robot']}


def compact_state(robot, previous, repair_arm, elapsed):
    arm = 0 if repair_arm == 'left' else 1
    keeper = 1 - arm
    x, old = np.asarray(robot['positions']), np.asarray(previous['positions'])
    ref = np.asarray(robot['reference_positions'])
    fingers, old_fingers = np.asarray(robot['fingers']), np.asarray(previous['fingers'])
    horizontal = (ref[arm] - ref[keeper])[:2]
    horizontal = horizontal / np.linalg.norm(horizontal) if np.linalg.norm(horizontal) > 1e-9 else np.array([1., 0.])
    basis = np.array([[*horizontal, 0.], [-horizontal[1], horizontal[0], 0.], [0., 0., 1.]])
    span = x[arm] - x[keeper]
    roles = [arm, keeper]
    mm = lambda a: np.round(np.asarray(a) * 1000, 1).tolist()
    width = np.mean(np.abs(fingers), axis=1)
    old_width = np.mean(np.abs(old_fingers), axis=1)
    return {'repair_from_keeper_mm': mm(basis @ span),
        'span_reference_error_mm': mm(basis @ (span - (ref[arm] - ref[keeper]))),
        'keeper_reference_height_error_mm': round(float(x[keeper, 2] - ref[keeper, 2]) * 1000, 1),
        'keeper_motion_mm': mm(basis @ (x[keeper] - old[keeper])),
        'relative_motion_mm': mm(basis @ (span - (old[arm] - old[keeper]))),
        'repair_keeper_finger_width_mm': mm(width[roles]),
        'repair_keeper_finger_change_mm': mm((width - old_width)[roles]),
        'repair_keeper_finger_asymmetry_mm': mm(np.abs(fingers.sum(axis=1))[roles]),
        'tilt_deg': round(float(np.degrees(np.arctan2(-span[2], np.linalg.norm(span[:2])))), 1),
        'span_length_mm': round(float(np.linalg.norm(span)) * 1000, 1),
        'motion_interval_s': round(float(elapsed), 3)}


def prefix_text(row):
    state = compact_state(row['robot'], row['before_robot'], row['plan']['repair_arm'],
                          row['entry_seconds_after_h0'])
    return INSTRUCTION + '\nRoles: ' + json.dumps({k: row['plan'][k] for k in ('repair_arm', 'keeper_arm')}) + \
        '\nMeasured entry state after h0: ' + json.dumps(state, separators=(',', ':'))


def tail_text(row):
    state = compact_state(row['lookahead_robot'], row['lookahead_previous_robot'],
        row['plan']['repair_arm'], row['lookahead_delta_seconds'])
    plans = [{'N': n, 'seconds': n / 20, 'speed_ratio_to60': 60 / n,
        'nominal_angular_speed_deg_s': round(abs(state['tilt_deg']) / (n / 20), 2)} for n in DURATIONS]
    return ('Current measured closure state: ' + json.dumps(state, separators=(',', ':')) +
        '\nKnown controller continuation: ' + json.dumps({
            'stage': row['lookahead_stage'], 'remaining_close_steps': row['remaining_close_steps'],
            'image_age_s': round(row['lookahead_seconds_after_h0'], 3),
            'executed_entry_hold_steps': row.get('prefix_observation_wait_steps', 0),
            'planned_post_prefix_hold_steps': row['plan']['post_prefix_wait_steps'],
            'relevel_settle_steps': row['plan'].get('final_settle_steps', 8),
            'final_verification_steps': row['plan'].get('final_hold_steps', 20)}, separators=(',', ':')) +
        '\nCandidate family: ' + json.dumps(plans, separators=(',', ':')) +
        '\nThe nominal speeds use the current pose, not a future measured pose. '
        'The remaining closing actions will execute before the selected relevel.')


def family_key(row):
    fixed = {k: v for k, v in row['plan'].items() if k != 'local_relevel_steps'}
    return json.dumps([row['before_image'], row['before_robot'], row['robot'],
        row['lookahead_robot'], row['lookahead_previous_robot'], row['lookahead_stage'],
        row['lookahead_seconds_after_h0'], fixed], sort_keys=True)


def encode_prefix(processor, row, device=None):
    from PIL import Image
    with np.load(row['before_image']) as archive:
        order = [0, 1, 2, 3] if row['plan']['repair_arm'] == 'left' else [0, 1, 3, 2]
        images = [Image.fromarray(archive['rgb'][i].copy()) for i in order]
    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prefix_text(row)}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=images, return_tensors='pt')
    return inputs.to(device) if device else inputs


def encode_tail(processor, row, device=None):
    import torch
    # Special chat boundaries prevent tokenizer merges across the cached prefix.
    tokens = processor.tokenizer.encode('<|im_start|>user\n' + tail_text(row) + '<|im_end|>\n',
        add_special_tokens=False)
    positions = []
    for n in DURATIONS:
        query = (f'<|im_start|>user\nPredict the joint outcome for N={n}. Answer A/B/C/D.'
                 '<|im_end|>\n<|im_start|>assistant\n')
        tokens.extend(processor.tokenizer.encode(query, add_special_tokens=False))
        positions.append(len(tokens) - 1)
        tokens.extend(processor.tokenizer.encode('<|im_end|>\n', add_special_tokens=False))
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
            'mm_token_type_ids': torch.zeros_like(ids)}, positions


def encode_full(processor, row, device=None):
    import torch
    prefix = encode_prefix(processor, row, device)
    tail, positions = encode_tail(processor, row, device)
    offset = prefix['input_ids'].shape[1]
    for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
        if key in prefix:
            prefix[key] = torch.cat([prefix[key], tail[key]], dim=1)
    return prefix, [offset + p for p in positions]


def full_logits(model, processor, row):
    import torch
    batch, positions = encode_full(processor, row, model.device)
    values = model(**batch, logits_to_keep=torch.tensor(positions, device=model.device), use_cache=False).logits[0].float()
    return dict(zip(DURATIONS, values))


def checkpoint_stateless_parts(model):
    """Checkpoint arithmetic modules, leaving mutable KV updates outside."""
    import torch
    from torch.utils.checkpoint import checkpoint
    text=model.get_base_model().model.language_model
    modules=[text.norm]
    for layer in text.layers:
        modules.extend([layer.mlp,layer.input_layernorm,layer.post_attention_layernorm])
        modules.extend(getattr(layer.self_attn,n) for n in ('q_proj','k_proj','v_proj','o_proj','q_norm','k_norm'))
    for module in modules:
        if getattr(module,'family_checkpointed',False):continue
        original=module.forward
        def forward(*args,_module=module,_original=original,**kwargs):
            if _module.training and torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        module.forward=forward;module.family_checkpointed=True


def stream_logits(model, processor, row):
    """Train/evaluate exactly the prefix-then-tail path used by serving.

    A monolithic BF16 pass need not round identically to two different sequence
    lengths. Gradients flow through the cached prefix here; it is not detached.
    Stateless-module checkpointing avoids mutating KV during recomputation.
    """
    import torch
    def compute():
        cache=prefill(model,processor,row)
        if model.training and torch.is_grad_enabled() and not hasattr(model,'family_prefix_gradient_norm'):
            key=cache['past_key_values'].layers[0].keys
            if not key.requires_grad:raise ValueError('training prefix KV was detached')
            key.register_hook(lambda gradient:setattr(model,'family_prefix_gradient_norm',float(gradient.float().norm())))
        values=cached_logits(model,processor,row,cache)
        return torch.stack([values[n] for n in DURATIONS])
    values=compute()
    return dict(zip(DURATIONS,values))


def prefill(model, processor, row):
    """Called only with a frozen outcome adapter; never reuse prerequisite KV."""
    batch = encode_prefix(processor, row, model.device)
    output = model(**batch, logits_to_keep=1, use_cache=True)
    base = model.get_base_model().model
    return {'past_key_values': output.past_key_values, 'length': batch['input_ids'].shape[1],
        'attention_mask': batch['attention_mask'], 'rope_deltas': base.rope_deltas.clone(),
        'adapter': model.active_adapter, 'prefix_text': prefix_text(row),
        'image': row['before_image']}


def cached_logits(model, processor, row, cache):
    import torch
    if model.active_adapter != cache['adapter']:
        raise ValueError('cached language KV belongs to a different adapter')
    if prefix_text(row) != cache['prefix_text'] or row['before_image'] != cache['image']:
        raise ValueError('cached visual/robot entry context changed')
    tail, positions = encode_tail(processor, row, model.device)
    past = cache['past_key_values']
    past.crop(cache['length'])
    model.get_base_model().model.rope_deltas = cache['rope_deltas'].clone()
    tail['attention_mask'] = torch.cat([cache['attention_mask'], tail['attention_mask']], dim=1)
    positions3d=torch.arange(cache['length'],cache['length']+tail['input_ids'].shape[1],device=model.device)
    tail['position_ids']=positions3d.view(1,1,-1).expand(3,1,-1)+cache['rope_deltas'].to(model.device).view(1,1,1)
    try:
        output = model(**tail, past_key_values=past,
            logits_to_keep=torch.tensor(positions, device=model.device), use_cache=True)
        return dict(zip(DURATIONS, output.logits[0].float()))
    finally:
        # Each attempt appends a fresh tail to the same immutable entry context.
        past.crop(cache['length'])


def forecast(values, answer_ids, plans, start_score, threshold):
    """Model admission and baseline fallback are reported separately."""
    import torch
    records = []
    for p in sorted(plans, key=lambda p: p['local_relevel_steps']):
        conditional = torch.softmax(values[p['local_relevel_steps']][answer_ids[:4]], 0).detach().cpu().tolist()
        probabilities = [start_score * x for x in conditional] + [1 - start_score]
        records.append({'plan': p, 'conditional_probabilities': conditional,
            'class_probabilities': probabilities, 'start_valid_score': start_score,
            'accepted': probabilities[0] >= threshold})
    admitted = next((r for r in records if r['accepted']), None)
    # Strong rejection is different from an inconclusive high-threshold score.
    rejected = all(1 - r['class_probabilities'][0] >= threshold for r in records)
    return {'records': records, 'model_admitted': admitted is not None,
        'explicit_local_rejection': admitted is None and rejected,
        'baseline_fallback': admitted is None and not rejected,
        'accepted': admitted is not None or not rejected,
        'local_relevel_steps': admitted['plan']['local_relevel_steps'] if admitted else 60}
