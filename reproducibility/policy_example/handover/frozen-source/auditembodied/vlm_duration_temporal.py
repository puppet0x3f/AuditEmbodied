"""Legal temporal inputs and state-balanced intervention supervision for a VLM."""
from __future__ import annotations

from collections import Counter
import json
import random

import numpy as np

from .vlm_duration_lora import OUTCOME_INSTRUCTION,task_prompt_for


TEMPORAL_INSTRUCTION=OUTCOME_INSTRUCTION.replace(
    'Camera order: external front, external agent, left wrist, right wrist.',
    'Images 1-4 are BEFORE the executed hold; images 5-8 are AFTER it. Within each time: external front, external agent, left wrist, right wrist.').replace(
    'only listed local step counts vary.',
    'executed hold duration and listed local plan step counts may vary between examples.')


def before_key(row):
    return json.dumps([row['before_image'],row['before_robot'],
                       row['plan']['repair_arm'],row['plan']['keeper_arm']],sort_keys=True)


def temporal_features(row):
    arm=0 if row['plan']['repair_arm']=='left' else 1;keeper=1-arm
    x0=np.asarray(row['before_robot']['positions']);x1=np.asarray(row['robot']['positions'])
    g0=np.mean(np.abs(np.asarray(row['before_robot']['fingers'])),axis=1)
    g1=np.mean(np.abs(np.asarray(row['robot']['fingers'])),axis=1)
    return {'executed_hold_steps':int(row['wait_steps']),'executed_hold_seconds':row['wait_steps']/20,
        'repair_relative_to_keeper_before_m':(x0[arm]-x0[keeper]).tolist(),
        'repair_relative_to_keeper_after_m':(x1[arm]-x1[keeper]).tolist(),
        'relative_displacement_change_m':((x1[arm]-x1[keeper])-(x0[arm]-x0[keeper])).tolist(),
        'keeper_displacement_m':(x1[keeper]-x0[keeper]).tolist(),
        'repair_keeper_finger_width_after_m':[float(g1[arm]),float(g1[keeper])],
        'repair_keeper_finger_width_change_m':[float(g1[arm]-g0[arm]),float(g1[keeper]-g0[keeper])]}


def temporal_prompt(row):
    return (TEMPORAL_INSTRUCTION+'\nPredict only the recovery AFTER the current observation. The preceding hold has already occurred.\n'
        +'Measured robot state BEFORE hold: '+json.dumps(row['before_robot'],separators=(',',':'))
        +'\nMeasured robot state AFTER hold: '+json.dumps(row['robot'],separators=(',',':'))
        +'\nMeasured temporal changes: '+json.dumps(temporal_features(row),separators=(',',':'))
        +'\nCandidate plan: '+json.dumps(row['plan'],separators=(',',':')))


def joint_action_features(row):
    """Role-relative observations and nominal plan kinematics at h2, before waiting."""
    arm = 0 if row['plan']['repair_arm'] == 'left' else 1
    keeper = 1 - arm
    previous, current = row['robot'], row['after_prefix_robot']
    x0, x1 = np.asarray(previous['positions']), np.asarray(current['positions'])
    f0, f1 = np.asarray(previous['fingers']), np.asarray(current['fingers'])
    span = x1[arm] - x1[keeper]
    theta = float(np.arctan2(-span[2], np.linalg.norm(span[:2])))
    duration = row['plan']['local_relevel_steps'] / 20
    roles = [arm, keeper]
    widths0, widths1 = np.mean(np.abs(f0), axis=1), np.mean(np.abs(f1), axis=1)
    return {'repair_relative_to_keeper_m': span.tolist(),
        'relative_motion_since_previous_observation_m': (span - (x0[arm] - x0[keeper])).tolist(),
        'keeper_motion_since_previous_observation_m': (x1[keeper] - x0[keeper]).tolist(),
        'repair_keeper_mean_abs_finger_encoder_m': widths1[roles].tolist(),
        'repair_keeper_finger_encoder_change_m': (widths1 - widths0)[roles].tolist(),
        'repair_keeper_finger_asymmetry_m': np.abs(f1.sum(axis=1))[roles].tolist(),
        'nominal_current_relevel_angle_rad': theta,
        'planned_relevel_seconds': duration,
        'nominal_angular_speed_rad_per_second': abs(theta) / duration,
        'nominal_arc_speed_m_per_second': float(np.linalg.norm(span)) * abs(theta) / duration,
        'planned_post_prefix_hold_seconds': row['plan']['post_prefix_wait_steps'] / 20}


def family_updates(train, steps, seed, balanced_replay=False, stratify_safe_sets=False):
    """Alternate complete duration families with uniform single-example cycles."""
    valid = [i for i, row in enumerate(train) if row['label'] != 'E']
    families = {}
    for i in valid:
        row = train[i]
        fixed_plan = {k: v for k, v in row['plan'].items() if k != 'local_relevel_steps'}
        key = json.dumps([row['image'], row['robot'], row['after_prefix_image'],
            row['after_prefix_robot'], row['late_history'], fixed_plan], sort_keys=True)
        families.setdefault(key, []).append(i)
    for group in families.values():
        group.sort(key=lambda i: train[i]['plan']['local_relevel_steps'])
        if len(group) != 3 or [train[i]['plan']['local_relevel_steps'] for i in group] != [30, 40, 60]:
            raise ValueError('family sampling needs executed30/40/60 candidates with identical observations and fixed plan')
    if not families:
        raise ValueError('no complete duration families')
    rng, family_queue, single_queue, updates = random.Random(seed), {}, {}, []
    groups = list(families.values())
    sources=('old','new') if balanced_replay else ('all',)
    singles={s:[i for i in valid if not balanced_replay or train[i].get('replay_source')==s] for s in sources}
    grouped={s:[g for g in groups if not balanced_replay or all(train[i].get('replay_source')==s for i in g)] for s in sources}
    if any(not singles[s] or not grouped[s] for s in sources):raise ValueError('replay needs complete old/new family pools')
    patterns={s:{} for s in sources};pattern_queue={}
    for source in sources:
        for group in grouped[source]:
            pattern=''.join('1' if train[i]['label']=='A' else '0' for i in group)
            patterns[source].setdefault(pattern,[]).append(group)
    for step in range(steps):
        source=sources[(step//2)%len(sources)]
        if step % 2 == 0:
            pool=source
            if stratify_safe_sets:
                if not pattern_queue.get(source):pattern_queue[source]=rng.sample(list(patterns[source]),len(patterns[source]))
                pattern=pattern_queue[source].pop();pool=(source,pattern);candidates=patterns[source][pattern]
            else:candidates=grouped[source]
            if not family_queue.get(pool):
                family_queue[pool] = rng.sample(candidates, len(candidates))
            indices = list(family_queue[pool].pop())
            kind = 'duration_family'
        else:
            if not single_queue.get(source):
                single_queue[source] = rng.sample(singles[source], len(singles[source]))
            indices, kind = [single_queue[source].pop()], 'uniform_single'
        updates.append({'kind': kind, 'task': 'outcome', 'indices': indices,
            'labels': [train[i]['label'] for i in indices]})
    counts = Counter(i for update in updates for i in update['indices'])
    return updates, {'valid_outcome_examples': len(valid), 'duration_families': len(groups),
        'update_counts': dict(Counter(u['kind'] for u in updates)),
        'presentations': sum(counts.values()), 'candidate_presentations': dict(counts),
        'all_valid_examples_presented': set(counts) == set(valid),
        'class_presentations': dict(Counter(c for u in updates for c in u['labels'])),
        'balanced_old_new_replay': balanced_replay,
        'stratified_safe_sets': stratify_safe_sets,
        'family_safe_set_presentations':dict(Counter(''.join('1' if c=='A' else '0' for c in u['labels'])
            for u in updates if len(u['indices'])==3)),
        'source_presentations': dict(Counter(train[i].get('replay_source','unspecified') for u in updates for i in u['indices'])),
        'note': 'Same h1/h2 and past history; only N varies within a family. W is held fixed. Uniform shuffled family/single cycles; no assumed monotonic safety.'}


def input_spec(row,task,mode):
    # The frozen prerequisite head always retains exactly its old inputs.
    if mode=='family_stream' and task=='start':
        from .vlm_duration_family import entry_example
        entry=entry_example(row)
        return [entry['image']],task_prompt_for(entry,task)
    if mode=='family_stream':raise ValueError('shared family outcomes use the three-position family encoder')
    if task!='start' and mode in ('stage_early','stage_late','stage_joint'):
        late=mode!='stage_early'
        instruction=OUTCOME_INSTRUCTION.replace(
            'Camera order: external front, external agent, left wrist, right wrist.',
            'Images1-4 are the previous observation and images5-8 the current observation. At each time: external front, external agent, left wrist, right wrist.').replace(
            'only listed local step counts vary.', 'the listed local step counts and planned post-prefix hold may vary.')
        phase=('The feedback approach/close prefix has ALREADY executed. Predict the whole post-fault episode, including any earlier guard violation, '
            'and the remaining planned hold, relevel and final verification. Do not execute the prefix again.' if late else
            'The feedback approach/close prefix has NOT executed yet. Predict the whole post-fault episode, including the '
            'planned prefix, post-prefix hold, relevel and final verification. Earlier executed holds also count.')
        if task == 'support_time':
            if not late or mode == 'stage_joint':
                raise ValueError('support-time supervision requires the fixed after-prefix observation')
            from .vlm_duration_support_time import SUPPORT_TIME_INSTRUCTION
            instruction = SUPPORT_TIME_INSTRUCTION
            phase = 'The remaining post-prefix hold, relevel and final verification have NOT executed yet.'
        previous=row['robot'] if late else row['before_robot']
        current=row['after_prefix_robot'] if late else row['robot']
        history=row['late_history'] if late else row['early_history']
        prompt=(instruction+'\n'+phase+'\nPrevious measured robot state: '+json.dumps(previous,separators=(',',':'))
            +'\nCurrent measured robot state: '+json.dumps(current,separators=(',',':'))
            +'\nObserved robot history up to CURRENT time: '+json.dumps(history,separators=(',',':'))
            +'\nCandidate plan: '+json.dumps(row['plan'],separators=(',',':')))
        if mode == 'stage_joint':
            prompt += ('\nRole-relative observations and nominal current-state plan kinematics: '
                + json.dumps(joint_action_features(row), separators=(',', ':'))
                + '\nThese nominal angles and speeds use ONLY the current measured robot pose. '
                'The planned waiting hold can change the pose before execution; these are not future measured coordinates. '
                'Judge task completion and ALL process guards jointly. A completed recovery can still violate its process guards.')
        paths=[row['image'],row['after_prefix_image']] if late else [row['before_image'],row['image']]
        return paths,prompt
    if task=='start' or mode!='paired':return [row['image']],task_prompt_for(row,task)
    return [row['before_image'],row['image']],temporal_prompt(row)


def risk_quota_updates(train,previous,updates,seed,threshold):
    """Replace single-example slots using TRAIN errors; keep every pair exact."""
    fields=('seed','scenario','image','robot','plan','label','before_image','wait_steps')
    if len(previous)!=len(train) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(train,previous)):
        raise ValueError('risk quota reference must match the training rows in order')
    valid=[i for i,r in enumerate(train) if r['label']!='E']
    hard={i for i in valid if train[i]['label']!='A' and previous[i]['class_probabilities'][0]>=threshold}
    pair_counts=Counter(i for u in updates if len(u['indices'])==2 for i in u['indices'])
    targets={i:8 if i in hard else 3 if train[i]['label']!='A' else 1 for i in valid}
    singles=[i for i in valid for _ in range(max(0,targets[i]-pair_counts[i]))]
    required=len(singles);slots=sum(len(u['indices'])==1 for u in updates)
    if required>slots:
        raise ValueError('risk quotas do not fit the fixed single-example budget')
    rejected_A=sorted((i for i in valid if train[i]['label']=='A' and
        previous[i]['class_probabilities'][0]<threshold),key=lambda i:previous[i]['class_probabilities'][0])
    positive_fill=rejected_A or [i for i in valid if train[i]['label']=='A']
    singles.extend(positive_fill[j%len(positive_fill)] for j in range(slots-required))
    random.Random(seed).shuffle(singles)
    revised=[]
    for update in updates:
        if len(update['indices'])==2:
            revised.append(dict(update))
        else:
            i=singles.pop()
            kind='hard_risk_quota' if i in hard else 'negative_quota' if train[i]['label']!='A' else 'positive_quota'
            revised.append({'kind':kind,'task':'outcome','indices':[i],'labels':[train[i]['label']]})
    counts=Counter(i for u in revised for i in u['indices'])
    return revised,{'update_counts':dict(Counter(u['kind'] for u in revised)),
        'candidate_presentations':dict(counts),'distinct_presented_examples':len(counts),
        'presentations':sum(counts.values()),'all_valid_examples_presented':set(counts)==set(valid),
        'class_presentations':dict(Counter(c for u in revised for c in u['labels'])),
        'risk_quota':{'threshold':threshold,'hard_indices':sorted(hard),'rejected_A_indices':rejected_A,
            'minimum_total_presentations':{'A':1,'non_A':3,'hard_non_A':8},
            'single_slots':slots,'required_single_slots':required,'positive_fill_slots':slots-required,
            'paired_updates_preserved':sum(len(u['indices'])==2 for u in revised),
            'all_quotas_met':all(counts[i]>=targets[i] for i in valid),
            'note':'Only prior TRAIN errors select hard negatives. Counts include unchanged paired updates. Remaining single slots replay rejected training A examples.'}}


def state_updates(train,steps,seed,strict_timing=False):
    """Three coverage updates per one genuine intervention/control pair."""
    valid=[i for i,r in enumerate(train) if r['label']!='E']
    states={};actions={};waits={}
    for i in valid:
        row=train[i];key=before_key(row);states.setdefault(key,[]).append(i)
        fixed={k:v for k,v in row['plan'].items() if k!='local_relevel_steps'}
        current=json.dumps([row['image'],row['robot'],fixed],sort_keys=True)
        actions.setdefault(current,[]).append(i)
        waits.setdefault((key,json.dumps(row['plan'],sort_keys=True)),[]).append(i)
    pools={'action_contrast':{},'wait_contrast':{},'safe_control':{}}
    def add(kind,a,b):pools[kind].setdefault(before_key(train[a]),[]).append((a,b))
    for group in actions.values():
        for j,a in enumerate(group):
            for b in group[j+1:]:
                if train[a]['plan']['local_relevel_steps']==train[b]['plan']['local_relevel_steps']:continue
                la,lb=train[a]['label'],train[b]['label']
                if la==lb=='A':add('safe_control',a,b)
                elif (la=='A')!=(lb=='A'):add('action_contrast',a,b) if la=='A' else add('action_contrast',b,a)
    for group in waits.values():
        for j,a in enumerate(group):
            for b in group[j+1:]:
                if train[a]['wait_steps']==train[b]['wait_steps']:continue
                la,lb=train[a]['label'],train[b]['label']
                if (la=='A')!=(lb=='A'):add('wait_contrast',a,b) if la=='A' else add('wait_contrast',b,a)
    if any(not pool for pool in pools.values()):raise ValueError('need executed action flips, wait flips and safe controls')
    rng=random.Random(seed);coverage=[];pair_queues={};updates=[]
    def coverage_cycle():
        queues={k:rng.sample(v,len(v)) for k,v in states.items()};result=[]
        while queues:
            order=list(queues);rng.shuffle(order)
            for key in order:
                result.append(queues[key].pop())
                if not queues[key]:del queues[key]
        return list(reversed(result))
    for step in range(steps):
        if step%4!=3:
            if not coverage:coverage=coverage_cycle()
            indices=[coverage.pop()];kind='state_coverage'
        else:
            kind=('action_contrast','wait_contrast','safe_control')[(step//4)%3]
            if not pair_queues.get(kind):
                order=list(pools[kind]);rng.shuffle(order);pair_queues[kind]=order
            key=pair_queues[kind].pop();indices=list(rng.choice(pools[kind][key]))
        updates.append({'kind':kind,'task':'outcome','indices':indices,'labels':[train[i]['label'] for i in indices]})
    strict_counts=Counter()
    if strict_timing:
        # Replace only action-pair slots. Coverage, wait pairs and A/A controls
        # remain identical to the original fixed-seed plan and budget.
        strict=sorted({(a,b) for group in pools['action_contrast'].values() for a,b in group
            if {train[a]['plan']['local_relevel_steps'],train[b]['plan']['local_relevel_steps']}=={40,60}
            and before_key(train[a])==before_key(train[b]) and train[a]['wait_steps']==train[b]['wait_steps']})
        slots=[u for u in updates if u['kind']=='action_contrast']
        if not strict or len(slots)<len(strict):
            raise ValueError('strict timing coverage needs at least one action-pair slot per observed 40/60 flip')
        pair_rng=random.Random(seed);queue=[]
        for update in slots:
            if not queue:queue=strict.copy();pair_rng.shuffle(queue)
            pair=queue.pop();strict_counts[pair]+=1
            update.update(kind='strict_timing',indices=list(pair),labels=[train[i]['label'] for i in pair])
    counts=Counter(i for r in updates for i in r['indices'])
    return updates,{'valid_outcome_examples':len(valid),'initial_fault_role_groups':len(states),
        'update_counts':dict(Counter(r['kind'] for r in updates)),
        'pair_counts':{k:sum(map(len,p.values())) for k,p in pools.items()},
        'pair_state_counts':{k:len(p) for k,p in pools.items()},
        'candidate_presentations':dict(counts),'distinct_presented_examples':len(counts),
        'presentations':sum(counts.values()),'all_valid_examples_presented':set(counts)==set(valid),
        'strict_timing_coverage':strict_timing,
        'strict_40_60_pair_presentations':[{'indices':list(pair),'updates':n} for pair,n in sorted(strict_counts.items())],
        'note':'State-balanced full coverage cycles. Action pairs share h1 and vary only relevel steps; wait pairs share verified h0 and full plan and vary executed hold. No fabricated positive, no order imposed on A/A.'}
