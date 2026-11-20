"""Collect 144 train-only interventions and form temporal VLM examples."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

from .duration_acquire import before_relevel
from .duration_pilot import SOURCE_FILES,STATE_KEYS,file_hash,prefix_frames
from .duration_wait_matrix import collect_state,continuation_outcome
from .vlm_duration_data import outcome,plan,read_rows


def validate(rows):
    groups={};faults={};errors=[];cases=[]
    for r in rows:
        faults.setdefault((r['seed'],r['scenario']),[]).append(r)
        groups.setdefault((r['seed'],r['scenario'],r['wait_steps']),[]).append(r)
    for key,group in groups.items():
        ref=group[0];c=ref.get('capture',{})
        if sorted(r.get('local_steps',{}).get('local_relevel_steps',-1) for r in group)!=[30,40,60]:
            errors.append([*key,'candidate_set']);continue
        labels={};whole={}
        for r in group:
            n=r['local_steps']['local_relevel_steps']
            if r.get('error') or not r.get('fault_class_valid') or r.get('blocked_calls'):
                errors.append([*key,n,'execution',r.get('error')]);continue
            if any(r[f]!=ref[f] for f in STATE_KEYS) or prefix_frames(r)!=prefix_frames(ref):errors.append([*key,n,'fault_prefix'])
            if before_relevel(r)!=before_relevel(ref):errors.append([*key,n,'relevel_prefix'])
            if r['post_wait_state']!=c.get('state') or r['prefix_state']!=c.get('before_state'):errors.append([*key,n,'capture_state'])
            if r['duration_selection']['input']!=c.get('robot') or r['selection']['input']!=c.get('before_robot'):
                errors.append([*key,n,'robot_observation'])
            waits=[f for f in r['trace'] if f['stage']=='inference_wait']
            last=prefix_frames(r)[-1]['robot']['action']
            if (len(waits)!=key[2] or any(f['phase']!='recovery' for f in waits) or
                any([f['robot']['action'][i] for i in (6,13)]!=[last[i] for i in (6,13)] for f in waits) or
                abs(r['inference_wait_seconds']-key[2]/20)>1e-8 or
                abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])>1e-8):
                errors.append([*key,n,'wait_cost_ledger'])
            if r['state_edit_count_after_initialization'] or r['contact_anchor_count'] or r['external_wrench_at_end']:
                errors.append([*key,n,'state_edit'])
            label,guards=continuation_outcome(r);r['continuation_label']=label;r['continuation_protection']=guards
            labels[str(n)]=label;whole[str(n)]=outcome(r)
        cases.append({'seed':key[0],'scenario':key[1],'wait_steps':key[2],
            'continuation_labels':labels,'whole_episode_labels':whole,'wait_protection':ref['pre_skill_protection']})
    for key,group in faults.items():
        ref=group[0]
        if any(r['prefix_state']!=ref['prefix_state'] or prefix_frames(r)!=prefix_frames(ref) for r in group):
            errors.append([*key,'different_waits_do_not_share_h0'])
    return {'valid':not errors,'physical_episodes':len(rows),'fault_states':len(faults),
        'post_wait_observations':len(groups),'errors':errors,'cases':cases,
        'classes':dict(Counter(r.get('continuation_label','ERROR') for r in rows))}


def build_dataset(output,rows):
    original=Path('outputs/l4-duration-newstates-20261108/dataset')
    delayed=Path('outputs/l4-duration-wait-matrix-20261108/dataset')
    destination=output/'dataset';destination.mkdir()
    counts={}
    for split in ('train','development'):
        old=read_rows(original/f'{split}.jsonl');wait=read_rows(delayed/f'{split}.jsonl')
        h0={}
        for r in old:
            key=r['seed'],r['scenario']
            if key in h0 and (r['image']!=h0[key]['image'] or r['robot']!=h0[key]['robot']):
                raise ValueError('ambiguous original observation')
            h0[key]=r
        examples=[]
        for r in old:
            if r['plan']['kind']!='local':continue
            examples.append({**r,'before_image':r['image'],'before_robot':r['robot'],'wait_steps':0,'source':'original_zero'})
        for r in wait:
            before=h0[r['seed'],r['scenario']]
            examples.append({**r,'before_image':before['image'],'before_robot':before['robot'],
                'wait_steps':r['preceding_wait_steps'],'source':'existing_wait'})
        if split=='train':
            captures={(r['seed'],r['scenario'],r['wait_steps']):r['capture'] for r in rows if 'capture' in r}
            # All waits in a fault share one verified h0, even though captured independently.
            first_before={}
            for r in rows:
                c=captures[r['seed'],r['scenario'],r['wait_steps']]
                before=first_before.setdefault((r['seed'],r['scenario']),c)
                examples.append({'seed':r['seed'],'scenario':r['scenario'],'image':c['observation_file'],
                    'robot':c['robot'],'before_image':before['before_image'],'before_robot':before['before_robot'],
                    'wait_steps':r['wait_steps'],'plan':plan(r),'label':r['continuation_label'],
                    'whole_episode_label':outcome(r),'source':'new_intervention'})
        if any(not Path(r[k]).is_file() for r in examples for k in ('image','before_image')):raise ValueError('missing RGB')
        (destination/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
        counts[split]={'examples':len(examples),'classes':dict(Counter(r['label'] for r in examples)),
            'sources':dict(Counter(r['source'] for r in examples)),'seeds':sorted({r['seed'] for r in examples})}
    if set(counts['train']['seeds'])&set(counts['development']['seeds']):raise ValueError('split overlap')
    (destination/'data-report.json').write_text(json.dumps(counts,indent=2)+'\n')
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise ValueError('collection exists')
    for w in (0,10,20):(output/'observations'/str(w)).mkdir(parents=True)
    source=Path(__file__).resolve().parent
    protocol={'train_seeds':list(range(40,48)),'scenarios':['left_loss','right_loss'],
        'wait_steps':[0,10,20],'relevel_steps':[30,40,60],'physical_episodes':144,
        'source_sha256':{str(source/n):file_hash(source/n) for n in (*SOURCE_FILES,'duration_wait_matrix.py','duration_temporal_data.py')},
        'scope':'Train-only physical interventions. No model decisions or new test outcomes. Hold commands and full cost audited; continuation labels start at the post-wait observation. All failures retained.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[('train',seed,scenario,str((output/'observations'/str(w)).resolve()),w,True)
        for seed in range(40,48) for scenario in ('left_loss','right_loss') for w in (0,10,20)]
    rows=[];started=time.perf_counter()
    with (output/'trials.jsonl').open('w') as f:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for group in pool.map(collect_state,jobs):
                for row in group:rows.append(row);f.write(json.dumps(row)+'\n')
                f.flush();print(json.dumps({'seed':group[0]['seed'],'scenario':group[0]['scenario'],
                    'wait_steps':group[0]['wait_steps'],'completed':len(rows)}),flush=True)
    report=validate(rows)
    report['sources_unchanged']=all(file_hash(p)==h for p,h in protocol['source_sha256'].items())
    report['valid']=report['valid'] and len(rows)==144 and report['sources_unchanged']
    report['wall_seconds']=time.perf_counter()-started
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['valid']:raise ValueError('invalid physical collection; see report')
    counts=build_dataset(output,rows)
    print(json.dumps({'physical_valid':report['valid'],'dataset':counts}),flush=True)


if __name__=='__main__':main()
