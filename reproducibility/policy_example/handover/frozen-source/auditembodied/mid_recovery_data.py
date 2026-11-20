"""Actual paired continuations, with observations restricted to relevel20."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

from .duration_continuation import run_continuation
from .duration_pilot import file_hash


def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def outcome(row):
    return ('B' if row['protection_violation'] else 'A') if row['success'] else ('D' if row['protection_violation'] else 'C')


def job(spec):
    source,metadata,score=spec;packets={}
    def observe(operation,port,candidate):
        if operation=='finish':raise ValueError('fixed data replay must not request a learned decision')
        packets[operation]=port
    row=run_continuation((source,'R_global'),transition='defer_rotation',account_control_latency=True,observer=observe)
    row['mid_observations']=packets
    sample=None
    if not row['diagnostic_errors']:
        assert set(packets)=={'start','lookahead'}
        assert packets['lookahead']['relevel_step']==20
        boundary=row['continuation_diagnostic']
        assert abs(boundary['boundary_state']['time']-packets['lookahead']['sim_time']-.5)<1e-8
        sample={'seed':metadata['seed'],'scenario':metadata['scenario'],'split':metadata['split'],
            'before_image':metadata['before_image'],'before_robot':metadata['before_robot'],
            'plan':metadata['plan'],'entry':packets['start'],'query':packets['lookahead'],
            'start_valid_score':score,'labels':{'local':outcome(source),'global':outcome(row)},
            'source_wait_steps':metadata['wait_steps'],'source_image':metadata['image'],
            'costs':{'local':2.9,'global':12.6},'prefix_exact':True,
            'source_local_total_seconds':source['total_post_fault_seconds'],
            'actual_global_total_seconds':row['total_post_fault_seconds']}
    return row,sample


def load_jobs():
    root=Path('outputs/vlm-duration-family-stream-20261109');p=json.loads((root/'experiment-plan.json').read_text())
    physical=[]
    for w in (0,66):physical+=read(Path(p['source_collection'])/f'trials-wait{w}.jsonl')
    physical+=read(Path(p['source_new_collection'])/'trials.jsonl')
    by_key={(r['seed'],r['scenario'],r['post_prefix_wait_steps'],r['local_steps']['local_relevel_steps']):r for r in physical}
    jobs=[];excluded=[]
    for split,predfile in [('train','train-after-predictions.jsonl'),('development','after-predictions.jsonl')]:
        predictions={(r['seed'],r['scenario'],r['wait_steps'],r['plan']['local_relevel_steps']):r for r in read(root/'set512'/predfile)}
        for example in read(root/'dataset'/f'{split}.jsonl'):
            if example['plan']['local_relevel_steps']!=60:continue
            key=(example['seed'],example['scenario'],example['wait_steps'],60);source=by_key[key]
            if sum(f['stage']=='local_relevel' for f in source['trace'])!=60:
                excluded.append({'split':split,'seed':key[0],'scenario':key[1],'wait':key[2],'reason':'no complete N60 execution'})
                continue
            score=predictions[key]['start_valid_score']
            jobs.append((source,example,score))
    return jobs,excluded


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'trials.jsonl').exists():raise ValueError('collection exists')
    jobs,excluded=load_jobs()
    if args.smoke:jobs=[j for j in jobs if (j[0]['seed'],j[0]['scenario'],j[1]['wait_steps']) in ((43,'left_loss',66),(52,'left_loss',0))]
    source_root=Path('outputs/vlm-duration-family-stream-20261109')
    frozen=json.loads((source_root/'closed-loop/protocol.json').read_text())['frozen']
    assert all(file_hash(p)==d for p,d in frozen.items())
    protocol={'smoke_only':args.smoke,'families':len(jobs),'excluded':excluded,'frozen':frozen,
        'observation_step':20,'decision_step':30,'transition':'defer_rotation',
        'splits':dict(Counter(j[1]['split'] for j in jobs)),
        'scope':'Source train/development seeds only. All original fault/wait steps physically replayed; only new global suffix labels. Input ends at relevel20.'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    examples={'train':[],'development':[]};rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (output/'trials.jsonl').open('w') as stream:
            for row,sample in pool.map(job,jobs):
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                print(json.dumps({'seed':row['seed'],'scenario':row['scenario'],'wait':row['inference_wait_steps'],
                    'label':outcome(row),'errors':row['diagnostic_errors']}),flush=True)
                if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
                examples[sample['split']].append(sample)
    for split,a in examples.items():(output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in a))
    assert all(file_hash(p)==d for p,d in frozen.items())
    report={**protocol,'wall_seconds':time.perf_counter()-started,'all_prefixes_exact':True,'frozen_unchanged':True,
        'label_pairs':{s:dict(Counter(r['labels']['local']+r['labels']['global'] for r in a)) for s,a in examples.items()}}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
