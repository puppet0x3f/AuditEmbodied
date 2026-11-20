"""Frozen initial policy versus unconditional and learned midcourse escalation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time

from .duration_aligned_loop import read_response
from .duration_closed_loop import trial_summary
from .duration_family_loop import family_trial
from .l4_lift import LiftConfig
from .duration_pilot import file_hash,STATE_KEYS,prefix_frames
from .l0 import _import_robosuite
from .mid_recovery_data import read


def summary(rows):
    result=trial_summary(rows);mid=[r['mid_selection'] for r in rows if 'mid_selection' in r]
    result.update(mid_decisions=len(mid),mid_actions=dict(Counter(r['selected_branch'] for r in mid)),
        mid_timely=sum(r['forecast_ready'] for r in mid),
        mid_stale=sum(not r['continuation_matches_forecast'] for r in mid),
        mid_baseline_fallbacks=sum(r['baseline_fallback'] for r in mid),
        mid_model_admissions=sum(r['decision'].get('model_admitted',False) and not r['baseline_fallback'] for r in mid),
        mid_explicit_rejections=sum(r['selected_branch']=='reject' for r in mid),
        duration_counts=dict(Counter(str(r.get('local_steps',{}).get('local_relevel_steps')) for r in rows)),
        abstentions=sum(bool(r.get('abstained')) or r['candidate']=='abstain' for r in rows),
        mean_wait_seconds=sum(r['inference_wait_seconds'] for r in rows)/len(rows),
        controller_wait_steps=sum(r.get('controller_compute_wait_steps',0) for r in rows))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--mid-adapter',required=True)
    parser.add_argument('--adapter',default='outputs/vlm-duration-family-stream-20261109/set512/adapter')
    parser.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    parser.add_argument('--model-path',default='outputs/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--gpu',default='1');parser.add_argument('--gpu-python',default='/opt/miniconda3/bin/python')
    parser.add_argument('--seeds',type=int,nargs='+',required=True);parser.add_argument('--engineering',action='store_true')
    parser.add_argument('--warm-data',default='outputs/vlm-mid-continuation-20261109/dataset/train.jsonl')
    args=parser.parse_args();output=Path(args.output_dir).resolve()
    if (output/'trials.jsonl').exists():raise ValueError('trial file exists')
    (output/'observations').mkdir(parents=True,exist_ok=True)
    point=json.loads(Path(args.working_point).read_text());frozen=[Path(args.working_point).resolve()]
    for folder in (point['start_adapter'],point['outcome_adapter'],args.adapter,args.mid_adapter):
        frozen.extend(Path(folder).resolve()/n for n in ('adapter_config.json','adapter_model.safetensors'))
    protocol={'seeds':args.seeds,'engineering_only':args.engineering,'threshold':point['threshold'],
        'methods':['family_set','mid_always','mid_selective'],'observation_step':20,'decision_step':30,
        'transition':'defer_rotation','frozen':{str(p):file_hash(p) for p in frozen},
        'scope':'Same frozen initial scope and duration head; new adapter only for N60. All physical waits, final holds and process violations accumulate. Late/stale mid replies use repaired global, separately labelled.'}
    protocol['physical_config']=asdict(LiftConfig())
    (output/'source').mkdir(exist_ok=True)
    protocol['source']={}
    for name in ('l4_lift.py','vlm_duration_policy.py','vlm_duration_family.py','vlm_mid_recovery.py','duration_family_loop.py','mid_recovery_loop.py'):
        path=Path(__file__).resolve().parent/name
        (output/'source'/name).write_bytes(path.read_bytes());protocol['source'][str(path)]=file_hash(path)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    rows=[]
    with (output/'worker.log').open('w') as log:
        worker=subprocess.Popen([args.gpu_python,'-m','auditembodied.vlm_duration_policy','--working-point',args.working_point,
            '--model-path',args.model_path,'--post-wait-adapter',args.adapter,'--post-wait-mode','family_stream',
            '--mid-adapter',args.mid_adapter],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        def message(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();return read_response(worker)[0]
        try:
            ready,_=read_response(worker);assert ready.get('ready'),ready
            warm=json.loads(Path('outputs/vlm-duration-family-stream-20261109/dataset/train.jsonl').read_text().splitlines()[0])
            for request in ({'op':'family_prefill','example':warm},
                {'op':'family_tail','example':warm,'attempt_id':0,'plans':[{**warm['plan'],'local_relevel_steps':n} for n in (30,40,60)]},
                {'op':'family_release'}):message(request)
            warm_mid=read(args.warm_data)[0];profile=[]
            for repeat in range(3):
                prefix=message({'op':'mid_prefill','example':warm_mid})
                cached=message({'op':'mid_tail','example':warm_mid,'attempt_id':0})
                fresh=message({'op':'mid_fresh','example':warm_mid,'attempt_id':0})
                error=max(abs(a-b) for x,y in zip(cached['records'],fresh['records']) for a,b in zip(x['class_probabilities'],y['class_probabilities']))
                assert error<=1e-6,error
                assert abs(prefix['start_valid_score']-warm_mid['start_valid_score'])<1e-6
                profile.append({'prefill_seconds':prefix['model_pipeline_seconds'],'tail_seconds':cached['model_pipeline_seconds'],
                    'fresh_split_seconds':fresh['model_pipeline_seconds'],'cached_max_probability_error':error})
            message({'op':'mid_release'});ready['mid_train_only_profile']=profile
            (output/'setup.json').write_text(json.dumps(ready,indent=2)+'\n')
            _import_robosuite(True)
            with (output/'trials.jsonl').open('w') as stream:
                for index,(seed,scenario) in enumerate((s,q) for s in args.seeds for q in ('left_loss','right_loss')):
                    methods=['none','always','selective'];shift=(index//2+index%2)%3;order=methods[shift:]+methods[:shift];pair=[]
                    for strategy in order:
                        row=family_trial(seed,scenario,output,worker,mid_strategy=strategy)
                        assert not row.get('error') and not row['blocked_calls'],row.get('error')
                        assert row['state_edit_count_after_initialization']==row['contact_anchor_count']==row['external_wrench_at_end']==0
                        assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-7
                        pair.append(row);rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                        print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','inference_wait_seconds','total_post_fault_seconds')},ensure_ascii=False),flush=True)
                    assert all(all(pair[0][k]==r[k] for k in STATE_KEYS) and prefix_frames(pair[0])==prefix_frames(r) for r in pair[1:])
        finally:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                except BrokenPipeError:pass
                worker.wait(timeout=30)
    assert all(file_hash(p)==h for p,h in protocol['frozen'].items())
    report={'engineering_only':args.engineering,'episodes':len(rows),'frozen_unchanged':True,
        'all_waits_physically_audited':True,'methods':{k:summary([r for r in rows if r['method']==k]) for k in protocol['methods']}}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
