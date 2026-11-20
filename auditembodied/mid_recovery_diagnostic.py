"""Frozen necessary-escalation diagnosis and search-exhaustion physical controls."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

from .duration_aligned_loop import read_response
from .duration_closed_loop import trial_summary
from .duration_continuation import run_continuation
from .duration_pilot import file_hash
from .l0 import _import_robosuite
from .mid_recovery_data import outcome,read
from .vlm_duration_data import plan


ROOT=Path('outputs/vlm-mid-continuation-20261109')
OLD=Path('outputs/vlm-duration-family-stream-20261109/closed-loop')
REFERENCE=Path('outputs/vlm-duration-defer-global-full13-20261109')
SEARCH_CASES=((28005,'right_loss'),(28007,'left_loss'),(28011,'left_loss'),(28011,'right_loss'))


def prepare(output,mode,cases,episodes):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'trials.jsonl').exists():raise ValueError('diagnostic trial file exists')
    frozen=json.loads((ROOT/'closed-loop/protocol.json').read_text())['frozen']
    assert all(file_hash(p)==h for p,h in frozen.items())
    protocol={'diagnostic_only':True,'use_for_training':False,'new_independent_test':False,
        'fine_tuning':False,'mode':mode,'cases':cases,'planned_episodes':episodes,
        'threshold':.999667229515218,'transition':'defer_rotation','frozen':frozen,
        'scope':'Existing states only. Replay original actuator commands and waits, preserve cumulative guards, charge all new control waits. Model language KV is specific to the frozen mid adapter.'}
    (output/'source').mkdir(exist_ok=True);protocol['runtime_sources']={}
    for name in ('l4_lift.py','duration_continuation.py','vlm_mid_recovery.py','vlm_duration_policy.py','mid_recovery_diagnostic.py'):
        p=Path(__file__).parent/name
        (output/'source'/name).write_bytes(p.read_bytes());protocol['runtime_sources'][str(p.resolve())]=file_hash(p)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


def finish(output,protocol,report):
    assert all(file_hash(p)==h for p,h in protocol['frozen'].items())
    report.update(diagnostic_only=True,new_independent_test=False,use_for_training=False,
        fine_tuning=False,frozen_unchanged=True,all_prefixes_exact=True,all_waits_audited=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def search_diagnostic(output,workers):
    sources={(r['seed'],r['scenario']):r for r in read(ROOT/'closed-loop/trials.jsonl') if r['method']=='mid_selective'}
    sources={k:sources[k] for k in SEARCH_CASES}
    assert all(not any(f['stage']=='local_relevel' for f in r['trace']) and not r['success'] for r in sources.values())
    protocol=prepare(output,'search_exhausted',list(SEARCH_CASES),8)
    protocol['source_trials']=str((ROOT/'closed-loop/trials.jsonl').resolve())
    protocol['source_sha256']=file_hash(ROOT/'closed-loop/trials.jsonl')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    run=partial(run_continuation,transition='defer_rotation',account_control_latency=True,boundary='search_exhausted')
    jobs=[(r,b) for r in sources.values() for b in ('continue','R_global')];rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (output/'trials.jsonl').open('w') as stream:
            for row in pool.map(run,jobs):
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                print(json.dumps({'seed':row['seed'],'scenario':row['scenario'],'branch':row.get('continuation_diagnostic',{}).get('branch'),
                    'label':outcome(row),'seconds':row.get('total_post_fault_seconds'),'errors':row['diagnostic_errors']}),flush=True)
                if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
                assert row['skill_failure']['reason']=='search_exhausted' and row['skill_failure']['attempts']==3
    pairs={}
    for row in rows:pairs.setdefault((row['seed'],row['scenario']),{})[row['continuation_diagnostic']['branch']]=row
    cases=[]
    for key,pair in pairs.items():
        a,b=pair['continue'],pair['R_global']
        for field in ('boundary_state','boundary_robot','boundary_position_targets','boundary_rotation_targets','protection_before','boundary_wait_steps'):
            assert a['continuation_diagnostic'][field]==b['continuation_diagnostic'][field],(key,field)
        cases.append({'seed':key[0],'scenario':key[1],'original_label':outcome(a),'global_label':outcome(b),
            'original_seconds':a['total_post_fault_seconds'],'global_seconds':b['total_post_fault_seconds'],
            'added_seconds':b['total_post_fault_seconds']-a['total_post_fault_seconds'],
            'global_keeper_lost_steps':b['protection']['keeper_grasp_lost_steps'],
            'global_table_cycle_verified':b['table_cycle_verified']})
    finish(output,protocol,{'episodes':len(rows),'cases':cases,
        'methods':{k:trial_summary([r for r in rows if r['continuation_diagnostic']['branch']==k]) for k in ('continue','R_global')},
        'automatic_takeover_default_enabled':False,
        'safe_takeover_in_all_diagnostic_states':all(c['global_label']=='A' for c in cases),
        'attribution':'The skill-failure signal is shared by policies. Automatic global takeover is an opt-in control diagnostic, not a VLM contribution; its physical result determines whether it can be enabled.'})


def mid_trial(source,worker):
    h0=source['acquisition']['h0'];entry={};pending={}
    def send(request):worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush()
    with ThreadPoolExecutor(max_workers=1) as reader:
        def callback(operation,port,candidate):
            if operation=='start':
                p=plan({'candidate':candidate,'local_steps':{'local_relevel_steps':60}})
                p.update(post_prefix_wait_steps=0,final_hold_steps=20)
                entry.update(before_image=h0['image'],before_robot=h0['robot'],plan=p,entry=port)
                tick=time.perf_counter();send({'op':'mid_prefill','example':entry})
                pending['prefill']=(tick,reader.submit(read_response,worker));return
            if operation=='lookahead':
                entry['query']=port;tick=time.perf_counter()
                send({'op':'mid_tail','example':entry,'attempt_id':port['attempt_id']})
                pending['tail']=(tick,reader.submit(read_response,worker));return
            pair=pending.get('tail')
            if pair is None or not pair[1].done():
                return {'action':'R_global','forecast_ready':False,'attempt_id':port['attempt_id'],
                    'baseline_fallback':True,'reason':'mid_forecast_not_ready'}
            response,arrived=pair[1].result()
            return {**response,'forecast_ready':True,'pipeline_wall_seconds':arrived-pair[0]}
        row=run_continuation((source,'model'),transition='defer_rotation',account_control_latency=True,
            observer=callback,pace_recovery=True)
        timings=[];ended=time.perf_counter()
        for kind,(issued,future) in pending.items():
            response,arrived=future.result();timings.append({'kind':kind,'pipeline_wall_seconds':arrived-issued,
                'worker_seconds':response['model_pipeline_seconds']})
        row['async_timing']=timings;row['post_episode_drain_seconds']=time.perf_counter()-ended
        if pending:send({'op':'mid_release'});read_response(worker)
    return row


def mid_diagnostic(output,args):
    sources=[r for r in read(OLD/'trials.jsonl') if r['method']=='family_set' and r.get('local_steps',{}).get('local_relevel_steps')==60]
    reference=json.loads((REFERENCE/'report.json').read_text())
    labels={(c['seed'],c['scenario']):c for c in reference['cases']}
    assert len(sources)==13 and len(labels)==13
    assert all(labels[r['seed'],r['scenario']]['continue_label']==outcome(r) for r in sources)
    protocol=prepare(output,'frozen_mid',[(r['seed'],r['scenario']) for r in sources],13)
    protocol.update(source_sha256=file_hash(OLD/'trials.jsonl'),reference_sha256=file_hash(REFERENCE/'trials.jsonl'),
        required_upgrades=7,safe_local_controls=6,observation_step=20,decision_step=30)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    point=ROOT/'../vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json'
    rows=[]
    with (output/'worker.log').open('w') as log:
        worker=subprocess.Popen([args.gpu_python,'-m','auditembodied.vlm_duration_policy',
            '--working-point',str(point),'--model-path',args.model_path,
            '--post-wait-adapter',str(OLD.parent/'set512/adapter'),'--post-wait-mode','family_stream',
            '--mid-adapter',str(ROOT/'set512/adapter')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        def message(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();return read_response(worker)[0]
        try:
            ready,_=read_response(worker);assert ready.get('ready'),ready
            warm=read(ROOT/'dataset/train.jsonl')[0]
            message({'op':'mid_prefill','example':warm});message({'op':'mid_tail','example':warm,'attempt_id':0});message({'op':'mid_release'})
            (output/'setup.json').write_text(json.dumps(ready,indent=2)+'\n')
            _import_robosuite(True)
            with (output/'trials.jsonl').open('w') as stream:
                for source in sources:
                    row=mid_trial(source,worker);key=(row['seed'],row['scenario']);known=labels[key]
                    row['reference_labels']={'local':known['continue_label'],'global':known['global_label']}
                    row['reference_global_total_seconds']=known['global_total_seconds']
                    stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                    print(json.dumps({'seed':row['seed'],'scenario':row['scenario'],'known':row['reference_labels'],
                        'decision':row.get('mid_selection',{}).get('selected_branch'),'label':outcome(row),
                        'seconds':row.get('total_post_fault_seconds'),'errors':row['diagnostic_errors']}),flush=True)
                    if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
        finally:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                except BrokenPipeError:pass
                worker.wait(timeout=30)
    required=[r for r in rows if r['reference_labels']['local']!='A'];controls=[r for r in rows if r['reference_labels']['local']=='A']
    def choice(r):return r['mid_selection']['selected_branch']
    def fallback(r):return r['mid_selection']['baseline_fallback']
    def admitted(r):return r['mid_selection']['decision'].get('model_admitted',False) and not fallback(r)
    metrics={'required_states':len(required),'required_model_global_choices':sum(choice(r)=='R_global' and admitted(r) for r in required),
        'required_fallback_global_choices':sum(choice(r)=='R_global' and fallback(r) for r in required),
        'required_unsafe_continues':sum(choice(r)=='continue' for r in required),
        'required_rejected':sum(choice(r)=='reject' for r in required),
        'required_actual_safe_successes':sum(outcome(r)=='A' for r in required),
        'safe_control_states':len(controls),'safe_controls_kept_local':sum(choice(r)=='continue' for r in controls),
        'safe_controls_upgraded':sum(choice(r)=='R_global' for r in controls),
        'safe_controls_rejected':sum(choice(r)=='reject' for r in controls),
        'safe_controls_actual_safe_successes':sum(outcome(r)=='A' for r in controls),
        'timely':sum(r['mid_selection']['forecast_ready'] for r in rows),
        'stale':sum(not r['mid_selection']['continuation_matches_forecast'] for r in rows),
        'fallbacks':sum(fallback(r) for r in rows),
        'model_admissions':sum(admitted(r) for r in rows),
        'false_candidate_admissions':sum(p['accepted'] and r['reference_labels'][p['key']]!='A'
            for r in rows for p in r['mid_selection']['decision'].get('records',[])),
        'candidate_class_correct':sum('ABCD'[max(range(4),key=lambda i:p['conditional_probabilities'][i])]==r['reference_labels'][p['key']]
            for r in rows for p in r['mid_selection']['decision'].get('records',[])),
        'candidate_count':2*len(rows),
        'safe_model_state_coverage':sum(any(p['accepted'] and r['reference_labels'][p['key']]=='A'
            for p in r['mid_selection']['decision'].get('records',[])) for r in rows),
        'actual_unsafe_model_choices':sum(admitted(r) and outcome(r)!='A' for r in rows),
        'actions':dict(Counter(choice(r) for r in rows))}
    cases=[{'seed':r['seed'],'scenario':r['scenario'],'reference_labels':r['reference_labels'],
        'choice':choice(r),'fallback':fallback(r),'model_admitted':admitted(r),'actual_label':outcome(r),
        'total_seconds':r['total_post_fault_seconds'],'original_total_seconds':next(s['total_post_fault_seconds'] for s in sources if s['seed']==r['seed'] and s['scenario']==r['scenario']),
        'global_reference_seconds':r['reference_global_total_seconds']} for r in rows]
    finish(output,protocol,{'episodes':len(rows),'metrics':metrics,'actual':trial_summary(rows),'cases':cases})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--mode',choices=('mid','search','all'),default='all')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--gpu',default='2')
    parser.add_argument('--gpu-python',default='/opt/miniconda3/bin/python')
    parser.add_argument('--model-path',default='outputs/models/Qwen3-VL-8B-Instruct')
    args=parser.parse_args();output=Path(args.output_dir).resolve()
    if args.mode in ('mid','all'):mid_diagnostic(output/'mid13',args)
    if args.mode in ('search','all'):search_diagnostic(output/'search8',args.workers)


if __name__=='__main__':main()
