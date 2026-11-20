"""Independent fixed60 / one-window / two-window closed-loop comparison."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import shlex
import time

from .duration_aligned_loop import read_response
from .duration_closed_loop import trial_summary
from .duration_family_loop import family_trial
from .duration_pilot import file_hash,STATE_KEYS,prefix_frames
from .mid_recovery_data import read


def summarize(rows):
    d=trial_summary(rows);windows=[w for r in rows for w in r.get('mid_selection',{}).get('windows',[])]
    d.update(mid_windows=len(windows),window_actions=dict(Counter(str(w['decision_step'])+':'+w['selected_branch'] for w in windows)),
        abstentions=sum(bool(r.get('abstained')) or r['candidate']=='abstain' for r in rows),
        timely=sum(w['forecast_ready'] and w['continuation_matches_forecast'] for w in windows),
        fallback_windows=sum(w['baseline_fallback'] for w in windows),
        uncertainty_exits=sum(w['selected_branch']=='reject' and w['decision'].get('uncertainty_exit',False) for w in windows),
        deadline_exits=sum(w['selected_branch']=='reject' and (not w['forecast_ready'] or not w['continuation_matches_forecast']) for w in windows),
        unadmitted_mid_global_executions=sum(w['selected_branch']=='R_global' and
            not w.get('global_admitted',not w['baseline_fallback'] and w['decision'].get('model_admitted',False)) for w in windows),
        explicit_model_admissions=sum(w['decision'].get('model_admitted',False) and not w['baseline_fallback'] for w in windows),
        mean_wait_seconds=sum(r['inference_wait_seconds'] for r in rows)/len(rows),
        mean_post_fault_wall_seconds=sum(r['total_post_fault_wall_seconds'] for r in rows)/len(rows),
        mean_pipeline_wall_seconds=sum(r['total_pipeline_wall_seconds'] for r in rows)/len(rows),
        controller_wait_steps=sum(r.get('controller_compute_wait_steps',0) for r in rows),
        duration_counts=dict(Counter(str(r.get('local_steps',{}).get('local_relevel_steps')) for r in rows)),
        search_exhaustions=sum('skill_failure' in r for r in rows),
        table_regrasp_entered=sum(bool(r.get('table_assist',{}).get('attempts')) for r in rows),
        table_safe_successes=sum('table_assist' in r and r['success'] and not r['protection_violation'] for r in rows))
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--mid-adapter',required=True)
    p.add_argument('--control-mid-adapter',help='Four-method comparison: current legacy/current consistent/new consistent/fixed60')
    p.add_argument('--async-runtime-comparison',action='store_true')
    p.add_argument('--engineering-prefix-seed',type=int,help='Run one training seed first, saving its eight episodes separately while keeping the worker resident')
    p.add_argument('--admission-policy',choices=('legacy','consistent'),default='consistent')
    p.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    p.add_argument('--adapter',default='outputs/vlm-duration-family-stream-20261109/set512/adapter')
    p.add_argument('--model-path',default='outputs/models/Qwen3-VL-8B-Instruct');p.add_argument('--gpu',default='3')
    p.add_argument('--model-max-memory')
    p.add_argument('--gpu-python',default='/opt/miniconda3/bin/python');p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--warm-data',required=True);p.add_argument('--table-template',required=True);p.add_argument('--engineering',action='store_true')
    p.add_argument('--ssh-host');p.add_argument('--remote-root');p.add_argument('--remote-working-point')
    p.add_argument('--remote-pythonpath')
    args=p.parse_args();out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);(out/'observations').mkdir(exist_ok=True)
    if args.async_runtime_comparison and (args.control_mid_adapter or args.admission_policy!='consistent'):raise ValueError('async comparison uses one frozen model and consistent admission')
    if (out/'trials.jsonl').exists():raise ValueError('closed-loop output exists')
    point=json.loads(Path(args.working_point).read_text());files=[Path(args.working_point).resolve(),Path(args.table_template).resolve()]
    for folder in (point['start_adapter'],point['outcome_adapter'],args.adapter,args.mid_adapter,*([args.control_mid_adapter] if args.control_mid_adapter else [])):
        files.extend(Path(folder).resolve()/n for n in ('adapter_config.json','adapter_model.safetensors'))
    protocol={'seeds':args.seeds,'engineering_only':args.engineering,'threshold':point['threshold'],
        'frozen':{str(x):file_hash(x) for x in files},'methods':['fixed60','current_legacy','current_consistent','new_consistent'] if args.control_mid_adapter else ['fixed60','mid_single','mid_two'],
        'one_checkpoint_for_both_window_policies':not bool(args.control_mid_adapter),'all_waits_physically_charged':True,
        'admission_policy':args.admission_policy,'mid_adapter':args.mid_adapter,'control_mid_adapter':args.control_mid_adapter,
        'table_skill_shared_by_all_methods':True,'new_test_tuning':False,
        'ssh_gpu_host':args.ssh_host,'remote_root':args.remote_root,'remote_working_point':args.remote_working_point,'image_transfer_charged':bool(args.ssh_host),
        'comparison':'Actual measured initial policy latency in each episode; report initial durations and timings, do not attribute their differences to the second window.'}
    protocol['model_max_memory']=args.model_max_memory
    if args.async_runtime_comparison:
        protocol.update(methods=['fixed60_fast','strict_fast','persistent','selective'],async_entry_rgb=True,
            same_host_physics_and_inference=not bool(args.ssh_host),all_waits_physically_charged=True,
            engineering_prefix_seed=args.engineering_prefix_seed)
    if args.engineering_prefix_seed is not None and args.engineering_prefix_seed in args.seeds:raise ValueError('engineering seed overlaps test')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(out/'source').mkdir(exist_ok=True)
    for name in ('l4_lift.py','vlm_duration_policy.py','vlm_mid_recovery.py','duration_family_loop.py','two_window_loop.py','table_assist.py','table_rgbd.py','async_rgb.py'):
        path=Path(__file__).parent/name;(out/'source'/name).write_bytes(path.read_bytes())
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    rows=[]
    uploaded=set()
    def translate(value):
        if isinstance(value,str):return value.replace(str(Path.cwd()),args.remote_root)
        if isinstance(value,list):return [translate(v) for v in value]
        if isinstance(value,dict):return {k:translate(v) for k,v in value.items()}
        return value
    def remote_request(request):
        if not args.ssh_host:return request
        result=translate(request);paths=[]
        if 'image' in request:paths.append(request['image'])
        if 'example' in request:
            paths.extend(request['example'][k] for k in ('before_image','entry_image') if k in request['example'])
        payload={}
        for name in paths:
            if name not in uploaded:
                payload[translate(str(Path(name).resolve()))]=base64.b64encode(Path(name).read_bytes()).decode('ascii')
                uploaded.add(name)
        if payload:result['_image_payloads']=payload
        return result
    with (out/'worker.log').open('w') as log:
        command=[args.gpu_python,'-m','auditembodied.vlm_duration_policy','--working-point',args.remote_working_point or args.working_point,
            '--model-path',args.model_path,'--post-wait-adapter',args.adapter,'--post-wait-mode','family_stream','--mid-adapter',args.mid_adapter]
        if args.model_max_memory:command.extend(['--model-max-memory',args.model_max_memory])
        if args.control_mid_adapter:command.extend(['--mid-control-adapter',args.control_mid_adapter])
        if args.ssh_host:
            if not args.remote_root or not args.remote_working_point:raise ValueError('remote GPU requires workspace and working-point paths')
            launch='cd '+shlex.quote(args.remote_root)+' && env '+shlex.join(['CUDA_VISIBLE_DEVICES='+args.gpu,'OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1','TOKENIZERS_PARALLELISM=false',
                *(['AE_GPU_CPUS='+os.environ['AE_GPU_CPUS']] if os.environ.get('AE_GPU_CPUS') else []),
                *(['PYTHONPATH='+args.remote_pythonpath] if args.remote_pythonpath else []),*command])
            command=['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=10',args.ssh_host,launch]
        worker=subprocess.Popen(command,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
        def message(request):
            started=time.perf_counter();request=remote_request(request)
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();answer=read_response(worker)[0]
            answer['client_pipeline_seconds']=time.perf_counter()-started
            return answer
        try:
            setup,_=read_response(worker);assert setup.get('ready'),setup
            warm=read('outputs/vlm-duration-family-stream-20261109/dataset/train.jsonl')[0]
            for request in ({'op':'family_prefill','example':warm},
                {'op':'family_tail','example':warm,'attempt_id':0,'plans':[{**warm['plan'],'local_relevel_steps':n} for n in (30,40,60)]},
                {'op':'family_release'}):message(request)
            train=read(args.warm_data);key=(train[0]['seed'],train[0]['scenario'],train[0]['source_wait_steps'])
            family=[r for r in train if (r['seed'],r['scenario'],r['source_wait_steps'])==key];profile=[]
            for adapter_name in (['mid','mid_control'] if args.control_mid_adapter else ['mid']):
                for repeat in range(2):
                    prefix=message({'op':'mid_prefill','example':family[0],'adapter_name':adapter_name})
                    for row in family:
                        cached=message({'op':'mid_tail','example':row,'attempt_id':row['decision_step'],'admission_policy':args.admission_policy})
                        fresh=message({'op':'mid_fresh','example':row,'attempt_id':row['decision_step'],'admission_policy':args.admission_policy})
                        error=max(abs(a-b) for x,y in zip(cached['records'],fresh['records']) for a,b in zip(x['class_probabilities'],y['class_probabilities']))
                        assert error<=1e-6,error
                        profile.append({'adapter':adapter_name,'window':row['decision_step'],'prefill_s':prefix['model_pipeline_seconds'],
                            'tail_s':cached['model_pipeline_seconds'],'tail_client_s':cached['client_pipeline_seconds'],'prefill_client_s':prefix['client_pipeline_seconds'],'cached_fresh_error':error})
                    message({'op':'mid_release'})
            setup['training_observation_profiles']=profile;(out/'setup.json').write_text(json.dumps(setup,indent=2)+'\n')
            from .l0 import _import_robosuite
            _import_robosuite(True)
            engineering=out/'engineering-prefix';engineering.mkdir(exist_ok=True);(engineering/'observations').mkdir(exist_ok=True)
            engineering_rows=[]
            cases=([(True,args.engineering_prefix_seed,q) for q in ('left_loss','right_loss')] if args.engineering_prefix_seed is not None else [])+[(False,s,q) for s in args.seeds for q in ('left_loss','right_loss')]
            with (out/'trials.jsonl').open('w') as stream,(engineering/'trials.jsonl').open('w') as engineering_stream:
                for index,(is_engineering,seed,scenario) in enumerate(cases):
                    methods=[('fixed60','none',True,'mid','legacy'),('current_legacy','single',False,'mid_control','legacy'),
                        ('current_consistent','single',False,'mid_control','consistent'),('new_consistent','single',False,'mid','consistent')] if args.control_mid_adapter else [
                        ('fixed60','none',True,'mid',args.admission_policy),('mid_single','single',False,'mid',args.admission_policy),('mid_two','two',False,'mid',args.admission_policy)]
                    if args.async_runtime_comparison:
                        methods=[('fixed60_fast','none',True,'mid','consistent'),('strict_fast','single',False,'mid','consistent'),
                            ('persistent','single',False,'mid','consistent'),('selective','single',False,'mid','consistent')]
                    shift=(index//2+index%2)%len(methods)
                    order=methods[shift:]+methods[:shift];pair=[]
                    for label,strategy,baseline,mid_adapter,policy in order:
                        row=family_trial(seed,scenario,engineering if is_engineering else out,worker,baseline=baseline,mid_strategy=strategy,
                            table_assist_template=str(Path(args.table_template).resolve()),request_transform=remote_request if args.ssh_host else None,
                            mid_adapter_name=mid_adapter,mid_admission_policy=policy,method_name=label,
                            mid_runtime=label if label in ('persistent','selective') else 'strict',async_entry_rgb=args.async_runtime_comparison)
                        assert not row.get('error') and not row['blocked_calls'],row.get('error')
                        assert all(row[k]==0 for k in ('state_edit_count_after_initialization','contact_anchor_count','external_wrench_at_end'))
                        assert abs(row['total_post_fault_seconds']-row['inference_wait_seconds']-row['skill_seconds']-row['hold_seconds'])<1e-7
                        pair.append(row);row['engineering_only']=is_engineering or args.engineering
                        if is_engineering:
                            engineering_rows.append(row);engineering_stream.write(json.dumps(row)+'\n');engineering_stream.flush()
                        else:rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                        print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','inference_wait_seconds','total_post_fault_seconds')},ensure_ascii=False),flush=True)
                    def physical_prefix(row):
                        return [{**f,'robot':{k:v for k,v in f['robot'].items() if k!='wrench'}} for f in prefix_frames(row)]
                    assert all(all(pair[0][k]==r[k] for k in STATE_KEYS) and physical_prefix(pair[0])==physical_prefix(r) for r in pair[1:])
                    if is_engineering:
                        (engineering/'report.json').write_text(json.dumps({'engineering_only':True,'episodes':len(engineering_rows),
                            'methods':{m:summarize([r for r in engineering_rows if r['method']==m]) for m in protocol['methods'] if any(r['method']==m for r in engineering_rows)}},indent=2)+'\n')
        finally:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                except BrokenPipeError:pass
                worker.wait(timeout=30)
            (out/'worker-exit.json').write_text(json.dumps({'exit_code':worker.poll()})+'\n')
    assert all(file_hash(path)==digest for path,digest in protocol['frozen'].items())
    report={'episodes':len(rows),'engineering_only':args.engineering,'frozen_unchanged':True,
        'methods':{method:summarize([r for r in rows if r['method']==method]) for method in protocol['methods']}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
