"""Frozen offline readouts of the five previously late forecasts; no physics."""
from pathlib import Path
import json
import os
import shlex
import subprocess
import time

from auditembodied.duration_aligned_loop import read_response
from auditembodied.vlm_duration_data import plan

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'vlm-two-window-rgbd-20261109'
REMOTE='/data2/zh/AuditEmbodied-two-window-20261109'
rows=[json.loads(s) for s in (OLD/'closed-loop-bridge/trials.jsonl').open()]
truth={(r['seed'],r['scenario'],r['method']):r for r in
       map(json.loads,(OLD/'counterfactual/cases.jsonl').open())}
jobs=[r for r in rows if any(not w['forecast_ready'] for w in r.get('mid_selection',{}).get('windows',[]))]
assert sum(sum(not w['forecast_ready'] for w in r['mid_selection']['windows']) for r in jobs)==5

def translate(x):
    if isinstance(x,str):return x.replace('/home/lc/AuditEmbodied',REMOTE)
    if isinstance(x,dict):return {k:translate(v) for k,v in x.items()}
    if isinstance(x,list):return [translate(v) for v in x]
    return x

command=['/data2/zh/.venvs/ae-two-window-20261109/bin/python','-m','auditembodied.vlm_duration_policy',
 '--working-point','outputs/vlm-two-window-rgbd-20261109/remote-working-point.json',
 '--model-path','/data2/zh/models/Qwen3-VL-8B-Instruct',
 '--post-wait-adapter','outputs/vlm-duration-family-stream-20261109/set512/adapter',
 '--post-wait-mode','family_stream','--mid-adapter','outputs/vlm-two-window-rgbd-20261109/full1024/adapter']
launch='cd '+shlex.quote(REMOTE)+' && env '+shlex.join([
 'CUDA_VISIBLE_DEVICES='+os.environ.get('AE_REPLAY_GPU','3'),
 'PYTHONPATH='+os.environ.get('AE_REMOTE_PYTHONPATH',REMOTE),
 'OMP_NUM_THREADS=2','OPENBLAS_NUM_THREADS=1','TOKENIZERS_PARALLELISM=false',*command])
records=[]
with (ROOT/'timeout-worker.log').open('w') as log:
    worker=subprocess.Popen(['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=10','zh-server',launch],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    def message(request):
        tick=time.perf_counter();worker.stdin.write(json.dumps(translate(request))+'\n');worker.stdin.flush()
        result,_=read_response(worker);result['offline_client_seconds']=time.perf_counter()-tick
        return result
    try:
        setup,_=read_response(worker);assert setup.get('ready'),setup
        (ROOT/'timeout-setup.json').write_text(json.dumps(setup,indent=2)+'\n')
        for r in jobs:
            mid=r['mid_selection'];h0=r['acquisition']['h0']
            p=plan({'candidate':mid['repair_candidate']});p.update(post_prefix_wait_steps=0,final_hold_steps=20)
            example={'two_window':True,'before_image':h0['image'],'before_robot':h0['robot'],
                'entry_image':r['acquisition']['mid_entry']['image'],'entry':mid['entry_input'],'plan':p}
            # Seed the identical prerequisite cache before prefill, as online scope did.
            cached=message({'op':'cache_start','image':h0['image'],'robot':h0['robot'],'plan':p})
            message({'op':'mid_prefill','example':example})
            for window in mid['windows']:
                example['query']=window['query_input']
                answer=message({'op':'mid_tail','example':example,'attempt_id':window['decision_step']})
                if window['forecast_ready']:
                    old=window['decision']['records']
                    error=max(abs(a-b) for x,y in zip(old,answer['records'])
                        for a,b in zip(x['class_probabilities'],y['class_probabilities']))
                    assert error<=1e-6,error
                else:
                    labels=truth[(r['seed'],r['scenario'],r['method'])]['labels']
                    branch={'continue':'LOCAL60','defer':'GLOBAL40','R_global':'GLOBAL'+str(window['decision_step'])}.get(answer['action'])
                    result={'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],
                        'decision_step':window['decision_step'],'start_score':cached['start_valid_score'],
                        'original_action':window['selected_branch'],'original_outcome':truth[(r['seed'],r['scenario'],r['method'])]['actual_label'],
                        'physical_labels':labels,'offline_action':answer['action'],'offline_selected_label':labels.get(branch),
                        'answer':answer,'diagnostic_only':True,'used_for_training':False}
                    records.append(result);print(json.dumps({k:result[k] for k in ('seed','scenario','method','decision_step','offline_action','offline_selected_label')}),flush=True)
            message({'op':'mid_release'})
    finally:
        if worker.poll() is None:
            try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
            except BrokenPipeError:pass
            worker.wait(timeout=30)
        (ROOT/'timeout-worker-exit.json').write_text(json.dumps({'exit_code':worker.poll()})+'\n')
assert len(records)==5
(ROOT/'timeout-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
(ROOT/'timeout-report.json').write_text(json.dumps({'frozen_recomputations':5,'not_a_new_test':True,
 'used_for_training':False,'note':'Offline timing does not change or replace recorded online waits. Branch outcomes already physically executed.',
 'records':records},indent=2)+'\n')
