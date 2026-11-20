import json,os,subprocess,time
from pathlib import Path
from auditembodied.duration_aligned_loop import read_response

root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
command=['/data2/zh/.venvs/ae-two-window-20261109/bin/python','-m','auditembodied.vlm_duration_policy',
 '--working-point','outputs/vlm-two-window-rgbd-20261109/remote-working-point.json',
 '--model-path','/data2/zh/models/Qwen3-VL-8B-Instruct',
 '--post-wait-adapter','outputs/vlm-duration-family-stream-20261109/set512/adapter','--post-wait-mode','family_stream',
 '--mid-adapter','outputs/vlm-admission-consistent-20261110/new1024/adapter']
def translate(x):
 if isinstance(x,str):return x.replace('/home/lc/AuditEmbodied','/data2/zh/AuditEmbodied-two-window-20261109')
 if isinstance(x,list):return [translate(y) for y in x]
 if isinstance(x,dict):return {k:translate(v) for k,v in x.items()}
 return x
with (root/'diagnostic-worker.log').open('w') as log:
 worker=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
 def request(r):
  started=time.perf_counter();worker.stdin.write(json.dumps(translate(r))+'\n');worker.stdin.flush();a,_=read_response(worker)
  a['client_seconds']=time.perf_counter()-started;return a
 try:
  setup,_=read_response(worker);assert setup['ready'];(root/'diagnostic-setup.json').write_text(json.dumps(setup,indent=2)+'\n')
  with (root/'diagnostic-predictions.jsonl').open('w') as out:
   for line in (root/'diagnostic-inputs.jsonl').open():
    r=json.loads(line);request({'op':'mid_prefill','example':r['example20']})
    answers={}
    for q in (20,30):answers[str(q)]=request({'op':'mid_tail','example':r['example'+str(q)],'attempt_id':q+10,'admission_policy':'consistent'})
    request({'op':'mid_release'})
    result={k:v for k,v in r.items() if not k.startswith('example')};result['answers']=answers
    if r['original_ready']:
     error=max(abs(x-y) for a,b in zip(r['original_decision']['records'],answers['20']['records']) for x,y in zip(a['class_probabilities'],b['class_probabilities']))
     assert error<1e-6,error
     result['original_score_error']=error
    out.write(json.dumps(result)+'\n');out.flush();print(json.dumps({'seed':r['seed'],'scenario':r['scenario'],'predictions':{k:v['action'] for k,v in answers.items()}}),flush=True)
 finally:
  if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
