import json,hashlib
from pathlib import Path
root=Path('/home/lc/AuditEmbodied/outputs/vlm-duration-family-stream-20261109')
source=Path('/home/lc/AuditEmbodied/outputs/vlm-duration-batch-intervention-20261109')
p=json.loads((source/'experiment-plan.json').read_text());physical={}
read=lambda f:[json.loads(x) for x in Path(f).read_text().splitlines()]
raw=[]
for w in (0,66):raw+=read(Path(p['source_collection'])/f'trials-wait{w}.jsonl')
raw+=read(source/'collection/trials.jsonl')
for r in raw:physical[r['seed'],r['scenario'],r['post_prefix_wait_steps'],r['local_steps']['local_relevel_steps']]=r
out=root/'dataset';out.mkdir(exist_ok=True);counts={}
for split in ('train','development'):
 rows=read(source/f'dataset/{split}.jsonl');result=[]
 for row in rows:
  raw=physical[row['seed'],row['scenario'],row['wait_steps'],row['plan']['local_relevel_steps']]
  t=raw['duration_selection']['boundary_state']['time'];trace=[f for f in raw['trace'] if f['time']<=t+1e-8]
  end=next(i for i in range(len(trace)-1,-1,-1) if trace[i]['stage'] in ('local_arc_close','local_closed_settle'))
  stage=trace[end]['stage'];begin=end
  while begin>0 and trace[begin-1]['stage']==stage:begin-=1
  assert end-begin+1>=22
  i=end-10;f=trace[i];previous=trace[max(begin,i-4)]
  robot=lambda f:{'positions':f['robot']['positions'],'fingers':f['robot']['fingers'],'reference_positions':row['robot']['reference_positions']}
  keeper=1 if row['plan']['repair_arm']=='left' else 0
  assert raw['post_fault']['holds'][keeper]
  r={**row,'entry_seconds_after_h0':raw['stage_capture']['h1']['state']['time']-raw['prefix_state']['time'],
   'lookahead_robot':robot(f),'lookahead_previous_robot':robot(previous),
   'lookahead_delta_seconds':f['time']-previous['time'],'lookahead_stage':stage,
   'lookahead_seconds_after_h0':f['time']-raw['prefix_state']['time'],'remaining_close_steps':10,
   'prefix_observation_wait_steps':raw.get('prefix_observation_wait_steps',0)}
  assert abs(t-f['time']-.5)<1e-7
  result.append(r)
 (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result));counts[split]=len(result)
p.update(data_dir=str(out),source_round=str(source),temporal_mode='family_stream',
 initial_outcome_adapter=str(Path(p['previous_round'])/'full512/adapter'),
 prefix_images='h0 four RGB, repair/keeper wrist order; no H1 or H2 RGB',
 model_context='Measured h1 entry robot + robot state10 closure steps before boundary; no labels or contact truth in input',
 actual_label_timing='Original C0, executed C1 and planned W0/66 retained; no relabeling for new execution timing',
 lookahead_steps=10,pace_recovery=True,train_examples=counts['train'],development_examples=counts['development'])
(root/'experiment-plan.json').write_text(json.dumps(p,indent=2)+'\n')
(root/'dataset/report.json').write_text(json.dumps({'counts':counts,'reconstructed_from_actual_past_frames':True,
 'all_queries_half_second_before_boundary':True,'source_dataset_hashes':{s:hashlib.sha256((source/f'dataset/{s}.jsonl').read_bytes()).hexdigest() for s in counts}},indent=2)+'\n')
print(counts)
