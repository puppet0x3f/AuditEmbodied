import copy,json
from pathlib import Path
from auditembodied.duration_continuation import boundary_prefix
from auditembodied.vlm_duration_data import plan
root=Path(__file__).resolve().parent;old=Path('outputs/vlm-admission-consistent-20261110')
rows=[json.loads(x) for x in (old/'closed-loop96/trials.jsonl').open()]
sources=[r for r in rows if r['method']=='new_consistent' and r.get('mid_selection',{}).get('windows')]
assert len(sources)==14
local={(r['seed'],r['scenario']):r for r in sources if sum(f['stage']=='local_relevel' for f in r['trace'])==60}
wanted={(r['seed'],r['scenario']) for r in sources}
for line in (old/'counterfactual/trials.jsonl').open():
 r=json.loads(line);key=(r['seed'],r['scenario'])
 if r.get('policy_source',{}).get('method')=='new_consistent' and key in wanted and r['method']=='continue60':local[key]=r
assert len(local)==14
examples=[];specs=[]
for source in sources:
 w=source['mid_selection']['windows'][0];full=local[(source['seed'],source['scenario'])]
 entry=source['mid_selection']['entry_input'];p=plan({'candidate':source.get('source_candidate',source['candidate'])});p.update(post_prefix_wait_steps=0,final_hold_steps=20)
 base={'two_window':True,'seed':source['seed'],'scenario':source['scenario'],'before_image':source['acquisition']['h0']['image'],
  'entry_image':source['acquisition']['mid_entry']['image'],'before_robot':source['acquisition']['h0']['robot'],'entry':entry,'plan':p}
 example20={**base,'query':w['query_input']}
 prefix=boundary_prefix(full,'relevel30');recent=[f for f in prefix if f['phase']=='recovery'][-5:];last=recent[-1]
 q=copy.deepcopy(w['boundary_input']);q.update(previous_robot={**recent[0]['robot'],'reference_positions':q['reference_positions']},
  delta_seconds=last['time']-recent[0]['time'],relevel_step=30,attempt_id=40,sim_time=last['time'],
  seconds_after_h0=last['time']-source['prefix_state']['time'],sensor_history=[{'time':f['time'],**f['robot']} for f in recent])
 example30={**base,'query':q}
 d=source.get('duration_selection',{});admitted=bool(d.get('forecast_ready') and not d.get('baseline_fallback') and d.get('decision',{}).get('model_admitted'))
 timer=next(t for t in source['async_timing'] if t['kind']=='mid_tail_30')
 examples.append({'seed':source['seed'],'scenario':source['scenario'],'example20':example20,'example30':example30,
  'initial_admitted':admitted,'original_ready':w['forecast_ready'] and w['continuation_matches_forecast'],
  'original_latency':timer['pipeline_wall_seconds'],'original_decision':w['decision']})
 specs.append({'source':source,'local':full})
(root/'diagnostic-inputs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in examples))
(root/'diagnostic-sources.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in specs))
print('14 source states; 28 causal observation inputs; no labels supplied to model')
