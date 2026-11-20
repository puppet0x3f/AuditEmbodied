from pathlib import Path
from collections import Counter
import ast,csv,json,statistics
import numpy as np
from auditembodied.duration_pilot import file_hash
root=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (root/'closed-loop-bridge/trials.jsonl').open()]
report=json.loads((root/'closed-loop-bridge/report.json').read_text())
counter=json.loads((root/'counterfactual/report.json').read_text())
cases=[json.loads(s) for s in (root/'counterfactual/cases.jsonl').open()]
methods=['fixed60','mid_single','mid_two']
label=lambda r:('B' if r['protection_violation'] else 'A') if r['success'] else ('D' if r['protection_violation'] else 'C')
by={m:{(r['seed'],r['scenario']):r for r in rows if r['method']==m} for m in methods}
keys=set.intersection(*(set(by[m]) for m in methods))
common=[k for k in keys if all(label(by[m][k])=='A' for m in methods)]
metrics={'methods':report['methods'],'counterfactual':counter,'common_safe_states':len(common),'common_safe_costs':{},'timing':{},'reasons':{}}
for m in methods:
 a=list(by[m].values());windows=[w for r in a for w in r.get('mid_selection',{}).get('windows',[])]
 tail=[t for r in a for t in r.get('async_timing',[]) if t['kind'].startswith('mid_tail')]
 metrics['timing'][m]={'windows':len(windows),'ready_at_boundary':sum(w['forecast_ready'] for w in windows),
  'stale':sum(not w['continuation_matches_forecast'] for w in windows),'baseline_fallbacks':sum(w['baseline_fallback'] for w in windows),
  'client_tail_s':None if not tail else {'median':statistics.median(t['pipeline_wall_seconds'] for t in tail),'max':max(t['pipeline_wall_seconds'] for t in tail)},
  'gpu_tail_s':None if not tail else {'median':statistics.median(t['worker_seconds'] for t in tail),'max':max(t['worker_seconds'] for t in tail)}}
 metrics['reasons'][m]={'abstention':dict(Counter(r.get('abstention_reason') for r in a if r.get('abstained'))),
  'actual_labels':dict(Counter(label(r) for r in a)),
  'model_and_fallback_actions':dict(Counter(str(w['decision_step'])+':'+w['selected_branch']+(' [fallback]' if w['baseline_fallback'] else '') for w in windows))}
 if common:metrics['common_safe_costs'][m]={'mean_control_s':statistics.mean(by[m][k]['total_post_fault_seconds'] for k in common),
   'mean_wall_s':statistics.mean(by[m][k]['total_post_fault_wall_seconds'] for k in common)}
metrics['upgrade_attribution']={}
def upgrade_source(r):
 w=r['windows'][-1]
 if not w['forecast_ready']:return 'deadline_fallback'
 if not w['continuation_matches_forecast']:return 'stale_fallback'
 if w['baseline_fallback']:return 'uncertain_fallback'
 return 'model' if w['decision'].get('model_admitted') else 'other'
for m in ('mid_single','mid_two'):
 a=[r for r in cases if r['method']==m]
 needed=[r for r in a if r['labels']['LOCAL60']!='A' and any(r['labels'][k]=='A' for k in ('GLOBAL30','GLOBAL40'))]
 upgrades=[r for r in a if r['actual_branch'].startswith('GLOBAL')]
 metrics['upgrade_attribution'][m]={
  'needed_states':len(needed),
  'safe_necessary_upgrades_by_source':dict(Counter(upgrade_source(r) for r in upgrades if r in needed and r['actual_label']=='A')),
  'unnecessary_upgrades_by_source':dict(Counter(upgrade_source(r) for r in upgrades if r['labels']['LOCAL60']=='A'))}
metrics['matched_second_window_attribution']=[]
for x in counter['matched_second_window_effect']['cases']:
 r=next(r for r in cases if r['method']=='mid_two' and r['seed']==x['seed'] and r['scenario']==x['scenario'])
 metrics['matched_second_window_attribution'].append({**x,
  'revision_source':upgrade_source(r) if r['actual_branch'].startswith('GLOBAL') else 'model_continue'})
checks=json.loads((root/'frozen.json').read_text());assert all(file_hash(p)==h for p,h in checks.items())
old=ast.parse((root/'source-before/l4_lift.py').read_text());new=ast.parse(Path('auditembodied/l4_lift.py').read_text())
for name in ['LiftConfig','TwoArmLiftEnv']:
 assert ast.dump(next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name==name))==ast.dump(next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name==name))
freeze=json.loads((root/'deployment-freeze.json').read_text());assert all(file_hash(p)==h for p,h in freeze.items())
metrics['frozen_original_models_threshold_gains_force_and_deployment_sources_unchanged']=True
assert len(rows)==72 and all(len(by[m])==24 for m in methods)
assert all(not r.get('error') and not r['blocked_calls'] and all(r[k]==0 for k in ['state_edit_count_after_initialization','contact_anchor_count','external_wrench_at_end']) for r in rows)
assert all(abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-7 for r in rows)
(root/'RESULTS.json').write_text(json.dumps(metrics,indent=2)+'\n')
fields=['seed','scenario','method','label','N','mid_windows','actions','abstention_reason','table_attempts','control_s','wait_s','wall_s','pipeline_wall_s']
with (root/'cases.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for r in rows:
  w=r.get('mid_selection',{}).get('windows',[])
  writer.writerow({'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],'label':label(r),
   'N':r.get('local_steps',{}).get('local_relevel_steps'),'mid_windows':len(w),'actions':';'.join(str(x['decision_step'])+':'+x['selected_branch'] for x in w),
   'abstention_reason':r.get('abstention_reason'),'table_attempts':len(r.get('table_assist',{}).get('attempts',[])),
   'control_s':r['total_post_fault_seconds'],'wait_s':r['inference_wait_seconds'],'wall_s':r['total_post_fault_wall_seconds'],'pipeline_wall_s':r['total_pipeline_wall_seconds']})
print(json.dumps({'methods':report['methods'],'counterfactual':counter['methods'],'matched':counter['matched_second_window_effect'],'common_safe_costs':metrics['common_safe_costs'],'timing':metrics['timing']},indent=2))
