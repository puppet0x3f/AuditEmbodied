from pathlib import Path
from collections import Counter
from itertools import combinations
import ast,csv,hashlib,json,statistics

ROOT=Path(__file__).resolve().parent
METHODS=['fixed60_fast','strict_fast','persistent','selective']
def read(p):return [json.loads(s) for s in p.open()]
def label(r):return ('B' if r['protection_violation'] else 'A') if r['success'] else ('D' if r['protection_violation'] else 'C')
def mean(x):
    a=list(x);return statistics.mean(a) if a else None
def costs(a):
    return {k:mean(r[v] for r in a) for k,v in [('total_s','total_post_fault_seconds'),('wait_s','inference_wait_seconds'),
        ('skill_s','skill_seconds'),('pipeline_s','total_pipeline_wall_seconds'),('wall_s','total_post_fault_wall_seconds')]}
def dist(x):
    a=list(x);return {'n':len(a),'median':statistics.median(a),'max':max(a)} if a else None

report=json.loads((ROOT/'closed-loop96/report.json').read_text())
rows=read(ROOT/'closed-loop96/trials.jsonl')
counter=json.loads((ROOT/'counterfactual/report.json').read_text())
cases=read(ROOT/'counterfactual/cases.jsonl')
effect=counter['matched_second_window_effect']
if 'states_with_two_actual_queries' in effect:
    effect['states_with_two_decision_windows']=effect.pop('states_with_two_actual_queries')
effect['scope']='Same physical prefix and recorded waits: prior admitted LOCAL60 at step30 versus the actual decision at step40. Two decision boundaries do not imply two requests; selective may skip query20. Native four-method costs are reported separately.'
(ROOT/'counterfactual/report.json').write_text(json.dumps(counter,indent=2)+'\n')
by={m:{(r['seed'],r['scenario']):r for r in rows if r['method']==m} for m in METHODS}
keys=set.intersection(*(set(v) for v in by.values()))
assert len(rows)==96 and len(keys)==24 and all(len(v)==24 for v in by.values())
assert {r['seed'] for r in rows}==set(range(55200,55212))
for name in ['frozen.json','pretest-source-hashes.json']:
    for p,h in json.loads((ROOT/name).read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
before=ast.parse((ROOT/'source-before/l4_lift.py').read_text());now=ast.parse((ROOT.parents[1]/'auditembodied/l4_lift.py').read_text())
for name in ['LiftConfig','TwoArmLiftEnv']:
    pick=lambda t:next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==name)
    assert ast.dump(pick(before))==ast.dump(pick(now))
assert all(not r.get('error') and not r['blocked_calls'] and all(r[k]==0 for k in
    ['state_edit_count_after_initialization','contact_anchor_count','external_wrench_at_end']) for r in rows)
assert all(abs(r['total_post_fault_seconds']-r['inference_wait_seconds']-r['skill_seconds']-r['hold_seconds'])<1e-7 for r in rows)
assert all(not r['diagnostic_errors'] for r in read(ROOT/'counterfactual/trials.jsonl'))
result={'episodes':96,'states':24,'methods':report['methods'],'costs':{},'labels':{},'timing':{},'runtime':{},
    'common_safe':{},'initial_duration_outcomes':{},'counterfactual':counter,'upgrade_attribution':{},
    'diagnostic28':json.loads((ROOT/'diagnostic28/report.json').read_text()),
    'deployment':json.loads((ROOT/'protocol.json').read_text())['deployment'],
    'checks':{'model_threshold_gains_force_unchanged':True,'test_sources_unchanged':True,'prefix_pairs_exact':True,
        'all_waits_charged':True,'no_unadmitted_mid_global':True,'all_counterfactual_prefixes_exact':True,
        'training':False,'threshold_adjustment':False,'performance_gate':False}}
for m in METHODS:
    a=list(by[m].values());w=[x for r in a for x in r.get('mid_selection',{}).get('windows',[])]
    tail=[x for r in a for x in r.get('async_timing',[]) if x['kind'].startswith('mid_tail')]
    assert all(x['global_admitted'] for x in w if x['selected_branch']=='R_global'),m
    for r in a:
        count=sum(f['stage']=='local_relevel' for f in r['trace'])
        if r['candidate']=='R_global' and count in (30,40):
            windows=r.get('mid_selection',{}).get('windows',[])
            assert any(x['selected_branch']=='R_global' and x['global_admitted'] or
                count==40 and x['selected_branch']=='defer' and x['decision'].get('model_admitted') for x in windows)
    result['costs'][m]=costs(a);result['labels'][m]=dict(Counter(label(r) for r in a))
    result['initial_duration_outcomes'][m]={str(n):dict(Counter(label(r) for r in a if r.get('local_steps',{}).get('local_relevel_steps')==n)) for n in (30,40,60)}
    result['timing'][m]={'windows':len(w),'available_and_matching':sum(x['forecast_ready'] and x['continuation_matches_forecast'] for x in w),
        'not_ready':sum(not x['forecast_ready'] for x in w),'stale':sum(not x['continuation_matches_forecast'] for x in w),
        'tail_client_s':dist(t['pipeline_wall_seconds'] for t in tail),'tail_gpu_s':dist(t['worker_seconds'] for t in tail),
        'mid_render_s':dist(r['acquisition']['mid_entry']['seconds'] for r in a if 'mid_entry' in r['acquisition']),
        'snapshot_copy_s':dist(r['acquisition']['mid_entry']['snapshot_copy_seconds'] for r in a if 'mid_entry' in r['acquisition'])}
    result['timing'][m]['by_request']={kind:{
        'client_s':dist(t['pipeline_wall_seconds'] for r in a for t in r.get('async_timing',[]) if t['kind']==kind),
        'gpu_s':dist(t['worker_seconds'] for r in a for t in r.get('async_timing',[]) if t['kind']==kind)}
        for kind in ['mid_prefill','mid_tail_30','mid_tail_40']}
    result['runtime'][m]={'carried_local_windows':sum(x.get('carried_local_admission',False) for x in w),
        'carried_local_episodes':sum(any(x.get('carried_local_admission') for x in r.get('mid_selection',{}).get('windows',[])) for r in a),
        'skipped_first_queries':sum(q.get('kind')=='mid_skipped' for r in a for q in r.get('query_schedule',[])),
        'second_queries':sum(t['kind']=='mid_tail_40' for r in a for t in r.get('async_timing',[])),
        'late_first_consumed_at40':sum(x['decision_step']==40 and x['forecast_ready'] and x['decision'].get('forecast_query_step')==20 for x in w),
        'unadmitted_globals':sum(x['selected_branch']=='R_global' and not x['global_admitted'] for x in w),
        'expired_positive_global30_records':sum(p.get('activation_step')==30 and p['accepted'] and not p.get('feasible_now',True) for x in w for p in x['decision'].get('records',[])),
        'local_control_wait_steps':sum(r.get('controller_compute_wait_steps',0) for r in a)}
    c=[x for x in cases if x['method']==m]
    def source(x):
        if not x['windows']:return 'fixed'
        z=x['windows'][-1]
        if z.get('carried_local_admission'):return 'prior_local_plan'
        if z.get('carried_global_admission'):return 'prior_global40_plan'
        if not z['forecast_ready'] or not z['continuation_matches_forecast']:return 'deadline_or_stale_exit'
        return 'late_first' if z['decision_step']==40 and z['decision'].get('forecast_query_step')==20 else 'second_query' if z['decision_step']==40 else 'first_query'
    needed=[x for x in c if x['labels']['LOCAL60']!='A' and any(x['labels'][k]=='A' for k in ['GLOBAL30','GLOBAL40'])]
    result['upgrade_attribution'][m]={'necessary_safe_upgrades_by_source':dict(Counter(source(x) for x in needed if x['actual_label']=='A' and x['actual_branch'].startswith('GLOBAL'))),
        'necessary_misses_by_source':dict(Counter(source(x) for x in needed if x['actual_label']!='A')),
        'unnecessary_upgrades_by_source':dict(Counter(source(x) for x in c if x['actual_branch'].startswith('GLOBAL') and x['labels']['LOCAL60']=='A')),
        'recoverable_exits':sum(x['actual_branch']=='reject' and 'A' in x['labels'].values() for x in c),
        'bad_prior_local_continuations':sum(x['actual_branch']=='LOCAL60' and x['actual_label']!='A' and source(x)=='prior_local_plan' for x in c)}
for subset in [METHODS,*[list(x) for x in combinations(METHODS,2)]]:
    common=sorted(k for k in keys if all(label(by[m][k])=='A' for m in subset))
    result['common_safe']['/'.join(subset)]={'states':len(common),'costs':{m:costs([by[m][k] for k in common]) for m in subset}}
result['revision_sources']=[{'seed':r['seed'],'scenario':r['scenario'],
    'forecast_ready':r['windows'][-1]['forecast_ready'],
    'query_step':r['windows'][-1]['decision'].get('forecast_query_step') if r['windows'][-1]['forecast_ready'] else None,
    'first_action':r['windows'][0]['selected_branch'],'final_action':r['windows'][-1]['selected_branch'],'actual_label':r['actual_label'],'labels':r['labels']}
    for r in cases if r['method']=='selective' and len(r['windows'])==2]
cf={(r['seed'],r['scenario'],r['method']):r for r in cases}
fields=['seed','scenario','method','label','N','actions','query_sources','carry_local','control_s','wait_s','pipeline_s','LOCAL60','GLOBAL30','GLOBAL40']
with (ROOT/'cases.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in rows:
        w=r.get('mid_selection',{}).get('windows',[])
        writer.writerow({'seed':r['seed'],'scenario':r['scenario'],'method':r['method'],'label':label(r),'N':r.get('local_steps',{}).get('local_relevel_steps'),
            'actions':';'.join(str(x['decision_step'])+':'+x['selected_branch'] for x in w),'query_sources':';'.join(str(x['decision'].get('forecast_query_step')) for x in w),
            'carry_local':sum(x.get('carried_local_admission',False) for x in w),'control_s':r['total_post_fault_seconds'],'wait_s':r['inference_wait_seconds'],
            'pipeline_s':r['total_pipeline_wall_seconds'],**cf.get((r['seed'],r['scenario'],r['method']),{}).get('labels',{})})
(ROOT/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['methods','costs','runtime','upgrade_attribution','common_safe']},ensure_ascii=False,indent=2))
