"""Reconstruct prospective decisions, complete costs, and whole-seed estimates."""
from collections import Counter,defaultdict
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import campaign as c
from policy import WaitingPolicy,MODES

ROOT=Path(__file__).resolve().parent
WAIT=('inference_wait','decision_budget_wait','decision_refresh_wait')


def longest(flags):
    n=m=0
    for x in flags:n=n+1 if x else 0;m=max(n,m)
    return m


def mean(rows,key):return float(np.mean([r[key] for r in rows])) if rows else None


def metrics(rows):
    return {'episodes':len(rows),'C':sum(r['task_complete'] for r in rows),
        'Y':sum(r['audit']['protected_completion'] for r in rows),'labels':dict(Counter(r['audit']['label'] for r in rows)),
        'invalid_entries':sum(not r['audit']['V_entry'] for r in rows),
        'protection_violations':sum(r['protection_violation'] for r in rows),
        'mean_total_seconds':mean(rows,'total_post_fault_seconds'),
        'mean_wait_seconds':mean(rows,'inference_wait_seconds'),'mean_skill_seconds':mean(rows,'skill_seconds'),
        'mean_model_setup_wall_seconds':mean(rows,'model_setup_wall_seconds'),
        'mean_episode_wall_seconds':mean(rows,'wall_seconds'),
        'changed_initial_choice':sum(r['candidate']!=r['wait_initial_decision']['candidate'] for r in rows),
        'choices':dict(Counter(r['candidate'] for r in rows)),
        'max_initial_service_seconds':max((r['selector_wall_seconds'] for r in rows),default=0),
        'max_final_service_seconds':max((r['wait_decision']['final_service_wall_seconds'] for r in rows),default=0),
        'scored_observations_at_finish':sum(r['selection']['decision']['scored_observations'] for r in rows)}


def paired(rows,a,b,by='method'):
    groups=defaultdict(dict)
    for r in rows:groups[r['seed'],r['scenario']][r[by]]=r
    pairs=[(k,g[a],g[b]) for k,g in groups.items() if a in g and b in g]
    seeds=sorted({k[0] for k,_,_ in pairs});draw=np.random.default_rng(20261110).integers(0,len(seeds),(2000,len(seeds)))
    delta=np.array([np.mean([int(x['audit']['protected_completion'])-int(y['audit']['protected_completion']) for k,x,y in pairs if k[0]==s]) for s in seeds])
    result={'paired_states':len(pairs),'independent_seeds':len(seeds),'Y_difference':float(delta.mean()),
        'Y_difference_ci95':np.quantile(delta[draw].mean(1),[.025,.975]).tolist(),
        'Y_gains':sum(x['audit']['protected_completion'] and not y['audit']['protected_completion'] for _,x,y in pairs),
        'Y_losses':sum(y['audit']['protected_completion'] and not x['audit']['protected_completion'] for _,x,y in pairs)}
    for key,keep in [('all_attempt_cost',lambda x,y:True),('common_protected_cost',lambda x,y:x['audit']['protected_completion'] and y['audit']['protected_completion'])]:
        use=[(k,x,y) for k,x,y in pairs if keep(x,y)]
        result[key]={'states':len(use)}
        if not use:continue
        totals=np.asarray([[sum(x['total_post_fault_seconds'] for k,x,y in use if k[0]==s),sum(y['total_post_fault_seconds'] for k,x,y in use if k[0]==s)] for s in seeds])
        sample=totals[draw].sum(1);sample=sample[sample[:,1]>0]
        result[key].update(first_mean=float(np.mean([x['total_post_fault_seconds'] for _,x,y in use])),
            second_mean=float(np.mean([y['total_post_fault_seconds'] for _,x,y in use])),
            reduction_fraction=float(1-totals[:,0].sum()/totals[:,1].sum()),
            reduction_ci95=np.quantile(1-sample[:,0]/sample[:,1],[.025,.975]).tolist())
    return result


def main():
    p=json.loads((ROOT/'protocol.json').read_text());c.verify(p)
    freeze=json.loads((ROOT/'formal-freeze.json').read_text());assert c.sha(ROOT/'protocol.json')==freeze['protocol_sha256']
    rows=[json.loads(l) for l in (ROOT/'trials.jsonl').open()];assert len(rows)==480
    values=json.loads((ROOT/'direct-selected.json').read_text())['mlp']
    groups=defaultdict(list)
    for row in rows:
        groups[row['seed'],row['scenario'],row['window']].append(row)
        assert row['audit']==c.admission(row)
        assert not row.get('error') and row['nominal_ready'] and row['fault_class_valid']
        frames=[f for f in row['trace'] if f['phase'] in ('recovery','final_hold')]
        assert np.allclose(np.diff([row['prefix_state']['time']]+[f['time'] for f in frames]),.05,atol=1e-8)
        gap=longest([not(any(f['evaluator']['holds']) or f['evaluator']['table_contact']) for f in frames])
        lost=sum(f['phase']=='recovery' and f.get('skill') in ('R_left','R_right')
            and not f['evaluator']['holds'][1 if f['skill']=='R_left' else 0] for f in frames)
        assert lost==row['protection']['keeper_grasp_lost_steps']
        assert abs(gap/20-row['protection']['max_unsupported_seconds'])<1e-8
        assert bool(gap>2 or lost or row['protection']['deep_interarm_contact_steps'] or row['blocked_calls'])==row['protection_violation']
        nwait=sum(f['stage'] in WAIT for f in frames);assert nwait==row['inference_wait_steps']
        assert abs(nwait/20-row['inference_wait_seconds'])<1e-7 and row['audit']['cost_identity']
        assert all(f.get('skill')=='continue' for f in frames if f['stage'] in ('inference_wait','decision_budget_wait'))
        history=row['wait_decision']['history'];assert set(history[0]['observation'])=={'positions','fingers','reference_positions'}
        before_decision=[f for f in frames if f['stage'] in ('inference_wait','decision_budget_wait')]
        assert len(history)==len(before_decision)+1
        for h,f in zip(history[1:],before_decision):
            assert h['observation']['positions']==f['robot']['positions'] and h['observation']['fingers']==f['robot']['fingers']
            assert abs(h['elapsed_seconds']-f['time']+row['prefix_state']['time'])<1e-8
        assert history[-1]['elapsed_seconds']<=row['wait_decision']['decision_time']-row['prefix_state']['time']+1e-8
        policy=WaitingPolicy(values,row['method']);initial=policy.initial(history[0]['observation'])
        assert initial==row['wait_initial_decision'] and policy(history)==row['selection']['decision']
        assert row['model_prewarmed']
    assert len(groups)==160 and all({r['method'] for r in g}==set(MODES) for g in groups.values())
    fields=('initialization_state','preclose_state','initialization_end_state','prefix_state')
    prefix_equal=all(all(r[k]==g[0][k] for r in g for k in fields) for g in groups.values())
    histories_equal=sum(all(r['wait_decision']['history']==g[0]['wait_decision']['history'] for r in g) for g in groups.values())
    entries_equal=sum(all(r['post_wait_state']==g[0]['post_wait_state'] for r in g) for g in groups.values())
    faults=[r for r in rows if r['scenario']!='sham'];healthy=[r for r in rows if r['scenario']=='sham']
    take=lambda source,w,m:[r for r in source if r['window']==w and r['method']==m]
    result={'episodes':len(rows),'independent_seeds':20,
        'validity':{'matched_initial_and_fault_prefixes':prefix_equal,'matched_history_groups':histories_equal,
            'matched_entry_state_groups':entries_equal,'total_groups':len(groups),'all_frozen_decisions_reproduced':True,
            'complete_costs':len(rows),'forbidden_calls':sum(len(r['blocked_calls']) for r in rows),
            'state_edits':sum(r['state_edit_count_after_initialization'] for r in rows),
            'contact_anchors':sum(r['contact_anchor_count'] for r in rows)},
        'fault_results':{w:{m:metrics(take(faults,w,m)) for m in MODES} for w in c.WAITS},
        'healthy_controls':{w:{m:metrics(take(healthy,w,m)) for m in MODES} for w in c.WAITS},
        'paired_methods':{w:{f'{a}_vs_{b}':paired([r for r in faults if r['window']==w],a,b) for a,b in itertools.combinations(MODES,2)} for w in c.WAITS},
        'paired_waits':{m:paired([r for r in faults if r['method']==m],'950ms','actual',by='window') for m in MODES},
        'choice_disagreements':[{'seed':k[0],'scenario':k[1],'window':k[2],
            'choices':{r['method']:r['candidate'] for r in g},'labels':{r['method']:r['audit']['label'] for r in g}}
            for k,g in groups.items() if len({r['candidate'] for r in g})>1],
        'outcome_disagreements':[{'seed':k[0],'scenario':k[1],'window':k[2],
            'choices':{r['method']:r['candidate'] for r in g},'labels':{r['method']:r['audit']['label'] for r in g}}
            for k,g in groups.items() if len({r['audit']['label'] for r in g})>1],
        'scope':'Prospective frozen single-state MLP versus refresh and a conservative wait-history wrapper. No learned delay predictor, oracle labels, model/threshold search or skill modification.'}
    (ROOT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    records=[{'window':w,'split':split,'method':m,**{k:v[k] for k in ('episodes','C','Y','mean_total_seconds','mean_wait_seconds','mean_skill_seconds','changed_initial_choice','invalid_entries')}}
        for split,source in [('fault',result['fault_results']),('healthy',result['healthy_controls'])] for w,mm in source.items() for m,v in mm.items()]
    with (ROOT/'main-results.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    labels={'cached':'Cached','refreshed':'Refreshed','wait_conditioned':'Wait history'}
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Policy & $C$ & $Y$ & Total (s) & Changed\\',r'\midrule']
    for m in MODES:
        r=result['fault_results']['950ms'][m]
        lines.append(f"{labels[m]} & {r['C']}/60 & {r['Y']}/60 & {r['mean_total_seconds']:.3f} & {r['changed_initial_choice']} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'wait-decisions.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('paired_methods','paired_waits')},indent=2))


if __name__=='__main__':main()
