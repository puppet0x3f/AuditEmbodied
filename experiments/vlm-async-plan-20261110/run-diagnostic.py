"""Fixed-readout actuation diagnosis; original waits are replayed, not removed."""
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing
from pathlib import Path
from auditembodied.duration_continuation import run_continuation
from auditembodied.mid_recovery_data import outcome
from auditembodied.duration_closed_loop import trial_summary
from auditembodied.vlm_mid_recovery import select_at_step

ROOT=Path(__file__).resolve().parent
TAU=.999667229515218


def choose(meta):
    first=meta['answers']['20'];second=meta['answers']['30']
    if meta['original_ready']:
        d=select_at_step(first['records'],TAU,20,30,True)
        if d['action']=='R_global':return 'R_global',30,d
        if d['action']=='reject':return 'reject',30,d
        if d['action']=='continue':return 'continue',30,d
        prior_global=d
    else:
        if not meta['initial_admitted']:return 'reject',30,{'reason':'no_prior_admission'}
        prior_global=None
    # This is a timing-controlled diagnosis, using the original first response
    # latency and the measured frozen second readout. Online costs are tested
    # separately in the 96 native episodes.
    if second['client_seconds']<=.5:
        d=select_at_step(second['records'],TAU,30,40)
    elif meta['original_latency']<=1.:
        d=select_at_step(first['records'],TAU,20,40)
    elif prior_global is not None:
        d={**prior_global,'action':'R_global','carried_admission':True}
    elif meta['initial_admitted']:
        d={'action':'continue','carried_local_admission':True}
    else:d={'action':'reject'}
    return d['action'],40,d


def execute(job):
    source,local,method,meta=job
    from auditembodied.l0 import _import_robosuite
    _import_robosuite(True)
    if method=='original':
        branch=source['mid_selection']['windows'][0]['selected_branch'];step=30;decision=source['mid_selection']['windows'][0]['decision']
    else:branch,step,decision=choose(meta)
    base=source if step==30 else local
    if branch=='R_global':
        assert decision.get('model_admitted')
        if method=='selective':assert decision['activation_step']==step
    row=run_continuation((base,branch),transition='defer_rotation',account_control_latency=True,boundary=f'relevel{step}')
    assert not row['diagnostic_errors'],row['diagnostic_errors']
    row.update(method=method,selected_step=step,selected_branch=branch,frozen_decision=decision,
        original_label=outcome(source),original_total_seconds=source['total_post_fault_seconds'])
    return row


def main():
    out=ROOT/'diagnostic28';out.mkdir(exist_ok=True)
    assert not (out/'trials.jsonl').exists()
    specs=[json.loads(s) for s in (ROOT/'diagnostic-sources.jsonl').open()]
    meta={(r['seed'],r['scenario']):r for r in map(json.loads,(ROOT/'diagnostic-predictions.jsonl').open())}
    jobs=[]
    for i,s in enumerate(specs):
        source=s['source'];m=meta[(source['seed'],source['scenario'])]
        for method in (('original','selective') if i%2==0 else ('selective','original')):jobs.append((source,s['local'],method,m))
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (out/'trials.jsonl').open('w') as f:
            for r in pool.map(execute,jobs):
                rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                print(json.dumps({k:r[k] for k in ['seed','scenario','method','selected_step','selected_branch','success','protection_violation','total_post_fault_seconds']}),flush=True)
    report={'diagnostic_only':True,'used_for_training':False,'physical_states':14,'episodes':len(rows),
        'original_prefix_waits_replayed':True,'cumulative_guards':True,'new_online_latency_claim':False,
        'availability':'Original first query readiness/latency; frozen second readout measured on the current same-host worker. No recorded waits subtracted.',
        'methods':{m:{**trial_summary([r for r in rows if r['method']==m]),
            'abstentions':sum(r['selected_branch']=='reject' for r in rows if r['method']==m)} for m in ['original','selective']},
        'cases':[{k:r[k] for k in ['seed','scenario','method','selected_step','selected_branch','success','protection_violation','total_post_fault_seconds','original_label']} for r in rows]}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
