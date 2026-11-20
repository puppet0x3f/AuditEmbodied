"""Recompute the bounded recovery follow-up after the unchanged base evaluator.

Python and NumPy only. No simulator, GPU, source workspace or model training.
"""
import argparse,gzip,hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np
import recompute as base
import controller_audit as controller
import prospective_geometry as geometry
import ranking_closure_audit as ranking
import budgeted_cad_audit as adaptive
import public_recovery_audit as public


def read(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False);(out/'tables').mkdir()
    old=[r for r in (json.loads(s) for s in (args.base_results/'scored-episodes.jsonl').read_text().splitlines()) if r['cohort']=='paired_holding']
    assert len(old)==960
    cases=[ranking.closure(r) for r in read(HERE/'data/ranking-closure-inputs.jsonl.gz')]
    report=ranking.summarize(old,cases)
    assert report==json.loads((HERE/'reference/ranking-closure-results.json').read_text())
    tables=ranking.tables(report)
    print('Recomputed paired method intervals and 960 first-command checks.',flush=True)
    geometry.configure('pot');new=[];states=[];fits=0
    for row in read(HERE/'data/budgeted_cad.jsonl.gz'):
        scored,n=adaptive.score(row,HERE/'data/budgeted-cad-native'/row['native_file'],base.score_lift,controller,geometry.fit)
        new.append(scored);fits+=n
        states.append({k:row[k] for k in ('seed','scenario','method','hold_mode','wait_steps','initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state')})
    oldstates=[]
    for row in read(HERE/'data/paired_holding.jsonl.gz'):
        if row['method']=='closed_loop' and row['hold_mode'] in ('registered','measured_position'):
            oldstates.append({k:row[k] for k in states[0]})
    adaptive_report=adaptive.summarize(new,old)
    adaptive_report.update(adaptive.pairing(states,oldstates),pose_fits_recomputed=fits)
    assert adaptive_report==json.loads((HERE/'reference/budgeted-cad-results.json').read_text())
    tables['budgeted-cad']=adaptive.table(adaptive_report)
    print(f'Recomputed {len(new)} arrival-adaptive executions, {fits} CAD fits and exact baseline pairing.',flush=True)
    public_report=None;public_input_checks=0
    if (HERE/'data/public_recovery.jsonl.gz').exists():
        pub=[];pubraw=[]
        for row in read(HERE/'data/public_recovery.jsonl.gz'):
            pub.append(public.score(row));pubraw.append(row)
            port=row.get('recovery_policy_input')
            if port:
                with np.load(HERE/'data/public-recovery-inputs'/port['filename'],allow_pickle=False) as inputs:
                    for k,digest in port['sha256'].items():
                        assert hashlib.sha256(np.ascontiguousarray(inputs[k]).tobytes()).hexdigest()==digest
                        public_input_checks+=1
        public_report=public.summarize(pub);public_report.update(public.pairing(pubraw))
        assert public_report==json.loads((HERE/'reference/public-recovery-results.json').read_text())
        tables['public-recovery']=public.table(public_report)
    for name,content in tables.items():
        assert content.split()==(HERE/'reference/tables'/(name+'.tex')).read_text().split(),name
        (out/'tables'/(name+'.tex')).write_text(content)
    result={'new_formal_executions_recomputed':len(new)+(public_report['formal_executions'] if public_report else 0),
        'maximum_new_formal_executions':400,'original_tables_replaced':0,
        'ranking_closure_report':report,'budgeted_cad_report':adaptive_report,
        'public_recovery_report':public_report,'public_input_tensors_checked':public_input_checks,
        'new_tables_match':list(tables)}
    assert result['new_formal_executions_recomputed']<=400
    (out/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'scored-adaptive-episodes.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in new))
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_report')},indent=2))

if __name__=='__main__':main()
