"""Actuator-replay pilot of relevel40 switching and table-assisted regrasp."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import multiprocessing
from pathlib import Path

from .duration_closed_loop import trial_summary
from .duration_continuation import run_continuation
from .duration_pilot import file_hash
from .mid_recovery_data import read,outcome
from .mid_recovery_diagnostic import prepare,finish,OLD,ROOT,SEARCH_CASES


LATE_CASES=((24000,'right_loss'),(24005,'right_loss'),(24007,'right_loss'))


def campaign(output,mode,workers):
    if mode=='late':
        cohort=LATE_CASES;path=OLD/'trials.jsonl';method='family_set';boundary='relevel40';branches=('continue','R_global')
    else:
        cohort=SEARCH_CASES;path=ROOT/'closed-loop/trials.jsonl';method='mid_selective';boundary='search_exhausted';branches=('R_global','R_table_assist')
    sources={(r['seed'],r['scenario']):r for r in read(path) if r['method']==method}
    sources={k:sources[k] for k in cohort}
    protocol=prepare(output,mode,list(cohort),2*len(cohort))
    protocol.update(source_trials=str(path.resolve()),source_sha256=file_hash(path),boundary=boundary,
        branches=list(branches),model_inference_rerun=False,training_started=False)
    for name in ('recovery_physical_pilot.py','table_assist.py','recovery_skills.py'):
        p=Path(__file__).parent/name
        if p.exists():(output/'source'/name).write_bytes(p.read_bytes());protocol['runtime_sources'][str(p.resolve())]=file_hash(p)
    if mode=='search':
        from dataclasses import asdict
        from .table_assist import TableAssistConfig
        protocol['table_assist_parameters']=asdict(TableAssistConfig())
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    run=partial(run_continuation,transition='defer_rotation',account_control_latency=True,boundary=boundary)
    jobs=[(r,b) for r in sources.values() for b in branches];rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        with (output/'trials.jsonl').open('w') as stream:
            for row in pool.map(run,jobs):
                stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                print(json.dumps({'seed':row['seed'],'scenario':row['scenario'],'branch':row.get('continuation_diagnostic',{}).get('branch'),
                    'label':outcome(row),'seconds':row.get('total_post_fault_seconds'),'errors':row['diagnostic_errors']}),flush=True)
                if row['diagnostic_errors']:raise RuntimeError(row['diagnostic_errors'])
    pairs={}
    for r in rows:pairs.setdefault((r['seed'],r['scenario']),{})[r['continuation_diagnostic']['branch']]=r
    cases=[]
    for key,pair in pairs.items():
        a,b=(pair[k] for k in branches)
        for name in ('boundary_state','boundary_robot','boundary_position_targets','boundary_rotation_targets','protection_before','boundary_wait_steps'):
            assert a['continuation_diagnostic'][name]==b['continuation_diagnostic'][name],(key,name)
        cases.append({'seed':key[0],'scenario':key[1],
            'labels':{k:outcome(pair[k]) for k in branches},
            'seconds':{k:pair[k]['total_post_fault_seconds'] for k in branches},
            'keeper_lost_steps':{k:pair[k]['protection']['keeper_grasp_lost_steps'] for k in branches},
            'unsupported_seconds':{k:pair[k]['protection']['max_unsupported_seconds'] for k in branches}})
    finish(output,protocol,{'episodes':len(rows),'methods':{k:trial_summary([r for r in rows if r['continuation_diagnostic']['branch']==k]) for k in branches},'cases':cases})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--mode',choices=('late','search','all'),default='all');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();root=Path(args.output_dir).resolve()
    if args.mode in ('late','all'):campaign(root/'switch40', 'late',args.workers)
    if args.mode in ('search','all'):campaign(root/'table-assist','search',args.workers)


if __name__=='__main__':main()
