"""Independent logged-sample audit of development or frozen formal executions."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'reproducibility'))
import controller_audit as audit
spec=importlib.util.spec_from_file_location('offline',REPO/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
sys.path.insert(0,str(HERE))
from geometry import fit


def analyze(folder,refit=False):
    rows=[];prefixes=defaultdict(dict);fits=0;worst_error=0.
    for path in sorted(folder.glob('*-closed_loop.json'))+sorted(folder.glob('*-retarget.json')):
        row=json.loads(path.read_text())
        assert row['nominal_ready'] and row['fault_class_valid'] and not row.get('error'),path.name
        assert not row['blocked_calls'] and row['contact_anchor_count']==row['state_edit_count_after_initialization']==0
        result,observations=audit.score(row,offline.score_lift)
        result.update(audit.native_view(folder/row['native_file'],row,result))
        prefixes[row['seed'],row['scenario'],row['load'],row['window']][row['method']]=row
        if refit:
            for d in observations:
                e=fit(d['input']);fits+=1
                assert e['valid']==d['estimate']['valid']
                if e['valid']:
                    for key in ('center','rotation','targets','optimization_vector'):
                        err=float(np.max(np.abs(np.asarray(e[key])-d['estimate'][key])));worst_error=max(worst_error,err)
                        assert err<1e-5
        rows.append(result)
    matched=0
    for group in prefixes.values():
        if len(group)!=2:continue
        a,b=group['retarget'],group['closed_loop']
        for key in ('initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state'):
            assert a[key]==b[key],(a['seed'],key)
        matched+=1
    report=audit.summarize(rows)
    report.update(matched_prefix_groups=matched,pose_fits_recomputed=fits,maximum_pose_refit_error=worst_error)
    return report,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=HERE/'formal');p.add_argument('--refit',action='store_true')
    p.add_argument('--out',type=Path,default=HERE/'analysis');a=p.parse_args()
    report,rows=analyze(a.folder,a.refit);a.out.mkdir(exist_ok=False)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.out/'scored.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    for name,table in audit.tables(report).items():(a.out/(name+'.tex')).write_text(table)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
