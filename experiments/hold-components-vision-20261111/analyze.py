"""Score unchanged executions and check matched pre-hold states."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'))
import controller_audit as controller
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
sys.path.insert(0,str(HERE))
import geometry as cad
import holding_components_audit as audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--refit',action='store_true');a=p.parse_args()
    rows=[];prefixes=defaultdict(list);fits=0
    for path in sorted((HERE/a.phase).glob('*.json')):
        r=json.loads(path.read_text())
        assert r['nominal_ready'] and r['fault_class_valid'] and not r.get('error'),path.name
        assert not r['blocked_calls'] and r['contact_anchor_count']==r['state_edit_count_after_initialization']==0,path.name
        result,count=audit.score(r,path.parent/r['native_file'],offline.score_lift,controller,
            cad.fit if a.refit and r['method']=='closed_loop' else None)
        rows.append(result);fits+=count
        prefixes[r['seed'],r['scenario']].append({k:r[k] for k in
            ('initialization_state','preclose_state','initialization_end_state','prefix_state')})
    for key,group in prefixes.items():assert all(x==group[0] for x in group),key
    report=audit.summarize(rows)
    report.update(matched_prehold_groups=len(prefixes),pose_fits_recomputed=fits)
    out=HERE/(a.phase+'-analysis');out.mkdir(exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'scored.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (out/('independent-visual.tex' if 'independent' in a.phase else 'holding-components.tex')).write_text(audit.table(report))
    print(json.dumps({'phase':a.phase,'episodes':len(rows),'matched_prehold_groups':len(prefixes),'native_samples':report['native_samples']}))
    print(json.dumps(report['fault_results'],indent=2));print(json.dumps(report['paired_effects'],indent=2))


if __name__=='__main__':main()
