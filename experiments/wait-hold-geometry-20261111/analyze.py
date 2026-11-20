"""Audit the new physical campaigns using the distributable numerical evaluator."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'))
import controller_audit as controller
import wait_geometry_audit as audit
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
sys.path.insert(0,str(HERE))
import geometry as cad


def analyze(phase,refit=False):
    folder=HERE/phase;rows=[];fits=0;prefixes=defaultdict(dict)
    for path in sorted(folder.glob('*.json')):
        row=json.loads(path.read_text())
        assert row['nominal_ready'] and row['fault_class_valid'] and not row.get('error'),path.name
        assert not row['blocked_calls'] and row['contact_anchor_count']==row['state_edit_count_after_initialization']==0
        cad.configure(row['geometry'])
        result,count=audit.score(row,folder/row['native_file'],offline.score_lift,controller,cad.fit if refit else None)
        rows.append(result);fits+=count
        key=(row['seed'],row['scenario'],row['geometry'],row['hold_mode'],row['wait_steps'])
        # Retain only the five short snapshots for the pairing check.
        prefixes[key][row['method']]={k:row[k] for k in
            ('initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state')}
    matched=0
    for key,group in prefixes.items():
        assert set(group)=={'retarget','closed_loop'},key
        assert group['retarget']==group['closed_loop'],key
        matched+=1
    report=audit.summarize(rows)
    report.update(matched_method_prefixes=matched,pose_fits_recomputed=fits)
    out=HERE/(phase+'-analysis');out.mkdir(exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'scored.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    for name,table in audit.tables(report,'geometry' in phase).items():(out/(name+'.tex')).write_text(table)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--refit',action='store_true')
    a=p.parse_args();print(json.dumps(analyze(a.phase,a.refit),indent=2))
