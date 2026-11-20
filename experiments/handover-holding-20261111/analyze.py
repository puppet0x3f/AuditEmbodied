"""Recompute handover holding outcomes and every prespecified paired contrast."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
sys.path.insert(0,str(HERE))
import handover_holding_audit as audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);a=p.parse_args()
    raw=[];rows=[]
    for path in sorted((HERE/a.phase).glob('*.json')):
        r=json.loads(path.read_text());raw.append(r)
        assert r['initialization_supported'] and not r['blocked_calls'],path.name
        assert not any(r['bad_state_warnings'].values()),path.name
        assert r['external_wrench_at_recovery_start']==0,path.name
        assert r['holding_input'] is not None,path.name
        rows.append(audit.score(r,path.parent/r['native_file'],offline))
    report=audit.summarize(rows);report.update(audit.check_pairing(raw))
    out=HERE/(a.phase+'-analysis');out.mkdir(exist_ok=False)
    with (out/'scored.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,compact in [('handover-holding',False),('handover-holding-main',True)]:
        (out/(name+'.tex')).write_text(audit.table(report,compact))
    print(json.dumps({k:v for k,v in report.items() if k not in ('conditions','effects')},indent=2))
    print(json.dumps(report['effects'],indent=2))


if __name__=='__main__':main()
