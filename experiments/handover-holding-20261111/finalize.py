"""Frozen outcome/statistical scoring plus explicit charged-timing quality counts.

The formal controller and its source freeze are unchanged. Unlike the original
strict pairing assertion, this report retains actual charged overrun episodes.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
spec=importlib.util.spec_from_file_location('handover_results',ROOT/'reproducibility/handover_holding_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    completion=json.loads((HERE/'formal-completion.json').read_text());assert len(completion)==720
    raw=[];rows=[]
    for path in sorted((HERE/'formal').glob('*.json')):
        r=json.loads(path.read_text());raw.append(r)
        assert not r['blocked_calls'],path.name
        if r['holding_input'] is not None:
            assert r['initialization_supported'] and not any(r['bad_state_warnings'].values()),path.name
            assert r['external_wrench_at_recovery_start']==0,path.name
        rows.append(audit.score(r,path.parent/r['native_file'],offline))
    assert len(raw)==720
    report=audit.summarize(rows);report.update(audit.check_pairing(raw))
    report['timing_sensitivity']=audit.timing_sensitivity(rows,raw)
    out=HERE/'formal-analysis';out.mkdir(exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'scored.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    with (HERE/'formal-trials.jsonl').open('x') as f:
        for r in raw:f.write(json.dumps(r,separators=(',',':'))+'\n')
    for name,compact in [('handover-holding',False),('handover-holding-main',True)]:
        content=audit.table(report,compact)
        (out/(name+'.tex')).write_text(content)
        (ROOT/'paper/tables'/(name+'.tex')).write_text(content)
    print(json.dumps({k:v for k,v in report.items() if k not in ('conditions','effects','timing_sensitivity')},indent=2))
    print(json.dumps(report['effects'],indent=2))


if __name__=='__main__':main()
