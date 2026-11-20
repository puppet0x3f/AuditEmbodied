"""Descriptive full-sample analysis of the already frozen shared-prefix cohort."""
import json
from pathlib import Path
import execution_error_audit as audit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'outputs/paired-executor-holding-20261111'


def main():
    key=lambda r:tuple(r[k] for k in ('seed','scenario','method','hold_mode','wait_steps'))
    scored={key(r):r for r in (json.loads(s) for s in (SOURCE/'paired-analysis/scored.jsonl').read_text().splitlines())}
    rows=[]
    for path in sorted((SOURCE/'paired').glob('*.json')):
        raw=json.loads(path.read_text());rows.append(audit.score(raw,scored[key(raw)]))
    assert len(rows)==960
    report=audit.summarize(rows)
    out=HERE/'existing-analysis';out.mkdir(exist_ok=False)
    with (out/'cases.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='conditions'}))


if __name__=='__main__':main()
