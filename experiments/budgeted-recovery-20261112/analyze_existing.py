import json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'))
import ranking_closure_audit as audit
source=ROOT/'outputs/paired-executor-holding-20261111'
rows=[json.loads(s) for s in (source/'paired-analysis/scored.jsonl').read_text().splitlines()]
timed=[]
for p in sorted((source/'paired').glob('*.json')):
    timed.append(audit.closure(json.loads(p.read_text())))
assert len(rows)==len(timed)==960
report=audit.summarize(rows,timed)
out=HERE/'existing-analysis';out.mkdir(exist_ok=False)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'closure-cases.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in timed))
for name,value in audit.tables(report).items():(out/(name+'.tex')).write_text(value)
print(json.dumps({'ranking':report['ranking'],'closing_samples':report['closure']['closing_samples'],'action_checks':report['closure']['action_target_checks']}))
