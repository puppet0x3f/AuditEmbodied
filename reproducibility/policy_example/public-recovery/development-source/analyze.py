import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'reproducibility'))
import public_recovery_audit as audit
p=argparse.ArgumentParser();p.add_argument('--phase',choices=('development','formal'),required=True);a=p.parse_args()
raw=[];rows=[];checks=0
for f in sorted((HERE/a.phase).glob('*.json')):
    r=json.loads(f.read_text());raw.append(r);rows.append(audit.score(r))
    port=r.get('recovery_policy_input')
    if port:
        with np.load(HERE/a.phase/port['filename'],allow_pickle=False) as inputs:
            for k,digest in port['sha256'].items():
                assert hashlib.sha256(np.ascontiguousarray(inputs[k]).tobytes()).hexdigest()==digest
                checks+=1
out=HERE/(a.phase+'-analysis');out.mkdir(exist_ok=False)
if a.phase=='formal':
    assert len(rows)==160
    report=audit.summarize(rows);report.update(audit.pairing(raw))
    (out/'public-recovery.tex').write_text(audit.table(report))
else:
    report={'development_attempts':len(rows),'fault':{k:sum(bool(r[k]) for r in rows if r['scenario']=='fault') for k in ('C','C_native','event_triggered','fault_valid')},
            'sham':{k:sum(bool(r[k]) for r in rows if r['scenario']=='sham') for k in ('C','C_native','event_triggered','fault_valid')},**audit.pairing(raw)}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'scored.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
print(json.dumps({'report':report,'policy_input_tensors_checked':checks},indent=2))
