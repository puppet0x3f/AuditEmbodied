import argparse,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'reproducibility'))
import controller_audit as controller
import budgeted_cad_audit as audit
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
sys.path.insert(0,str(HERE));import geometry as cad
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--refit',action='store_true');a=p.parse_args()
raw=[];rows=[];fits=0
for path in sorted((HERE/a.phase).glob('*.json')):
    r=json.loads(path.read_text());raw.append(r)
    assert r['nominal_ready'] and r['fault_class_valid'] and not r.get('error'),path.name
    scored,n=audit.score(r,path.parent/r['native_file'],offline.score_lift,controller,cad.fit if a.refit else None)
    rows.append(scored);fits+=n
out=HERE/(a.phase+'-analysis');out.mkdir(exist_ok=False)
(out/'scored.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
if a.phase=='formal':
    source=ROOT/'outputs/paired-executor-holding-20261111'
    old=[json.loads(s) for s in (source/'paired-analysis/scored.jsonl').read_text().splitlines()]
    report=audit.summarize(rows,old)
    oldraw=[json.loads(p.read_text()) for p in sorted((source/'paired').glob('*.json'))
            if '-closed_loop' in p.name and ('-registered-' in p.name or '-measured_position-' in p.name)]
    report.update(audit.pairing(raw,oldraw),pose_fits_recomputed=fits)
    (out/'budgeted-cad.tex').write_text(audit.table(report))
else:
    report={'episodes':len(rows),'C':sum(r['C'] for r in rows),'Y500':sum(r['native_Y'] for r in rows),'pose_fits_recomputed':fits}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
