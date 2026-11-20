import argparse,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'reproducibility'));sys.path.insert(0,str(HERE.parent))
import controller_audit as controller
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
import lifting_smooth_audit as audit
sys.path.insert(0,str(HERE));import geometry as cad
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--refit',action='store_true');a=p.parse_args()
raw=[];rows=[];fits=0
for path in sorted((HERE/a.phase).glob('*.json')):
    r=json.loads(path.read_text());raw.append(r)
    assert r['nominal_ready'] and r['fault_class_valid'] and not r.get('error'),path.name
    assert r['contact_anchor_count']==r['state_edit_count_after_initialization']==0
    scored,n=audit.score(r,path.parent/r['native_file'],offline.score_lift,controller,cad.fit if a.refit else None)
    rows.append(scored);fits+=n
report=audit.summarize(rows);report.update(audit.pairing(raw),pose_fits_recomputed=fits)
out=HERE/(a.phase+'-analysis');out.mkdir()
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'scored.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
(out/'lifting-smooth.tex').write_text(audit.table(report))
print(json.dumps(report,indent=2))
