import json
from pathlib import Path
from auditembodied.p0 import trial_job
from auditembodied.duration_pilot import file_hash
root=Path('/home/lc/AuditEmbodied');parent=root/'outputs/p0-20261109'
protocol=json.loads((parent/'protocol.json').read_text())
direct=json.loads((parent/'direct-selected.json').read_text())
choices={(10000,'left_loss','protected'),(10000,'coupled_height','direct'),(10000,'right_loss','global')}
references={}
with (parent/'core-trials.jsonl').open() as f:
 for line in f:
  row=json.loads(line);key=(row['seed'],row['scenario'],row['method'])
  if key in choices:references[key]=row
fields=['initialization_state','preclose_state','initialization_end_state','prefix_state','trace',
        'candidate','success','protection_violation','protection','recovery_seconds','skill_seconds',
        'inference_wait_seconds','hold_seconds','total_post_fault_seconds','recovery_path_m']
checks=[]
for (seed,scenario,method),old in sorted(references.items()):
 new=trial_job(((seed,scenario,method,None,None),protocol['test_config'],protocol['rule'],direct))
 differences=[k for k in fields if old.get(k)!=new.get(k)]
 checks.append({'seed':seed,'scenario':scenario,'method':method,'fields_compared':fields,'differences':differences,'passed':not differences})
changed={path:{'parent':digest,'current':file_hash(path)} for path,digest in protocol['physics_source_sha256'].items() if file_hash(path)!=digest}
result={'scope':'Previously observed P0 states only; default runtime compatibility, no new-seed scoring or fitting',
        'changed_sources':changed,'checks':checks,'passed':len(checks)==3 and all(r['passed'] for r in checks)}
(root/'outputs/startgate-runtime-check-20261109.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert result['passed']
