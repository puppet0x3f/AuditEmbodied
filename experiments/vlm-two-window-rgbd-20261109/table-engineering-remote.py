import json,shutil,time
from pathlib import Path
from unittest.mock import patch
from auditembodied import l4_lift
from auditembodied.table_rgbd import acquire
from auditembodied.duration_continuation import run_continuation
from auditembodied.l0 import _import_robosuite
root=Path('outputs/vlm-two-window-rgbd-20261109');out=root/'training-table-v4';out.mkdir(exist_ok=False)
source=json.loads((root/'remote-engineering-loop-v2/trials.jsonl').open().readline())
source['trace']=source['trace'][:source['continuation_diagnostic']['boundary_trace_length']]
(out/'source-prefix.json').write_text(json.dumps(source))
for name in ['table_rgbd.py','table_assist.py']:
 shutil.copy2(Path('auditembodied')/name,out/name)
_import_robosuite(True);original=l4_lift.TwoArmLiftEnv

def factory(*args,**kwargs):
 env=original(*args,**kwargs);env.raw.has_offscreen_renderer=True;reset=env.reset
 def warmed():reset();acquire(env)
 env.reset=warmed;return env
with patch.object(l4_lift,'TwoArmLiftEnv',factory):
 row=run_continuation((source,'R_table_assist'),transition='defer_rotation',account_control_latency=True,boundary='search_exhausted',table_assist_template=str((root/'table-template.npz').resolve()))
(out/'trial.json').write_text(json.dumps(row))
print(json.dumps({k:row.get(k) for k in ['success','protection','abstention_reason','error','diagnostic_errors','total_post_fault_seconds','inference_wait_seconds']}),flush=True)
print(json.dumps({k:v for k,v in row.get('table_assist',{}).items() if k not in ['telemetry','registrations']}),flush=True)
