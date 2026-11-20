import json,sys,time,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from auditembodied import l4_lift
from auditembodied.duration_continuation import run_continuation
from auditembodied.l0 import _import_robosuite
from auditembodied.table_rgbd import acquire,build_template
from auditembodied.recovery_skills import actual_eef_rotations
root=Path('outputs/vlm-two-window-rgbd-20261109');folder=root/sys.argv[1];folder.mkdir(exist_ok=False)
for name in ('table_rgbd.py','table_assist.py','l4_lift.py','duration_continuation.py'):shutil.copy2(Path('auditembodied')/name,folder/name)
_import_robosuite(True)
original=l4_lift.TwoArmLiftEnv
# A training-side fixed reference, measured before any fault.
if sys.argv[1]=='table-v1':
 e=original(50);e.raw.has_offscreen_renderer=True;e.reset();e.bootstrap()
 for _ in range(10):e.step(e.action(e.table_targets))
 print(build_template(e,root/'table-template.npz',np.stack([e._eef_pos(i) for i in range(2)]),actual_eef_rotations(e)),flush=True);e.close()
def factory(*a,**k):
 e=original(*a,**k);e.raw.has_offscreen_renderer=True;r=e.reset
 def reset():r();acquire(e)
 e.reset=reset;return e
source=json.loads(Path('outputs/vlm-late-switch-table-assist-20261109/engineering-source.json').read_text())
start=time.perf_counter()
with patch.object(l4_lift,'TwoArmLiftEnv',factory):
 row=run_continuation((source,'R_table_assist'),transition='defer_rotation',account_control_latency=True,boundary='search_exhausted',table_assist_template=str((root/'table-template.npz').resolve()))
(folder/'trial.json').write_text(json.dumps(row))
print(json.dumps({k:row.get(k) for k in ('success','protection_violation','protection','failure_reason','error','abstention_reason','total_post_fault_seconds','inference_wait_seconds','diagnostic_errors')}),flush=True)
print(json.dumps({k:v for k,v in row.get('table_assist',{}).items() if k!='telemetry'}),flush=True)
print('wall',time.perf_counter()-start,flush=True)
