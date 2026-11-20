import json,time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
from unittest.mock import patch
from auditembodied import l4_lift
from auditembodied.table_rgbd import acquire
from auditembodied.duration_continuation import run_continuation
from auditembodied.mid_recovery_data import read,outcome
from auditembodied.mid_recovery_diagnostic import SEARCH_CASES,ROOT
from auditembodied.duration_closed_loop import trial_summary
from auditembodied.l0 import _import_robosuite
ROOT_OUT=Path('outputs/vlm-two-window-rgbd-20261109')
def job(source):
 _import_robosuite(True);original=l4_lift.TwoArmLiftEnv
 def factory(*a,**k):
  e=original(*a,**k);e.raw.has_offscreen_renderer=True;r=e.reset
  def reset():r();acquire(e)
  e.reset=reset;return e
 with patch.object(l4_lift,'TwoArmLiftEnv',factory):
  return run_continuation((source,'R_table_assist'),transition='defer_rotation',account_control_latency=True,boundary='search_exhausted',table_assist_template=str((ROOT_OUT/'table-template.npz').resolve()))
if __name__=='__main__':
 out=ROOT_OUT/'table-diagnostic';out.mkdir(exist_ok=False)
 sources={(r['seed'],r['scenario']):r for r in read(ROOT/'closed-loop/trials.jsonl') if r['method']=='mid_selective'}
 rows=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  with (out/'trials.jsonl').open('w') as f:
   for r in pool.map(job,[sources[k] for k in SEARCH_CASES]):
    rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
    print(json.dumps({'seed':r['seed'],'scenario':r['scenario'],'label':outcome(r),'reason':r.get('abstention_reason'),'errors':r['diagnostic_errors']}),flush=True)
 report={'diagnostic_only':True,'used_for_training':False,'tuning_after_diagnostic':False,'summary':trial_summary(rows),'cases':[{'seed':r['seed'],'scenario':r['scenario'],'label':outcome(r),'reason':r.get('abstention_reason'),'protection':r['protection'],'attempts':r['table_assist']['attempts'],'seconds':r['total_post_fault_seconds'],'wait':r['inference_wait_seconds'],'errors':r['diagnostic_errors']} for r in rows]}
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
