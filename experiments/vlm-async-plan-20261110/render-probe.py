import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from auditembodied import l4_lift
from auditembodied.l0 import _import_robosuite
from auditembodied.duration_closed_loop import render_rgb
from auditembodied.async_rgb import EntryRenderer

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
_import_robosuite(True)
holder={};stats=[];original=l4_lift.TwoArmLiftEnv
def factory(*args,**kwargs):
    env=original(*args,**kwargs);holder['env']=env;env.raw.has_offscreen_renderer=True
    return env
for seed,scenario in [(40,'left_loss'),(40,'right_loss'),(51,'left_loss'),(51,'right_loss')]:
    def capture(stage):
        env=holder['env'];before=l4_lift.snapshot(env);expected=render_rgb(env)
        renderer=EntryRenderer(env);actual,meta=renderer.submit(env).result();renderer.close()
        assert np.array_equal(expected,actual),(seed,scenario,stage,np.max(np.abs(expected.astype(int)-actual)))
        assert before==l4_lift.snapshot(env)
        np.savez_compressed(out/f'{seed}-{scenario}-{stage}.npz',rgb=actual)
        stats.append({'seed':seed,'scenario':scenario,'stage':stage,**meta,'pixels_equal':True})
    def select(port):capture('h0');return {'candidate':'R_left' if scenario=='left_loss' else 'R_right','local_relevel_steps':60}
    def mid(op,port,candidate):
        if op=='start':capture('entry')
        if op=='finish':return {'action':'continue','forecast_ready':True,'attempt_id':port['attempt_id'],'model_admitted':True}
    with patch.object(l4_lift,'TwoArmLiftEnv',factory):
        row=l4_lift.run_trial(seed,scenario,selector=select,fixed_selector_wait_steps=19,continuation_selector=mid,
            continuation_admission_policy='consistent')
    assert not row.get('error'),row.get('error')
(out/'report.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats),flush=True)
