"""Render matched known paths using saved actions; no new policy evaluation."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import imageio.v3 as iio

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied import l4_lift as runner
from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=('rescued','limited'),required=True);a=p.parse_args()
    rows=[json.loads(l) for l in (ROOT/'trials.jsonl').open()]
    groups={}
    for r in rows:
        if r['load']=='density150' and r['scenario']!='sham':
            groups.setdefault((r['seed'],r['scenario'],r['window']),{})[r['method']]=r
    key=next(k for k,g in sorted(groups.items()) if
        ((g['retarget']['audit']['label']=='A' and g['refresh']['audit']['label']!='A') if a.case=='rescued' else
         (g['retarget']['audit']['label']!='A')))
    cfg=json.loads((ROOT/'protocol.json').read_text())['test_config']
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    original=suite.PotWithHandlesObject
    def scaled(*args,**kwargs):kwargs['density']=1500.;return original(*args,**kwargs)
    out=ROOT/'paper-keyframes';out.mkdir(exist_ok=True)
    for method in ('refresh','retarget'):
        row=groups[key][method];trace=row['trace']
        indices=[max(i for i,f in enumerate(trace) if f['stage']==s) for s in
                 ('global_clear','global_descend','global_close','final_stable_hold')]
        images=[];maximum=0.
        def compare(x,y):
            nonlocal maximum
            if isinstance(y,dict):
                for k,v in y.items():compare(x[k],v)
            elif isinstance(y,(list,tuple)):
                assert len(x)==len(y)
                for aa,bb in zip(x,y):compare(aa,bb)
            else:
                d=abs(float(x)-float(y));maximum=max(maximum,d);assert d<1e-8,(x,y,d)
        with patch.object(suite,'PotWithHandlesObject',scaled):
            env=runner.TwoArmLiftEnv(row['seed'],runner.LiftConfig(**cfg));env.raw.has_offscreen_renderer=True
            try:
                env.reset();compare(runner.snapshot(env),row['initialization_state'])
                env.bootstrap();compare(runner.snapshot(env),row['preclose_state'])
                def forbidden(*aa,**kk):raise AssertionError('No state restoration after bootstrap')
                with ExitStack() as guards:
                    for name in ('reset','bootstrap','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit'):
                        guards.enter_context(patch.object(env,name,side_effect=forbidden))
                    for i,f in enumerate(trace):
                        env.step(f['robot']['action'])
                        compare(runner.evaluate(env),f['evaluator'])
                        compare(np.stack([env._eef_pos(j) for j in range(2)]).tolist(),f['robot']['positions'])
                        compare(actual_gripper_positions(env).tolist(),f['robot']['fingers'])
                        compare(actual_eef_rotations(env).tolist(),f['robot']['rotations'])
                        compare(float(env.sim.data.time),f['time'])
                        if i in indices:
                            from robosuite import macros
                            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
                            convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
                            before=runner.snapshot(env)
                            for camera in ('agentview','frontview'):
                                rgb=env.sim.render(camera_name=camera,width=512,height=512)[::convention].copy()
                                iio.imwrite(out/f'{a.case}-{method}-{indices.index(i)}-{camera}.png',rgb)
                            compare(runner.snapshot(env),before)
                            images.append({'index':indices.index(i),'trace_index':i,'stage':f['stage'],
                                'seconds_after_fault':f['time']-row['prefix_state']['time'],'evaluator':f['evaluator']})
            finally:env.close()
        meta={'case':a.case,'method':method,'seed':row['seed'],'scenario':row['scenario'],'load':row['load'],
            'window':row['window'],'label':row['audit']['label'],'original_trace_frames':len(trace),
            'maximum_original_observation_error':maximum,'all_original_observations_match':True,'frames':images,
            'scope':'Known action replay; illustrative case selection only, not new independent policy trials.'}
        (out/f'{a.case}-{method}.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
