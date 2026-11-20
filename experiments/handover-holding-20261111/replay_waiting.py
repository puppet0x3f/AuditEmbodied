"""Illustrate two known CAD paths with verified original-action replay."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import imageio.v3 as iio

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'outputs/paired-executor-holding-20261111'
sys.path.insert(0,str(OLD/'frozen-source'))
from auditembodied import l4_lift as runner
from auditembodied.recovery_skills import actual_eef_rotations,actual_gripper_positions


def main():
    scored=[json.loads(s) for s in (OLD/'paired-analysis/scored.jsonl').read_text().splitlines()]
    groups={}
    for r in scored:
        if r['method']=='closed_loop' and r['wait_steps']==20 and r['scenario']!='sham':
            groups.setdefault((r['seed'],r['scenario']),{})[r['hold_mode']]=r
    key=next(k for k,g in sorted(groups.items()) if not g['registered']['native_Y'] and g['measured_position']['native_Y'])
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    factory=suite.PotWithHandlesObject
    def physical_object(*a,**kw):kw['density']=1500.;return factory(*a,**kw)
    folder=HERE/'waiting-keyframes';folder.mkdir(exist_ok=False)
    reports=[]
    for hold in ('registered','measured_position'):
        raw=OLD/'paired'/f'{key[0]}-{key[1]}-20-{hold}-closed_loop.json'
        row=json.loads(raw.read_text());trace=row['trace'];maximum=0.
        indices=[max(i for i,f in enumerate(trace) if f['phase']=='fault'),
                 max(i for i,f in enumerate(trace) if f['stage']=='inference_wait'),
                 next(i for i,f in enumerate(trace) if f['stage']=='global_table_settle'),
                 max(i for i,f in enumerate(trace) if f['stage']=='global_close' and f['time']<=row['global_execution']['pickup_attempts'][0]['close_time']+1e-8),
                 len(trace)-1]
        def compare(current,original):
            nonlocal maximum
            if isinstance(original,dict):
                for k,v in original.items():compare(current[k],v)
            elif isinstance(original,(tuple,list)):
                assert len(current)==len(original)
                for a,b in zip(current,original):compare(a,b)
            else:
                error=abs(float(current)-float(original));maximum=max(maximum,error)
                assert error<1e-8,(current,original,error)
        frames=[]
        with patch.object(suite,'PotWithHandlesObject',physical_object):
            env=runner.TwoArmLiftEnv(row['seed'],runner.LiftConfig(**json.loads((OLD/'config.json').read_text())))
            env.raw.has_offscreen_renderer=True
            try:
                env.reset();compare(runner.snapshot(env),row['initialization_state'])
                env.bootstrap();compare(runner.snapshot(env),row['preclose_state'])
                def forbidden(*a,**kw):raise AssertionError('No edits after bootstrap')
                with ExitStack() as guards:
                    for name in ('reset','bootstrap','_move_arm_to_pose','_sync_after_reset','_sync_after_kinematic_edit'):
                        guards.enter_context(patch.object(env,name,side_effect=forbidden))
                    for i,f in enumerate(trace):
                        env.step(f['robot']['action'])
                        compare(runner.evaluate(env),f['evaluator'])
                        compare([env._eef_pos(j).tolist() for j in range(2)],f['robot']['positions'])
                        compare(actual_gripper_positions(env).tolist(),f['robot']['fingers'])
                        compare(actual_eef_rotations(env).tolist(),f['robot']['rotations'])
                        compare(float(env.sim.data.time),f['time'])
                        if i in indices:
                            from robosuite import macros
                            from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
                            convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
                            before=runner.snapshot(env)
                            rgb=env.sim.render(camera_name='agentview',width=512,height=512)[::convention].copy()
                            j=indices.index(i);iio.imwrite(folder/f'{hold}-{j}.png',rgb)
                            compare(runner.snapshot(env),before)
                            frames.append({'index':j,'stage':f['stage'],'seconds':f['time']-row['prefix_state']['time'],
                                           'holds':f['evaluator']['holds']})
            finally:env.close()
        reports.append({'hold':hold,'seed':key[0],'scenario':key[1],
            'source':str(raw.relative_to(ROOT)),'frames':frames,'original_frames':len(trace),
            'maximum_observation_error':maximum,'all_original_observations_match':True,
            'native_protected_completion':groups[key][hold]['native_Y'],'new_policy_trials':0})
        print(json.dumps(reports[-1]),flush=True)
    (HERE/'waiting-replay-report.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
