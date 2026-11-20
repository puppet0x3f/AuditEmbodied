"""Development feasibility only; frozen execution gains and native physical guards."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import imageio.v3 as iio

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied import l4_lift as runner
from geometry import CAMERAS,SIZE,mask_features,fit


def reference(targets,rotations):
    rz=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rot=np.asarray(rotations)[0]@np.diag([-1.,1.,-1.])@rz
    center=np.asarray(targets).mean(0)+[0.,0.,.0036]-rot@np.array([0.,0.,.06])
    return center,rot


def projection(env,camera):
    # Fixed calibrated cameras only; no movable body/object measurements.
    idx=env.sim.model.camera_name2id(camera)
    f=SIZE/2/np.tan(np.deg2rad(env.sim.model.cam_fovy[idx])/2)
    k=np.array([[f,0,SIZE/2],[0,f,SIZE/2],[0,0,1.]])
    rot=np.asarray(env.sim.data.cam_xmat[idx]).reshape(3,3)@np.diag([1.,-1.,-1.])
    xyz=np.asarray(env.sim.data.cam_xpos[idx])
    image_rows=np.array([[1.,0.,0.],[0.,-1.,SIZE-1.],[0.,0.,1.]])
    return image_rows@k@np.c_[rot.T,-rot.T@xyz]


class Observer:
    def __init__(self,path,apply):self.path=path;self.apply=apply
    def __call__(self,env,targets,rotations):
        started=time.perf_counter()
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        center,rot=reference(targets,rotations)
        port={'reference_center':center.tolist(),'reference_rotation':rot.tolist(),
            'reference_targets':np.asarray(targets).tolist(),'reference_rotation_targets':np.asarray(rotations).tolist(),'views':[]}
        images=[]
        before=runner.snapshot(env)
        for camera in CAMERAS:
            rgb=env.sim.render(camera_name=camera,width=SIZE,height=SIZE)[::convention].copy()
            port['views'].append({'camera':camera,'projection':projection(env,camera).tolist(),'colors':mask_features(rgb)})
            images.append(rgb)
        assert runner.snapshot(env)==before
        estimate=fit(port)
        elapsed=time.perf_counter()-started
        self.path.mkdir(parents=True,exist_ok=True)
        for camera,rgb in zip(CAMERAS,images):iio.imwrite(self.path/f'{camera}.png',rgb)
        # Evaluator-only diagnostics are collected after the policy has returned.
        truth=runner.evaluate(env)
        delta=np.asarray(estimate.get('targets',targets))-np.asarray(truth['handle_positions'])
        result={'apply':self.apply,'estimate':estimate,'input':port,'service_wall_seconds':elapsed,
            'service_wait_steps':max(20,runner.wait_control_steps(elapsed)),
            'observation_state':before,'observation_truth':truth,
            'estimated_handle_error_m':np.linalg.norm(delta,axis=1).tolist()}
        (self.path/'observation.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'service_seconds':elapsed,'estimate':estimate,'truth':truth,'errors':result['estimated_handle_error_m']}),flush=True)
        return result


def trial(seed,scenario,load,method,output):
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    original=suite.PotWithHandlesObject
    class Environment(runner.TwoArmLiftEnv):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);self.raw.has_offscreen_renderer=True
    def scaled(*a,**k):k['density']=1000*load;return original(*a,**k)
    config=json.loads((REPO/'outputs/p1-load-transfer-20261109/protocol.json').read_text())['test_config']
    hook=Observer(output,method=='retarget') if method!='fixed' else None
    with ExitStack() as stack:
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',scaled))
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**config),
            selector=lambda p:{'candidate':'R_global'},account_selector_latency=True,regrasp_hook=hook)
    row.update(method=method,load='native' if load==1 else 'density150',window='actual')
    output.mkdir(parents=True,exist_ok=True)
    (output/'trial.json').write_text(json.dumps(row)+'\n')
    print(json.dumps({k:row.get(k) for k in ('seed','scenario','method','success','task_complete','failure_reason','error','protection','total_post_fault_seconds','inference_wait_seconds')}),flush=True)
    return row


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=60);p.add_argument('--scenario',default='left_loss');p.add_argument('--load',type=float,default=1.5);p.add_argument('--method',choices=('fixed','refresh','retarget'),default='retarget');p.add_argument('--name',required=True);a=p.parse_args()
    output=ROOT/'engineering'/a.name
    assert not output.exists()
    trial(a.seed,a.scenario,a.load,a.method,output)
