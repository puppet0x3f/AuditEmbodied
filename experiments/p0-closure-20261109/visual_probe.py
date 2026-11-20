"""Same-prefix camera diagnosis with unchanged source, models and physics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'outputs/handover-audit-20261109'
sys.path.insert(0,str(STUDY/'frozen-source'))
from auditembodied import l3_matrix
from auditembodied.l0 import CAMERAS
from auditembodied.l3_visual_test import DecisionObserver
from auditembodied.visual_recovery import MultiSupportVerifier


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=60)
    parser.add_argument('--warm',type=int,choices=[0,1],required=True);parser.add_argument('--tag',default='');args=parser.parse_args()
    folder=Path(__file__).resolve().parent/'visual-probes';folder.mkdir(exist_ok=True)
    name=f'{args.seed}-warm{args.warm}'+('-'+args.tag if args.tag else '')
    result={}
    class Observer(DecisionObserver):
        def after_reset(self,env):
            super().after_reset(env)
            if args.warm:
                for camera in CAMERAS:env.sim.render(camera_name=camera,width=64,height=64)
    observer=Observer('image_robot')
    class Probe:
        def run(self,env,expert,episode,capture,*unused,observe):
            before={k:np.asarray(getattr(env.sim.data,k)).copy() for k in ('qpos','qvel','ctrl','act','cam_xpos','cam_xmat')}
            packet=observe();rgb=packet['rgb'];robot=packet['robot']
            model=MultiSupportVerifier.load(STUDY/'models/visual-support.npz')
            camera_ids=[env.sim.model.camera_name2id(c) for c in CAMERAS]
            from OpenGL import GL
            result.update(seed=args.seed,warm=args.warm,observed_time=float(env.sim.data.time),
                gl_renderer=GL.glGetString(GL.GL_RENDERER).decode(),gl_vendor=GL.glGetString(GL.GL_VENDOR).decode(),
                probabilities=model.predict_proba(rgb[None],robot[None])[0].tolist(),
                thresholds=model.thresholds.tolist(),
                model_camera_pos=np.asarray(env.sim.model.cam_pos)[camera_ids].tolist(),
                model_camera_quat=np.asarray(env.sim.model.cam_quat)[camera_ids].tolist(),
                model_camera_fovy=np.asarray(env.sim.model.cam_fovy)[camera_ids].tolist(),
                geomgroup=env.sim._render_context_offscreen.vopt.geomgroup.tolist(),
                image_convention=__import__('robosuite').macros.IMAGE_CONVENTION)
            np.savez_compressed(folder/(name+'.npz'),rgb=rgb,robot=robot,**before)
            return {'sequence':['continue'],'skill':'prefix_observation_probe','used_global_recovery':False}
    protocol=json.loads((STUDY/'protocol.json').read_text())
    row=l3_matrix.run_trial(args.seed,'object_sham','visual',l3_matrix.MatrixConfig(**protocol['physical_config']),observer=observer,recovery_policy=Probe())
    assert result,'Failed before observing prefix'
    result.update(initialization_state=row['initialization_state'],prefix_state=row['prefix_state'],
        purpose='Diagnostic original prefix capture followed by a fixed continuation; no policy evaluation score is claimed.')
    (folder/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('initialization_state','prefix_state')}),flush=True)


if __name__=='__main__':main()
