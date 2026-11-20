"""RGB acquisition layer. Only its explicit numerical port reaches geometry.fit."""
import hashlib
import numpy as np
from geometry import CAMERAS,SIZE,mask_features,fit


def projection(env,camera):
    idx=env.sim.model.camera_name2id(camera)
    f=SIZE/2/np.tan(np.deg2rad(env.sim.model.cam_fovy[idx])/2)
    k=np.array([[f,0,SIZE/2],[0,f,SIZE/2],[0,0,1.]])
    rot=np.asarray(env.sim.data.cam_xmat[idx]).reshape(3,3)@np.diag([1.,-1.,-1.])
    xyz=np.asarray(env.sim.data.cam_xpos[idx])
    image_rows=np.array([[1.,0.,0.],[0.,-1.,SIZE-1.],[0.,0.,1.]])
    return image_rows@k@np.c_[rot.T,-rot.T@xyz]


def reference(targets,rotations):
    rz=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rot=np.asarray(rotations)[0]@np.diag([-1.,1.,-1.])@rz
    center=np.asarray(targets).mean(0)+[0.,0.,.0036]-rot@np.array([0.,0.,.06])
    return center,rot


def render(env):
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    return [env.sim.render(camera_name=c,width=SIZE,height=SIZE)[::convention].copy() for c in CAMERAS]


class Observer:
    def __init__(self,apply):self.apply=apply;self.images=[]
    def __call__(self,env,targets,rotations):
        center,rot=reference(targets,rotations)
        port={'reference_center':center.tolist(),'reference_rotation':rot.tolist(),
            'reference_targets':np.asarray(targets).tolist(),'reference_rotation_targets':np.asarray(rotations).tolist(),'views':[]}
        self.images=render(env)
        for camera,rgb in zip(CAMERAS,self.images):
            port['views'].append({'camera':camera,'projection':projection(env,camera).tolist(),'colors':mask_features(rgb)})
        estimate=fit(port)
        return {'apply':self.apply,'estimate':estimate,'input':port,
            'rgb_sha256':[hashlib.sha256(rgb.tobytes()).hexdigest() for rgb in self.images],
            'pickup_rotation':'unchanged registered wrist targets'}
