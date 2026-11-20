"""RGB landmark visual servoing, independent of CAD fitting and the old executor.

Calibrated views triangulate three observed color centroids. A rigid landmark
registration transfers measured robot reference grasps; there is no object CAD,
object-state access, learned outcome gate, or call into the old recovery code.
"""
import hashlib
import time
import numpy as np

CAMERAS=('agentview','frontview','sideview','birdview');SIZE=320


def features(rgb):
    image=np.asarray(rgb,dtype=float);result=[]
    for channel in range(3):
        others=np.max(image[:,:,[k for k in range(3) if k!=channel]],axis=2)
        y,x=np.nonzero((image[:,:,channel]>35)&(image[:,:,channel]>1.4*others+5))
        result.append({'pixels':len(x),'center':[float(x.mean()),float(y.mean())] if len(x)>=12 else None})
    return result


def projection(env,camera):
    i=env.sim.model.camera_name2id(camera);f=SIZE/2/np.tan(np.deg2rad(env.sim.model.cam_fovy[i])/2)
    k=np.array([[f,0,SIZE/2],[0,f,SIZE/2],[0,0,1.]])
    rot=np.asarray(env.sim.data.cam_xmat[i]).reshape(3,3)@np.diag([1.,-1.,-1.])
    xyz=np.asarray(env.sim.data.cam_xpos[i]);flip=np.array([[1.,0.,0.],[0.,-1.,SIZE-1.],[0.,0.,1.]])
    return flip@k@np.c_[rot.T,-rot.T@xyz]


def triangulate(views):
    points=[]
    for channel in range(3):
        equations=[]
        for view in views:
            point=view['colors'][channel]['center']
            if point is None:continue
            p=np.asarray(view['projection']);x,y=point
            equations.extend([x*p[2]-p[0],y*p[2]-p[1]])
        if len(equations)<4:return None
        a=np.asarray(equations);_,_,v=np.linalg.svd(a,full_matrices=False)
        points.append(v[-1,:3]/v[-1,3])
    return np.asarray(points)


def estimate(port):
    reference=triangulate(port['reference_views']);current=triangulate(port['views'])
    if reference is None or current is None:return {'valid':False,'reason':'insufficient_visible_landmarks'}
    # The two handles define the grasp line. Align that line by the minimum
    # rotation rather than inferring rotation about it from a view-dependent
    # body-color centroid, which is especially sensitive to occlusion.
    a=reference[1]-reference[2];b=current[1]-current[2]
    a/=np.linalg.norm(a);b/=np.linalg.norm(b)
    axis=np.cross(a,b);cosine=float(np.dot(a,b))
    if cosine<-.99:return {'valid':False,'reason':'ambiguous_antipodal_handle_line'}
    x,y,z=axis;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    rotation=np.eye(3)+skew+skew@skew/(1+cosine)
    origin=reference[1:].mean(0);center=current[1:].mean(0)
    targets=center+(np.asarray(port['reference_targets'])-origin)@rotation.T
    wrists=np.einsum('ij,ajk->aik',rotation,np.asarray(port['reference_rotations']))
    return {'valid':True,'center':current[0].tolist(),'rotation':rotation.tolist(),
        'targets':targets.tolist(),'rotation_targets':wrists.tolist(),
        'landmarks':current.tolist(),'reference_landmarks':reference.tolist(),
        'rigid_residual_mm':float(abs(np.linalg.norm(reference[1]-reference[2])-np.linalg.norm(current[1]-current[2]))*1000)}


class VisualObserver:
    def __init__(self):self.images=[];self.reference_views=None

    def acquire(self,env):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.images=[env.sim.render(camera_name=c,width=SIZE,height=SIZE)[::convention].copy() for c in CAMERAS]
        return [{'camera':c,'projection':projection(env,c).tolist(),'colors':features(rgb)} for c,rgb in zip(CAMERAS,self.images)]

    def reference_scene(self,env):
        self.reference_views=self.acquire(env);self.reference_images=[x.copy() for x in self.images]

    def __call__(self,env,targets,rotations):
        assert self.reference_views is not None
        port={'reference_views':self.reference_views,'reference_targets':np.asarray(targets).tolist(),
              'reference_rotations':np.asarray(rotations).tolist(),'views':self.acquire(env)}
        return {'input':port,'estimate':estimate(port),
                'rgb_sha256':[hashlib.sha256(x.tobytes()).hexdigest() for x in self.images]}


def rotation_step(first,second,fraction):
    # Project an interpolated rotation back onto SO(3).
    u,_,v=np.linalg.svd((1-fraction)*np.asarray(first)+fraction*np.asarray(second))
    result=u@v
    if np.linalg.det(result)<0:u[:,-1]*=-1;result=u@v
    return result


def recover(p):
    log={'policy':'RGB landmark visual servo; triangulation and rigid registration without CAD',
         'observations':[],'pickup_attempts':[],'service_steps':0}
    reference=np.asarray(p['table_targets']);wrists=np.asarray(p['registered_rotations'])
    def robot():return np.asarray(p['robot']()['positions'])
    def observe(grippers):
        at=p['time']();tick=time.perf_counter();d=p['observe']();elapsed=time.perf_counter()-tick
        steps=max(1,int(np.ceil(elapsed/.05)))
        p['phase']('regrasp_observation_wait',robot(),steps,grippers)
        d.update(observed_time=at,service_wall_seconds=elapsed,service_wait_steps=steps)
        log['observations'].append(d);log['service_steps']+=steps
        return d['estimate']
    def align(desired,steps,grippers):
        before=np.asarray(p['robot']()['rotations']);at=robot()
        for j in range(steps):
            p['set_rotations']([rotation_step(before[i],desired[i],(j+1)/steps) for i in range(2)])
            p['phase']('global_align_wrist',at,1,grippers)
    grips=tuple(p['gripper_commands']);p['set_rotations'](p['robot']()['rotations'])
    # A table-supported release is the task interface. This controller uses a
    # fixed touchdown stroke and closes subsequent approach in landmark space.
    lower=robot();lower[:,2]-=.18
    p['move']('global_lower_to_table',lower,60,grips)
    p['phase']('global_table_loaded',lower,12,grips)
    p['phase']('global_release',robot(),14,(-1.,-1.))
    p['phase']('global_table_settle',robot(),12,(-1.,-1.))
    clear=robot();clear[:,2]+=.14;p['move']('global_clear',clear,28,(-1.,-1.))
    for attempt in range(2):
        entry={'index':attempt,'start_time':p['time']()};log['pickup_attempts'].append(entry)
        observed=observe((-1.,-1.))
        if not observed['valid']:entry['stop']='insufficient_landmarks';break
        align(observed['rotation_targets'],24,(-1.,-1.))
        # Reobserve each approach segment. Motion lengths and all rendering
        # services are charged, including any second pickup.
        for offset,stage,steps in [(.09,'global_hover',24),(.04,'global_descend',16),(0.,'global_descend',16)]:
            observed=observe((-1.,-1.))
            if not observed['valid']:break
            targets=np.asarray(observed['targets'])+np.asarray(observed['rotation'])[:,2]*offset
            p['set_rotations'](observed['rotation_targets'])
            p['move'](stage,targets,steps,(-1.,-1.))
        if not observed['valid']:entry['stop']='insufficient_approach_landmarks';break
        p['phase']('global_close',targets,24)
        widths=np.mean(np.abs(p['robot']()['fingers']),axis=1)
        entry.update(close_time=p['time'](),encoder_loaded=((widths>.004)&(widths<.028)).tolist())
        if not all(entry['encoder_loaded']) and attempt==0:
            p['phase']('global_release',robot(),14,(-1.,-1.))
            p['phase']('global_table_settle',robot(),10,(-1.,-1.))
            clear=robot();clear[:,2]+=.14;p['move']('global_clear',clear,28,(-1.,-1.))
            entry.update(end_time=p['time'](),stop='encoder_retry');continue
        start=robot();rotation=np.asarray(p['robot']()['rotations'])
        # Coordinate leveling and recentering on the table, then lift. This is
        # separate from the old executor's deferred in-air leveling schedule.
        for j in range(90):
            q=(j+1)/90
            p['set_rotations']([rotation_step(rotation[i],wrists[i],q) for i in range(2)])
            p['phase']('global_recenter_on_table',start*(1-q)+reference*q,1)
        p['move']('global_lift',reference+[0.,0.,.17],80)
        p['phase']('global_lift_settle',reference+[0.,0.,.17],20)
        entry.update(end_time=p['time'](),stop='lifted');break
    return log
