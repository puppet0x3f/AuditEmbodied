"""Calibrated RGB silhouette fitting of the registered pot CAD; NumPy only.

No environment, object state, contacts, seed, load or fault inputs are accepted.
The known dimensions and colored surfaces are an explicit task-specific prior.
"""
import itertools
import numpy as np

SIZE=320
CAMERAS=('agentview','frontview','sideview','birdview')
DIRECTIONS=np.stack([np.cos(np.arange(32)*2*np.pi/32),np.sin(np.arange(32)*2*np.pi/32)],axis=1)


def corners(center,half):
    return np.asarray(center)+np.asarray(list(itertools.product((-1.,1.),repeat=3)))*np.asarray(half)


def configure(shape='pot'):
    """Bind registered CAD constants; no physical state enters this binding."""
    global BODY_HALF,HANDLE_Z,BODY,HANDLES,TEMPLATES
    BODY_HALF=np.array((.07,.07,.07) if shape=='pot' else (.09,.055,.04))
    length,width,radius=(.09,.09,.01) if shape=='pot' else (.075,.10,.01)
    HANDLE_Z=float(BODY_HALF[2]-radius)
    BODY=corners((0.,0.,0.),BODY_HALF)
    HANDLES=[]
    for sign in (1.,-1.):
        HANDLES.append(np.concatenate([
            corners((0.,sign*(BODY_HALF[1]+length),HANDLE_Z),(width/2+radius,radius,radius))]+
            [corners((x,sign*(BODY_HALF[1]+length/2),HANDLE_Z),(radius,length/2,radius))
             for x in (-width/2,width/2)]))
    TEMPLATES=(BODY,*HANDLES)


configure()


def exp_rotation(v):
    v=np.asarray(v);angle=np.linalg.norm(v)
    if angle<1e-14:return np.eye(3)
    x,y,z=v/angle;k=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def mask_features(rgb):
    x=np.asarray(rgb,dtype=float)
    results=[]
    for channel in range(3):
        other=np.max(x[:,:,[j for j in range(3) if j!=channel]],axis=2)
        mask=(x[:,:,channel]>35)&(x[:,:,channel]>1.4*other+5)
        y,z=np.nonzero(mask)
        points=np.stack([z,y],axis=1).astype(float)
        # Insufficient visible color contributes no fictitious observation.
        results.append({'pixels':int(len(points)),
            'support':(points@DIRECTIONS.T).max(0).tolist() if len(points)>=12 else None,
            'center':points.mean(0).tolist() if len(points)>=12 else None})
    return results


def project(points,matrix):
    p=np.c_[points,np.ones(len(points))]@np.asarray(matrix).T
    return p[:,:2]/p[:,2:3]


def prediction(v,rotation,views):
    rot=exp_rotation(v[3:])@rotation
    values=[]
    for view in views:
        for template,observed in zip(TEMPLATES,view['colors']):
            if observed['support'] is None:continue
            xy=project(template@rot.T+v[:3],view['projection'])
            values.extend((xy@DIRECTIONS.T).max(0))
    return np.asarray(values)


def fit(port):
    """Fit one SE(3) to RGB convex silhouettes, with robust pixel residuals."""
    assert set(port)=={'reference_rotation','reference_center','reference_targets','reference_rotation_targets','views'}
    rotation=np.asarray(port['reference_rotation'])
    views=port['views']
    target=np.concatenate([c['support'] for view in views for c in view['colors'] if c['support'] is not None])
    if len(target)<96:
        return {'valid':False,'reason':'insufficient_visible_color'}
    v=np.r_[port['reference_center'],np.zeros(3)].astype(float)
    lam=1.;last=float('inf');steps=0
    def residual(x):return prediction(x,rotation,views)-target
    def objective(r):return np.sum(np.where(abs(r)<3.,.5*r*r,3.*abs(r)-4.5))
    for steps in range(60):
        r=residual(v);score=objective(r)
        weights=np.minimum(1.,3./np.maximum(abs(r),1e-12))
        delta=np.array([1e-5]*3+[1e-4]*3)
        jac=np.stack([(residual(v+np.eye(6)[j]*delta[j])-r)/delta[j] for j in range(6)],axis=1)
        h=jac.T@(weights[:,None]*jac)
        gradient=jac.T@(weights*r)
        step=np.linalg.solve(h+lam*np.diag(np.maximum(np.diag(h),1.)),gradient)
        step[:3]=np.clip(step[:3],-.04,.04);step[3:]=np.clip(step[3:],-.2,.2)
        proposal=v-step;new=objective(residual(proposal))
        if new<score:
            v=proposal;lam=max(lam/3,1e-7)
            if abs(score-new)<1e-5 or np.linalg.norm(step)<1e-7:break
        else:lam=min(lam*5,1e9)
        last=score
    r=residual(v);rot=exp_rotation(v[3:])@rotation
    delta=rot@rotation.T
    targets=np.asarray(port['reference_targets'])
    moved=v[:3]+(targets-np.asarray(port['reference_center']))@delta.T
    return {'valid':True,'center':v[:3].tolist(),'rotation':rot.tolist(),
        'delta_rotation':delta.tolist(),'targets':moved.tolist(),
        'rotation_targets':np.einsum('ij,ajk->aik',delta,np.asarray(port['reference_rotation_targets'])).tolist(),
        'optimization_vector':v.tolist(),'iterations':steps+1,
        'mean_absolute_pixel_residual':float(np.mean(abs(r))),
        'objective':float(objective(r))}
