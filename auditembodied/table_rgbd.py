"""RGB-D registration of a measured table-grasp template; no object-state reads."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def acquire(env, size=320):
    from robosuite.utils.camera_utils import get_real_depth_map,get_camera_transform_matrix
    points=[];colors=[]
    for camera in ('frontview','agentview'):
        rgb,raw_depth=env.sim.render(camera_name=camera,height=size,width=size,depth=True)
        rgb=rgb[::-1].copy();depth=get_real_depth_map(env.sim,raw_depth[::-1])
        y,x=np.indices(depth.shape)
        v=np.stack((x*depth,y*depth,depth,np.ones_like(depth)),-1)
        world=v@np.linalg.inv(get_camera_transform_matrix(env.sim,camera,size,size)).T
        points.append(world[...,:3].reshape(-1,3));colors.append(rgb.reshape(-1,3))
    return np.concatenate(points),np.concatenate(colors)


def colored_cloud(points,colors,center,radius=.45):
    # The three saturated template colours are visible sensor measurements.
    # No MuJoCo segmentation, material IDs, object sites or contact arrays.
    c=np.asarray(colors,dtype=float);order=np.argsort(c,axis=1)
    label=order[:,-1]
    strong=(c.max(1)>28)&(c.max(1)>1.55*np.take_along_axis(c,order[:,-2,None],axis=1)[:,0]+8)
    near=np.linalg.norm(points[:,:2]-np.asarray(center)[:2],axis=1)<radius
    keep=strong&near&np.isfinite(points).all(1)
    points,label=points[keep],label[keep]
    # Balance visible surface colours so a large body face does not drown out
    # the small handles in the rigid fit. Deterministic spatial subsampling.
    indices=[]
    for i in range(3):
        part=np.flatnonzero(label==i)
        indices.extend(part[np.linspace(0,len(part)-1,min(800,len(part)),dtype=int)])
    return points[indices],label[indices]


def build_template(env,path,grasps,rotations):
    points,colors=acquire(env)
    center=np.asarray(grasps).mean(0)
    cloud,labels=colored_cloud(points,colors,center,.32)
    # Estimate the tabletop from the surrounding depth surface, not a site.
    near=np.linalg.norm(points[:,:2]-center[:2],axis=1)<.45
    z=points[near,2];z=z[(z<center[2]-.025)&(z>center[2]-.3)]
    bins,edges=np.histogram(z,bins=150)
    at=int(np.argmax(bins));table_z=float(np.median(z[(z>=edges[at])&(z<=edges[at+1])]))
    good=cloud[:,2]>table_z+.006;cloud,labels=cloud[good],labels[good]
    if len(cloud)<50 or any(np.sum(labels==i)<3 for i in range(3)):
        raise ValueError('insufficient visible RGB-D template colours')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,points=cloud,labels=labels,grasps=grasps,rotations=rotations,table_z=table_z)
    return {'path':str(path.resolve()),'points':len(cloud),'counts':np.bincount(labels,minlength=3).tolist(),'table_z':table_z}


def _rigid(a,b):
    ca=a.mean(0);cb=b.mean(0);u,_,v=np.linalg.svd((a-ca).T@(b-cb))
    r=v.T@u.T
    if np.linalg.det(r)<0:v[-1]*=-1;r=v.T@u.T
    return r,cb-r@ca


def register(env,path,keeper,positions,rotations,prior=None):
    with np.load(path) as d:
        ref=d['points'];labels=d['labels'];grasps=d['grasps'];rref=d['rotations'];table_z=float(d['table_z'])
    points,colors=acquire(env)
    obs,classes=colored_cloud(points,colors,np.asarray(positions).mean(0),.55)
    good=(obs[:,2]>table_z+.004)&(obs[:,2]<np.max(positions[:,2])+.16)
    obs,classes=obs[good],classes[good]
    counts=np.bincount(classes,minlength=3)
    visible=[i for i in range(3) if counts[i]>=3]
    if len(obs)<40 or 0 not in visible or len(visible)<2:
        return {'valid':False,'reason':'coloured_surface_not_visible','counts':counts.tolist()}
    trees={i:cKDTree(obs[classes==i]) for i in visible}
    arrays={i:obs[classes==i] for i in visible}
    ref_centers=np.stack([np.median(ref[labels==i],0) for i in visible])
    centers=np.stack([np.median(arrays[i],0) for i in visible])
    initial=[_rigid(ref_centers,centers)]
    prior_rotation=None
    if len(visible)<3 and prior and prior.get('valid') and 'keeper_position' in prior:
        delta=np.asarray(rotations)[keeper]@np.asarray(prior['keeper_rotation']).T
        offset=np.asarray(positions)[keeper]-delta@np.asarray(prior['keeper_position'])
        prior_rotation=delta@np.asarray(prior['rotation'])
        initial.insert(0,(prior_rotation,delta@np.asarray(prior['translation'])+offset))
    # The keeper's measured pose supplies an initialisation only. Registration
    # remains free to estimate object slip relative to that hand.
    base=np.asarray(rotations)[keeper]@rref[keeper].T
    for angle in (-24.,0.,24.):
        r=Rotation.from_rotvec([np.radians(angle),0.,0.]).as_matrix()@base
        initial.append((r,centers[0]-r@ref_centers[0]))
    best=None
    for r,t in initial:
        for _ in range(35):
            moved=ref@r.T+t;source=[];target=[]
            for i in visible:
                index=np.flatnonzero(labels==i);distance,nn=trees[i].query(moved[index])
                valid=distance<max(.012,float(np.quantile(distance,.78)))
                chosen=index[valid]
                if len(chosen):source.append(moved[chosen]);target.append(arrays[i][nn[valid]])
            if not source:break
            dr,dt=_rigid(np.concatenate(source),np.concatenate(target))
            r,t=dr@r,dr@t+dt
            if np.linalg.norm(dt)<1e-5 and np.linalg.norm(dr-np.eye(3))<1e-4:break
        moved=ref@r.T+t;distances=np.concatenate([trees[i].query(moved[labels==i])[0] for i in visible])
        score=float(np.mean(np.sort(distances)[:max(1,int(.8*len(distances)))]))
        selection_score=score+(.002*np.linalg.norm(r-prior_rotation) if prior_rotation is not None else 0.)
        if best is None or selection_score<best[0]:best=(selection_score,score,r.copy(),t.copy())
    _,score,r,t=best;goals=grasps@r.T+t
    transformed=ref@r.T+t
    # Table proximity is assessed from reconstructed visible geometry, not TCP.
    bottom=float(np.quantile(transformed[:,2],.02))
    return {'valid':bool(score<.018),'reason':'ok' if score<.018 else 'registration_residual',
        'residual_m':score,'counts':counts.tolist(),'positions':goals.tolist(),
        'rotations':np.einsum('ij,njk->nik',r,rref).tolist(),'rotation':r.tolist(),
        'translation':t.tolist(),'table_z':table_z,'surface_bottom_z':bottom,
        'table_gap_m':bottom-table_z,'tilt_deg':float(np.degrees(np.arccos(np.clip(r[2,2],-1,1)))),
        'visible_colours':visible,'keeper_position':np.asarray(positions)[keeper].tolist(),
        'keeper_rotation':np.asarray(rotations)[keeper].tolist()}
