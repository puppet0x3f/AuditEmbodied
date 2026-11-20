"""Passive 500 Hz recording; no field from this module reaches the policy."""
import numpy as np


class NativeRecorder:
    def __init__(self,env):
        self.env=env;self.records=[];self.original=env.sim.step
        model=env.sim.model
        names=[model.geom_id2name(i) for i in range(model.ngeom)]
        self.names=names
        self.pot={model.geom_name2id(n) for n in env.raw.pot.contact_geoms}
        self.table={i for i,n in enumerate(names) if n.startswith('table')}
        handles=[env.raw.pot.handle0_geoms,env.raw.pot.handle1_geoms]
        self.handles=[{model.geom_name2id(n) for n in hs} for hs in handles]
        self.pads=[]
        for robot in env.raw.robots:
            gripper=robot.gripper['right']
            self.pads.append([{model.geom_name2id(n) for n in gripper.important_geoms[k]}
                              for k in ('left_fingerpad','right_fingerpad')])
        env.sim.step=self.step

    def step(self,*args,**kwargs):
        answer=self.original(*args,**kwargs);env=self.env
        if getattr(env,'record_phase','') not in ('recovery','final_hold'):return answer
        data=env.sim.data;contact_pairs=[];table=False;penetration=0.;hits=np.zeros((2,2),dtype=bool)
        for ci,c in enumerate(data.contact[:data.ncon]):
            a,b=int(c.geom1),int(c.geom2)
            if (a in self.pot and b in self.table) or (b in self.pot and a in self.table):table=True
            for arm in range(2):
                for finger in range(2):
                    if (a in self.handles[arm] and b in self.pads[arm][finger]) or (b in self.handles[arm] and a in self.pads[arm][finger]):hits[arm,finger]=True
            na,nb=self.names[a],self.names[b]
            if na.startswith('robot0') and nb.startswith('robot1') or na.startswith('robot1') and nb.startswith('robot0'):
                penetration=max(penetration,-float(c.dist))
            if a in self.pot or b in self.pot:contact_pairs.append(ci)
        holds=hits.all(1);unsupported=not(table or holds.any());force=0.
        if unsupported:
            import mujoco
            for ci in contact_pairs:
                wrench=np.zeros(6);mujoco.mj_contactForce(env.sim.model._model,data._data,ci,wrench)
                force+=float(np.linalg.norm(wrench[:3]))
        rot=np.asarray(data.body_xmat[env.raw.pot_body_id]).reshape(3,3)
        self.records.append((float(data.time),env.record_stage,*holds,table,
            *np.asarray(data.site_xpos[env.raw.pot_center_id]),
            float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1)))),penetration,
            unsupported,force))
        return answer

    def save(self,path):
        names=('time','stage','left','right','table','x','y','z','tilt_deg',
               'interarm_penetration_m','force_measured','object_contact_force_norm_N')
        kinds=(float,str,bool,bool,bool,float,float,float,float,float,bool,float)
        arrays={n:np.asarray([r[i] for r in self.records],dtype=k) for i,(n,k) in enumerate(zip(names,kinds))}
        np.savez_compressed(path,**arrays)
        return len(self.records)
