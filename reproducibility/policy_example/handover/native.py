"""Passive native handover contacts and initial-wait contact-force diagnostics."""
import numpy as np
import time


class NativeRecorder:
    def __init__(self,env,oracle,table_contact):
        self.env=env;self.oracle=oracle;self.table_contact=table_contact
        self.enabled=False;self.initial_wait=False;self.stage='';self.records=[];self.overhead_seconds=0.
        self.original=env.sim.step
        self.names=[env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom)]
        self.object_geoms={env.sim.model.geom_name2id(n) for n in env.raw.hammer.contact_geoms}
        env.sim.step=self.step

    def step(self,*a,**kw):
        result=self.original(*a,**kw)
        if not self.enabled:return result
        started=time.perf_counter()
        env=self.env;data=env.sim.data;truth=self.oracle.evaluate(env)
        penetration=force=0.
        for i,c in enumerate(data.contact[:data.ncon]):
            x,y=int(c.geom1),int(c.geom2);nx,ny=self.names[x],self.names[y]
            if nx.startswith('robot0') and ny.startswith('robot1') or nx.startswith('robot1') and ny.startswith('robot0'):
                penetration=max(penetration,-float(c.dist))
            if self.initial_wait and (x in self.object_geoms or y in self.object_geoms):
                import mujoco
                wrench=np.zeros(6);mujoco.mj_contactForce(env.sim.model._model,data._data,i,wrench)
                force+=float(np.linalg.norm(wrench[:3]))
        self.records.append((float(data.time),self.stage,truth['giver_holds'],truth['receiver_holds'],
            truth['receiver_supports'],self.table_contact(env),penetration,self.initial_wait,force))
        self.overhead_seconds+=time.perf_counter()-started
        return result

    def save(self,path):
        keys=('time','stage','giver_holds','receiver_holds','receiver_supports','table','penetration',
              'initial_wait','object_contact_force_norm_N')
        kinds=(float,str,bool,bool,bool,bool,float,bool,float)
        arrays={k:np.asarray([r[i] for r in self.records],dtype=d) for i,(k,d) in enumerate(zip(keys,kinds))}
        np.savez_compressed(path,**arrays)
        return len(self.records)
