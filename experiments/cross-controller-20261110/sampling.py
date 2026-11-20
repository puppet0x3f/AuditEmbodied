"""Passive native-tick records, kept outside the policy observation port."""
import time
import numpy as np

FIELDS=('time','phase','stage','holds0','holds1','table_contact','center_x','center_y','center_z',
        'tilt_deg','clearance','interarm_penetration_m','object_contact_force_norm_sum_N',
        'table_contact_force_norm_sum_N','object_contact_count')
PHASES=('initialization','fault','recovery','final_hold')


class NativeRecorder:
    def __init__(self,env,evaluate):
        import mujoco
        self.env=env;self.evaluate=evaluate;self.mujoco=mujoco;self.records=[];self.stages=[];self.seconds=0.
        self.hz=1/float(env.sim.model.opt.timestep)
        self.step=env.sim.step
        self.names=[env.sim.model.geom_id2name(i) or '' for i in range(env.sim.model.ngeom)]
        self.object_ids={env.sim.model.geom_name2id(n) for n in env.raw.pot.contact_geoms}
        self.force=np.zeros(6)
        env.sim.step=self.sampled_step

    def sampled_step(self,*args,**kwargs):
        result=self.step(*args,**kwargs)
        context=getattr(self.env,'audit_context',None)
        if context is None:return result
        started=time.perf_counter();env=self.env
        phase,stage,_=context
        if stage not in self.stages:self.stages.append(stage)
        truth=self.evaluate(env);deep=0.;norms=0.;table_norms=0.;count=0
        for index,c in enumerate(env.sim.data.contact[:env.sim.data.ncon]):
            a,b=(int(x) for x in (c.geom1,c.geom2));na,nb=self.names[a],self.names[b]
            if na.startswith('robot0') and nb.startswith('robot1') or na.startswith('robot1') and nb.startswith('robot0'):
                deep=max(deep,-float(c.dist))
            if a in self.object_ids or b in self.object_ids:
                self.mujoco.mj_contactForce(env.sim.model._model,env.sim.data._data,index,self.force)
                norm=float(np.linalg.norm(self.force[:3]));norms+=norm;count+=1
                if na.startswith('table') or nb.startswith('table'):table_norms+=norm
        self.records.append([float(env.sim.data.time),PHASES.index(phase),self.stages.index(stage),
            *truth['holds'],truth['table_contact'],*truth['pot_center'],truth['tilt_deg'],truth['clearance'],deep,norms,table_norms,count])
        self.seconds+=time.perf_counter()-started
        return result

    def save(self,path):
        np.savez_compressed(path,values=np.asarray(self.records),fields=np.asarray(FIELDS),
            stages=np.asarray(self.stages),phases=np.asarray(PHASES))
        return {'samples':len(self.records),'hz':self.hz,
            'wall_seconds':self.seconds,'stages':self.stages,'path':str(path)}
