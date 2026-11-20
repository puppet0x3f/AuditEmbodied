"""Bounded physical interventions on initial holding and registered geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'frozen-source'))
from auditembodied import l4_lift as runner
from auditembodied.recovery_skills import actual_eef_rotations
import geometry as cad
from observation import Observer,render
from native import NativeRecorder
from closed_loop import recover

CONFIG=json.loads((HERE/'config.json').read_text())


def one(job):
    seed,scenario,method,steps,hold,shape,phase=job
    case=f'{seed}-{scenario}-{shape}-{steps:02d}-{hold}-{method}'
    folder=HERE/phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(case+'.json')
    if path.exists():
        raise FileExistsError(path)
    cad.configure(shape)
    cad.fit(json.loads((HERE/'warmup-port.json').read_text()))
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    factory=suite.PotWithHandlesObject
    observer=Observer(True);recorders=[];environments=[]

    class Environment(runner.TwoArmLiftEnv):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.raw.has_offscreen_renderer=True
            self.hold_reference=None
            environments.append(self)

        def reset(self):
            super().reset()
            before=runner.snapshot(self);render(self)
            assert runner.snapshot(self)==before
            recorders.append(NativeRecorder(self))
            body=self.raw.pot_body_id
            self.registered_mass=float(self.sim.model.body_mass[body])

        def action(self,targets,grippers=(1.,1.)):
            if hold!='measured_position' or getattr(self,'record_stage','')!='inference_wait':
                return super().action(targets,grippers)
            if self.hold_reference is None:
                self.hold_reference={
                    'positions':np.stack([self._eef_pos(i) for i in range(2)]).tolist(),
                    'registered_positions':np.asarray(targets).tolist(),
                    'registered_rotations':self.rotation_targets.copy().tolist(),
                    'gripper_commands':list(grippers)}
            return super().action(np.asarray(self.hold_reference['positions']),grippers)

    def physical_object(*a,**kw):
        if shape=='pot':
            kw['density']=1500.
        else:
            hx,hy,hz=.09,.055,.04;t=.01;length=.075;width=.10;r=.01
            volume=4*hx*hy*t+8*t*hz*(hx+hy)+2*(8*(width/2+r)*r*r+16*r*(length/2)*r)
            kw.update(body_half_size=(hx,hy,hz),handle_length=length,handle_width=width,
                      handle_radius=r,density=1.818/volume)
        return factory(*a,**kw)

    started=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',physical_object))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**CONFIG),
            selector=lambda port:{'candidate':'R_global'},fixed_selector_wait_steps=steps,
            regrasp_hook=observer,global_executor=recover if method=='closed_loop' else None)
    env=environments[0]
    row.update(method=method,load='mass1818',window=f'{steps*50}ms',wait_steps=steps,
               hold_mode=hold,geometry=shape,cohort=phase,wall_seconds=time.perf_counter()-started,
               holding_input=env.hold_reference,registered_mass_kg=env.registered_mass)
    native_path=folder/(case+'-native.npz')
    row['native_samples']=recorders[0].save(native_path)
    row['native_file']=native_path.name
    goal=bool(row.get('task_complete',False))
    protected=bool(not row.get('protection_violation',True) and row.get('table_cycle_verified',False))
    row['audit']={'label':'error' if row.get('error') else 'A' if goal and protected else 'B' if goal else 'C' if protected else 'D',
                  'protected_completion':bool(goal and protected)}
    row['uniform_audit']={'V_path':True,'C':goal,'P':protected,
        'I':bool(row.get('table_cycle_verified',False)),'Y':bool(goal and protected),
        'label':row['audit']['label']}
    path.write_text(json.dumps(row,separators=(',',':'))+'\n')
    status={'case':case,'label':row['audit']['label'],'complete':goal,
            'seconds':row.get('total_post_fault_seconds'),'error':row.get('error'),
            'nominal_ready':row.get('nominal_ready'),'fault_valid':row.get('fault_class_valid'),
            'mass':row['registered_mass_kg'],'wall':row['wall_seconds']}
    print(json.dumps(status),flush=True)
    if phase.startswith('development'):
        import imageio.v3 as iio
        for camera,rgb in zip(cad.CAMERAS,observer.images):
            iio.imwrite(folder/(case+'-'+camera+'.png'),rgb)
    return status


def jobs(phase):
    protocol=json.loads((HERE/'protocol.json').read_text())
    seeds=protocol['development_seeds'] if phase=='development' else protocol['formal_seeds']
    if phase=='prefix-check':
        return [(69000,'left_loss','closed_loop',1,'registered','tray',phase)]
    result=[]
    for i,seed in enumerate(seeds):
        scenarios=('left_loss','right_loss','sham')
        for j,scenario in enumerate(scenarios[i%3:]+scenarios[:i%3]):
            for steps in ((1,20) if i%2==0 else (20,1)):
                methods=('retarget','closed_loop') if (i+j+steps)%2 else ('closed_loop','retarget')
                for method in methods:
                    result.append((seed,scenario,method,steps,'measured_position','tray',phase))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('development','prefix-check','formal'),required=True)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args()
    if a.phase=='formal':
        freeze=json.loads((HERE/'formal-freeze.json').read_text())
        for name,digest in freeze['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    planned=jobs(a.phase)
    assert len(planned)==(240 if a.phase=='formal' else 1 if a.phase=='prefix-check' else 12)
    assert not (HERE/a.phase).exists(),a.phase
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        statuses=list(pool.map(one,planned))
    (HERE/(a.phase+'-completion.json')).write_text(json.dumps(statuses,indent=2)+'\n')
    print(json.dumps({'phase':a.phase,'episodes':len(statuses),'errors':sum(bool(s['error']) for s in statuses)}),flush=True)


if __name__=='__main__':main()
