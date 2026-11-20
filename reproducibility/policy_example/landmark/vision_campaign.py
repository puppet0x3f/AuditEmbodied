"""Bounded holding-component experiment and independent visual-control validation."""
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
    seed,scenario,steps,hold,phase=job
    independent='independent' in phase
    method='visual_servo' if independent else 'closed_loop'
    case=f'{seed}-{scenario}-{steps:02d}-{hold}-{method}'
    folder=HERE/phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(case+'.json')
    if path.exists():raise FileExistsError(path)
    cad.configure('pot')
    if not independent:cad.fit(json.loads((HERE/'warmup-port.json').read_text()))
    runner._import_robosuite(False)
    from robosuite.environments.manipulation import two_arm_lift as suite
    factory=suite.PotWithHandlesObject
    if independent:
        from visual_servo import VisualObserver, recover as visual_recover
        observer=VisualObserver();executor=visual_recover
    else:
        observer=Observer(True);executor=recover
    recorders=[];environments=[]

    class Environment(runner.TwoArmLiftEnv):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.raw.has_offscreen_renderer=True
            self.hold_reference=None;self.hold_diagnostics=[]
            environments.append(self)

        def reset(self):
            super().reset()
            before=runner.snapshot(self);render(self)
            assert runner.snapshot(self)==before
            if independent:observer.reference_scene(self)
            recorders.append(NativeRecorder(self))
            self.registered_mass=float(self.sim.model.body_mass[self.raw.pot_body_id])

        def action(self,targets,grippers=(1.,1.)):
            if getattr(self,'record_stage','')!='inference_wait':
                return super().action(targets,grippers)
            positions=np.stack([self._eef_pos(i) for i in range(2)])
            rotations=actual_eef_rotations(self)
            if self.hold_reference is None:
                self.hold_reference={'measured_positions':positions.tolist(),
                    'measured_rotations':rotations.tolist(),
                    'registered_positions':np.asarray(targets).tolist(),
                    'registered_rotations':self.rotation_targets.tolist(),
                    'gripper_commands':list(grippers)}
            use_position=hold in ('measured_position','measured_pose')
            use_rotation=hold in ('measured_rotation','measured_pose')
            command_position=np.asarray(self.hold_reference['measured_positions']) if use_position else np.asarray(targets)
            command_rotation=np.asarray(self.hold_reference['measured_rotations']) if use_rotation else self.rotation_targets.copy()
            angles=np.degrees(np.arccos(np.clip((np.einsum('aij,aij->a',command_rotation,rotations)-1)/2,-1,1)))
            self.hold_diagnostics.append({'time':float(self.sim.data.time),
                'position_error_mm':(np.linalg.norm(command_position-positions,axis=1)*1000).tolist(),
                'rotation_error_deg':angles.tolist()})
            previous=self.rotation_targets.copy();self.rotation_targets=command_rotation
            try:return super().action(command_position,grippers)
            finally:self.rotation_targets=previous

    def physical_object(*a,**kw):
        kw['density']=1500.
        return factory(*a,**kw)

    started=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(runner,'TwoArmLiftEnv',Environment))
        stack.enter_context(patch.object(suite,'PotWithHandlesObject',physical_object))
        row=runner.run_trial(seed,scenario,config=runner.LiftConfig(**CONFIG),
            selector=lambda port:{'candidate':'R_global'},fixed_selector_wait_steps=steps,
            regrasp_hook=observer,global_executor=executor)
    env=environments[0]
    row.update(method=method,load='mass1818',window=f'{steps*50}ms',wait_steps=steps,
        hold_mode=hold,geometry='pot',cohort=phase,wall_seconds=time.perf_counter()-started,
        holding_input=env.hold_reference,hold_tracking=env.hold_diagnostics,
        registered_mass_kg=env.registered_mass)
    native_path=folder/(case+'-native.npz')
    row['native_samples']=recorders[0].save(native_path);row['native_file']=native_path.name
    goal=bool(row.get('task_complete',False))
    protected=bool(not row.get('protection_violation',True) and row.get('table_cycle_verified',False))
    row['audit']={'label':'error' if row.get('error') else 'A' if goal and protected else 'B' if goal else 'C' if protected else 'D',
                  'protected_completion':bool(goal and protected)}
    row['uniform_audit']={'V_path':True,'C':goal,'P':protected,
        'I':bool(row.get('table_cycle_verified',False)),'Y':bool(goal and protected),'label':row['audit']['label']}
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
    section=protocol['independent' if 'independent' in phase else 'mechanism']
    seeds=section['development_seeds'] if phase.startswith('development') else section['formal_seeds']
    result=[]
    for i,seed in enumerate(seeds):
        scenarios=tuple(section['scenarios']);holds=tuple(section['hold_modes'])
        for j,scenario in enumerate(scenarios[i%3:]+scenarios[:i%3]):
            for steps in ((1,20) if i%2 else (20,1)):
                shift=(i+j+steps)%len(holds)
                for hold in holds[shift:]+holds[:shift]:result.append((seed,scenario,steps,hold,phase))
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('development-mechanism','mechanism','development-independent','development-independent-final','independent'),required=True)
    p.add_argument('--workers',type=int,default=20)
    a=p.parse_args()
    if not a.phase.startswith('development'):
        freeze=json.loads((HERE/(a.phase+'-freeze.json')).read_text())
        for name,digest in freeze['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    planned=jobs(a.phase)
    assert not (HERE/a.phase).exists(),a.phase
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        statuses=list(pool.map(one,planned))
    (HERE/(a.phase+'-completion.json')).write_text(json.dumps(statuses,indent=2)+'\n')
    print(json.dumps({'phase':a.phase,'episodes':len(statuses),'errors':sum(bool(s['error']) for s in statuses)}),flush=True)


if __name__=='__main__':main()
