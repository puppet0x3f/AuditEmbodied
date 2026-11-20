from types import SimpleNamespace

import numpy as np

from auditembodied.recovery_skills import actual_eef_wrenches
from auditembodied.table_assist import run_table_assist,TableAssistConfig


def test_wrist_sensor_vectors_are_transformed_to_world():
    names={'f0':0,'t0':1,'f1':2,'t1':3}
    model=SimpleNamespace(sensor_name2id=names.__getitem__,sensor_objid=[0,0,1,1])
    r=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    data=SimpleNamespace(site_xmat=np.array([r,np.eye(3)]).reshape(2,9))
    robots=[SimpleNamespace(ee_force={'right':np.array([1.,0,0])},ee_torque={'right':np.array([0.,1,0])},
        gripper={'right':SimpleNamespace(important_sensors={'force_ee':f'f{i}','torque_ee':f't{i}'})}) for i in range(2)]
    env=SimpleNamespace(sim=SimpleNamespace(model=model,data=data),raw=SimpleNamespace(robots=robots))
    np.testing.assert_allclose(actual_eef_wrenches(env),[[0.,1,0,-1,0,0],[1.,0,0,0,1,0]])


def test_unconfirmed_support_times_out_with_keeper_closed(monkeypatch):
    import auditembodied.table_assist as module
    positions=np.array([[0.,0,1.1],[.3,0,1.1]])
    env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=0.)),rotation_targets=np.array([np.eye(3)]*2))
    env._eef_pos=lambda i:positions[i]
    monkeypatch.setattr(module,'actual_eef_rotations',lambda e:env.rotation_targets.copy())
    monkeypatch.setattr(module,'actual_gripper_positions',lambda e:np.array([[.01,-.01],[.04,-.04]]))
    monkeypatch.setattr(module,'actual_eef_wrenches',lambda e:np.array([[0.,0,10.,0,0,0],[0.,0,0,0,0,0]]))
    commands=[]
    def phase(name,targets,steps,grips):
        commands.append((name,grips));positions[:]=targets;env.sim.data.time+=steps*.05
    record={};cfg=TableAssistConfig()
    assert not run_table_assist(env,phase,lambda _:None,keeper=0,entry_positions=positions.copy(),
        position_targets=positions.copy(),table_targets=positions-[0,0,.15],
        registered_rotations=env.rotation_targets.copy(),reference_wrench=np.zeros((2,6)),
        lift_height=.16,lift_steps=50,record=record,config=cfg)
    assert record['reason']=='support_transfer_not_confirmed'
    assert len(commands)==cfg.retreat_steps+cfg.wrist_restore_steps+cfg.settle_steps+cfg.lower_steps
    assert all(grips[0]==1. for _,grips in commands)
    assert not any(name=='assist_table_loaded' for name,_ in commands)
