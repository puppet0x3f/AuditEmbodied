"""Position-reference intervention around native public-policy action boundaries."""
import functools,json,time
import numpy as np
from vector_gate import RecordedImageEnv,MultiStepWrapper,factory as gate_factory
from baseline import make_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

BOUNDARIES=(32,272)

def snapshot(env):
    raw=env.env;data=raw.sim.data
    return {'time':float(data.time),'state':raw.sim.get_state().flatten().tolist(),
        'ctrl':data.ctrl.tolist(),'qacc_warmstart':data.qacc_warmstart.tolist(),
        'gripper_command_state':[r.gripper.current_action.tolist() for r in raw.robots],
        'controller_position_goals':[None if r.controller.goal_pos is None else r.controller.goal_pos.tolist() for r in raw.robots],
        'controller_rotation_goals':[None if r.controller.goal_ori is None else r.controller.goal_ori.tolist() for r in raw.robots]}

class ControlledImageEnv(RecordedImageEnv):
    def __init__(self,seed,cfg,render=True):
        super().__init__(seed,cfg,render)
        # Materialize native controller/solver caches before the scored reset.
        if render:
            self.env.reset();self.env.step(np.zeros(14))
        self.hold='unpaused';self.wait=0;self.pause_step=0;self.prefix_actions=None
        raw=self.env.env;original=raw._get_observations;self.capture_seconds=0.
        def measured(*args,**kwargs):
            started=time.perf_counter();value=original(*args,**kwargs)
            self.capture_seconds+=time.perf_counter()-started
            return value
        raw._get_observations=measured
        self.chunk_steps=int(cfg.n_action_steps)
    def reset(self):
        # A fresh native reset avoids cached controller/solver state from a previous path.
        np.random.seed(self.seed_value);raw=self.env.reset();obs=self.get_observation(raw)
        for robot in self.env.env.robots:
            robot.gripper.current_action=np.zeros(robot.gripper.dof)
        self.records=[];self.actions=[];self.initial_state=snapshot(self.env)
        self.boundary_states={};self.holding_input=None;self.hold_tracking=[];self.wait_records=[]
        self.capture_seconds=0.;self.prefix_action_max_residual=0.
        return obs
    def observation(self,phase,policy_step,action,capture_before):
        raw=self.env.env;group=raw.transport
        positions=np.stack([raw.sim.data.site_xpos[r.eef_site_id] for r in raw.robots])
        payload=bool(group.payload_in_target_bin);trash=bool(group.trash_in_trash_bin)
        return {'time':float(raw.sim.data.time),'phase':phase,'policy_step':policy_step,
            'eef_positions':positions.tolist(),'payload_position':np.asarray(group.payload_pos).tolist(),
            'lid_position':np.asarray(group.lid_handle_pos).tolist(),'trash_position':np.asarray(group.trash_pos).tolist(),
            'payload_in_target_bin':payload,'trash_in_trash_bin':trash,'C':payload and trash,
            'payload_grasps':[bool(raw._check_grasp(r.gripper,group.objects['payload'].contact_geoms)) for r in raw.robots],
            'action':np.asarray(action).tolist(),'observation_service_seconds':self.capture_seconds-capture_before}
    def pause(self):
        raw=self.env.env;controllers=[r.controller for r in raw.robots]
        measured=np.stack([raw.sim.data.site_xpos[r.eef_site_id] for r in raw.robots])
        registered=np.stack([c.goal_pos.copy() for c in controllers]);rotations=[c.goal_ori.copy() for c in controllers]
        target=registered if self.hold=='registered' else measured
        self.holding_input={'measured_positions':measured.tolist(),'registered_positions':registered.tolist(),
            'command_positions':target.tolist(),'registered_rotations':[r.tolist() for r in rotations],
            'grippers':np.asarray(self.actions[-1])[[6,13]].tolist(),'entry':snapshot(self.env)}
        originals=[c.set_goal for c in controllers]
        for i,c in enumerate(controllers):
            original=originals[i]
            def fixed(action,set_pos=None,set_ori=None,original=original,index=i):
                return original(action,set_pos=target[index],set_ori=rotations[index])
            c.set_goal=fixed
        try:
            for i in range(self.wait):
                before=self.capture_seconds
                action=np.zeros(14);action[[6,13]]=np.asarray(self.actions[-1])[[6,13]]
                self.env.step(action)
                actual=np.stack([raw.sim.data.site_xpos[r.eef_site_id] for r in raw.robots])
                goals=np.stack([c.goal_pos for c in controllers])
                assert np.allclose(goals,target,atol=1e-12,rtol=0)
                assert all(np.allclose(c.goal_ori,r,atol=1e-12,rtol=0) for c,r in zip(controllers,rotations))
                self.hold_tracking.append({'time':float(raw.sim.data.time),'command_positions':goals.tolist(),
                    'measured_positions':actual.tolist()})
                self.wait_records.append(self.observation('wait',len(self.actions),action,before))
        finally:
            for c,original in zip(controllers,originals):c.set_goal=original
        self.holding_input['end']=snapshot(self.env)
        assert abs(self.holding_input['end']['time']-self.holding_input['entry']['time']-.05*self.wait)<1e-8
    def step(self,action):
        index=len(self.actions)
        if index in BOUNDARIES:self.boundary_states[str(index)]=snapshot(self.env)
        # Replay the native policy's exact roll-in and the already queried action chunk.
        # This is physical action execution, never a post-reset simulator state edit.
        if self.prefix_actions is not None and index<self.pause_step+self.chunk_steps:
            expected=np.asarray(self.prefix_actions[index]);residual=float(np.max(abs(expected-action)))
            assert residual<1e-5,(self.seed_value,index,residual)
            self.prefix_action_max_residual=max(self.prefix_action_max_residual,residual)
            action=expected
        if self.hold!='unpaused' and index==self.pause_step:self.pause()
        before=self.capture_seconds;obs,reward,done,info=RobomimicImageWrapper.step(self,action)
        record=self.observation('policy',index+1,action,before);record['reward']=float(reward)
        assert bool(reward>=1)==record['C']
        self.records.append(record);self.actions.append(np.asarray(action).tolist())
        return obs,reward,done,info

class ControlledSteps(MultiStepWrapper):
    def configure(self,hold,wait,pause_step,prefix_actions):
        self.env.hold=hold;self.env.wait=wait;self.env.pause_step=pause_step;self.env.prefix_actions=prefix_actions

def factory(seed,cfg,render=True):
    return ControlledSteps(ControlledImageEnv(seed,cfg,render),n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),max_episode_steps=int(cfg.task.env_runner.max_steps))
