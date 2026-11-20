"""Physical receiver grasp loss at a shared-support event, followed by fresh-policy recovery."""
from collections import deque
import numpy as np
from controlled_transport import ControlledImageEnv, snapshot
from vector_gate import MultiStepWrapper
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.gym_util.multistep_wrapper import aggregate, dict_take_last_n


class RecoveryImageEnv(ControlledImageEnv):
    def reset(self):
        obs=super().reset()
        self.support_run=0;self.event_triggered=False;self.just_intervened=False
        self.intervention_records=[];self.post_intervention_observations=[]
        self.fault_prefix=None;self.fault_end=None;self.fault_valid=False
        self.event_active_step=None;self.discarded_chunk_actions=0
        return obs

    def static_steps(self,steps,positions,rotations,grippers,phase):
        raw=self.env.env;controllers=[r.controller for r in raw.robots]
        originals=[c.set_goal for c in controllers]
        for arm,c in enumerate(controllers):
            original=originals[arm]
            def fixed(action,set_pos=None,set_ori=None,original=original,index=arm):
                return original(action,set_pos=positions[index],set_ori=rotations[index])
            c.set_goal=fixed
        try:
            for _ in range(steps):
                action=np.zeros(14,dtype=np.float32);action[[6,13]]=grippers
                before=self.capture_seconds
                obs,reward,done,info=RobomimicImageWrapper.step(self,action)
                record=self.observation(phase,len(self.actions),action,before)
                record['reward']=float(reward);self.intervention_records.append(record)
                self.post_intervention_observations.append(obs)
                self.post_intervention_observations=self.post_intervention_observations[-2:]
        finally:
            for c,original in zip(controllers,originals):c.set_goal=original
        return obs,reward,done,info

    def intervene(self):
        raw=self.env.env;controllers=[r.controller for r in raw.robots]
        self.event_triggered=True;self.just_intervened=True
        self.event_active_step=len(self.actions);self.fault_prefix=snapshot(self.env)
        registered=np.stack([c.goal_pos.copy() for c in controllers])
        rotations=[c.goal_ori.copy() for c in controllers]
        grips=(1.,-1.) if self.fault_kind=='fault' else (1.,1.)
        self.static_steps(12,registered,rotations,grips,'fault' if self.fault_kind=='fault' else 'sham')
        self.fault_end=snapshot(self.env)
        holds=self.intervention_records[-1]['payload_grasps']
        self.fault_valid=bool(holds[0] and (not holds[1] if self.fault_kind=='fault' else holds[1]))
        measured=np.stack([raw.sim.data.site_xpos[r.eef_site_id] for r in raw.robots])
        target=registered if self.recovery_hold=='registered' else measured
        self.holding_input={'entry':snapshot(self.env),'registered_positions':registered.tolist(),
            'measured_positions':measured.tolist(),'command_positions':target.tolist(),
            'registered_rotations':[r.tolist() for r in rotations],'grippers':list(grips)}
        answer=self.static_steps(self.recovery_wait,target,rotations,grips,'wait')
        self.holding_input['end']=snapshot(self.env)
        return answer

    def step(self,action):
        self.just_intervened=False
        answer=super().step(action)
        if not self.event_triggered:
            both=all(self.records[-1]['payload_grasps'])
            self.support_run=self.support_run+1 if both else 0
            if self.support_run>=3:
                answer=self.intervene()
        return answer


class RecoverySteps(MultiStepWrapper):
    def configure_recovery(self,fault,hold,wait):
        self.env.fault_kind=fault;self.env.recovery_hold=hold;self.env.recovery_wait=wait
    def step(self,actions):
        for index,action in enumerate(actions):
            if self.done and self.done[-1]:break
            obs,reward,done,info=self.env.step(action)
            self.obs.append(obs);self.reward.append(reward)
            done=bool(done or len(self.reward)>=self.max_episode_steps)
            self.done.append(done);self._add_info(info)
            if self.env.just_intervened:
                self.obs=deque(self.env.post_intervention_observations,maxlen=self.n_obs_steps+1)
                self.env.discarded_chunk_actions=len(actions)-index-1
                break
        return self._get_obs(self.n_obs_steps),aggregate(self.reward,self.reward_agg_method),aggregate(self.done,'max'),dict_take_last_n(self.info,self.n_obs_steps)


def factory(seed,cfg,render=True):
    return RecoverySteps(RecoveryImageEnv(seed,cfg,render),n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),max_episode_steps=int(cfg.task.env_runner.max_steps))
