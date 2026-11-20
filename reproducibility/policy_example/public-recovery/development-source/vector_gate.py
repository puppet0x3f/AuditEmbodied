"""Native released-policy capability gate, using the official vector wrappers."""
import argparse,functools,json,random,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from baseline import make_env,load_policy,transform_action,HERE,description
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util import async_vector_env as vector_module
from gym.vector import utils as gym_vector_utils
# Gym 0.23 moved the space argument first; preserve the released 0.21 wrapper API.
vector_module.read_from_shared_memory=lambda shared,space,n=1:gym_vector_utils.read_from_shared_memory(space,shared,n)
vector_module.write_to_shared_memory=lambda index,value,shared,space:gym_vector_utils.write_to_shared_memory(space,index,value,shared)
vector_module.concatenate=lambda items,out,space:gym_vector_utils.concatenate(space,items,out)

class RecordedImageEnv(RobomimicImageWrapper):
    def __init__(self,seed,cfg,render=True):
        np.random.seed(seed)
        env=make_env(cfg,render)
        super().__init__(env,cfg.shape_meta,render_obs_key='shouldercamera0_image')
        self.seed(seed);self.records=[];self.actions=[];self.initial_state=None;self.seed_value=seed
    def reset(self):
        self.seed(self.seed_value);obs=super().reset()
        self.records=[];self.actions=[];self.initial_state=self.env.get_state()['states'].tolist()
        return obs
    def step(self,action):
        obs,reward,done,info=super().step(action)
        d=description(self.env,len(self.records)+1);d['reward']=float(reward)
        self.records.append(d);self.actions.append(np.asarray(action).tolist())
        return obs,reward,done,info

def factory(seed,cfg,render=True):
    return MultiStepWrapper(RecordedImageEnv(seed,cfg,render),n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),max_episode_steps=int(cfg.task.env_runner.max_steps))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=('lstm_gmm','diffusion_policy_cnn'),required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(91000,91005)));a=p.parse_args()
    config=HERE/('lstm-checkpoint-config.yaml' if a.policy=='lstm_gmm' else 'cnn-checkpoint-config.yaml')
    cfg=OmegaConf.load(config);out=HERE/'development-vector'/a.policy;out.mkdir(parents=True,exist_ok=True)
    assert not list(out.glob('*.json')),'Existing outcomes are never overwritten'
    np.random.seed(20261119);random.seed(20261119)
    functions=[functools.partial(factory,s,cfg) for s in a.seeds]
    env=AsyncVectorEnv(functions,dummy_env_fn=functools.partial(factory,a.seeds[0],cfg,False))
    policy,cfg=load_policy(a.policy,'cuda:0');torch.manual_seed(20261119);policy.reset()
    env.reset_async();obs=env.reset_wait();services=[];step=0;started=time.perf_counter()
    while step<int(cfg.task.env_runner.max_steps):
        port={k:torch.from_numpy(v).to('cuda:0') for k,v in obs.items()}
        t=time.perf_counter()
        with torch.no_grad():action=policy.predict_action(port)['action'].detach().cpu().numpy()
        services.append(time.perf_counter()-t)
        assert np.isfinite(action).all()
        original_shape=action.shape
        if cfg.task.abs_action:action=transform_action(action,bool(cfg.task.abs_action)).reshape(*original_shape[:-1],14)
        obs,reward,done,info=env.step(action);step+=int(cfg.n_action_steps)
        if step%100<int(cfg.n_action_steps):print(json.dumps({'steps':step,'max_rewards':reward.tolist(),'wall_seconds':time.perf_counter()-started}),flush=True)
        if np.all(done):break
    traces=env.call('get_attr','records');actions=env.call('get_attr','actions');initial=env.call('get_attr','initial_state')
    outcomes=[]
    for seed,trace,acts,state in zip(a.seeds,traces,actions,initial):
        r={'seed':seed,'checkpoint':a.policy,'C':any(t['success'] for t in trace),'max_reward':max(t['reward'] for t in trace),
            'trace':trace,'actions':acts,'initial_state':state,'query_seconds':services,'batch_seeds':a.seeds,
            'policy_query_seed':20261119,'native_action_steps':len(trace),'timing':'Released synchronous benchmark with measured batch query latency; no imposed physical waiting.'}
        assert r['C']==(r['max_reward']>=1)
        (out/(str(seed)+'.json')).write_text(json.dumps(r,separators=(',',':'))+'\n');outcomes.append({'seed':seed,'C':r['C']})
    env.close();(out/'summary.json').write_text(json.dumps(outcomes,indent=2)+'\n');print(json.dumps(outcomes),flush=True)
