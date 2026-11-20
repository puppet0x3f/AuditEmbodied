"""Evaluate unmodified public image policies on their native Transport task."""
import argparse,collections,gc,json,random,sys,time,resource,multiprocessing
import collections.abc
# Python 3.11 relocated this alias; the released environment uses the old import.
collections.Iterable=collections.abc.Iterable
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'official-source/diffusion_policy-main'))
import numpy as np
import torch,dill,hydra
from omegaconf import OmegaConf
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

def load_policy(name,device):
    print('loading checkpoint',name,flush=True)
    payload=torch.load(HERE/(name+'.ckpt'),map_location='cpu',pickle_module=dill,weights_only=False)
    print('instantiating frozen policy',name,flush=True)
    cfg=payload['cfg'];policy=hydra.utils.instantiate(cfg.policy)
    state='ema_model' if name=='diffusion_policy_cnn' and cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][state],strict=True)
    print('strict state load passed',name,flush=True)
    del payload;gc.collect()
    policy.eval();print('moving policy to',device,flush=True);policy.to(device);policy.reset()
    print('policy GPU ready',name,flush=True)
    (HERE/(name+'-checkpoint-config.yaml')).write_text(OmegaConf.to_yaml(cfg))
    return policy,cfg

def make_env(cfg,enable_render=True):
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.envs.env_robosuite import EnvRobosuite
    modalities=collections.defaultdict(list)
    for k,v in cfg.shape_meta.obs.items():modalities[v.get('type','low_dim')].append(k)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modalities)
    meta=json.loads((HERE/'native-env-meta.json').read_text());kw=meta['env_kwargs']
    kw['camera_names']=[k.removesuffix('_image') for k in modalities['rgb']]
    kw['controller_configs']['control_delta']=not bool(cfg.task.abs_action)
    env=EnvRobosuite(meta['env_name'],render=False,render_offscreen=enable_render,use_image_obs=enable_render,**kw)
    env.env.hard_reset=False
    return env

def transform_action(action,absolute):
    if not absolute:return action
    a=action.reshape(-1,2,10);rotation=RotationTransformer('axis_angle','rotation_6d').inverse(a[...,3:9])
    return np.concatenate((a[...,:3],rotation,a[...,9:]),axis=-1).reshape(-1,14)

def policy_worker(connection,name,device):
    policy,cfg=load_policy(name,device);connection.send({'ready':True})
    while True:
        request=connection.recv()
        if request is None:break
        if request['kind']=='reset':
            seed=request['seed'];np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
            policy.reset();connection.send({'reset':True});print('policy reset',seed,flush=True);continue
        port={k:torch.from_numpy(v).to(device) for k,v in request['port'].items()}
        with torch.no_grad():predicted=policy.predict_action(port)['action'][0].detach().cpu().numpy()
        connection.send(predicted)

class PolicyProcess:
    def __init__(self,name,device):
        context=multiprocessing.get_context('spawn');self.connection,child=context.Pipe()
        self.process=context.Process(target=policy_worker,args=(child,name,device));self.process.start()
        child.close()
        try:assert self.connection.recv()['ready']
        except EOFError:
            self.process.join();raise RuntimeError(f'Policy worker terminated before ready: {self.process.exitcode}')
    def reset(self,seed):
        self.connection.send({'kind':'reset','seed':seed});assert self.connection.recv()['reset']
    def predict(self,port):
        self.connection.send({'kind':'predict','port':port});return self.connection.recv()
    def close(self):
        self.connection.send(None);self.process.join()

def description(env,step):
    raw=env.env
    return {'step':step,'time':float(raw.sim.data.time),
        'eef_positions':[raw.sim.data.site_xpos[r.eef_site_id].tolist() for r in raw.robots],
        'payload_position':np.asarray(raw.transport.payload_pos).tolist(),
        'lid_position':np.asarray(raw.transport.lid_handle_pos).tolist(),
        'trash_position':np.asarray(raw.transport.trash_pos).tolist(),
        'success':bool(raw._check_success())}

def run(policy,cfg,env,seed,device,out):
    np.random.seed(seed);random.seed(seed)
    policy.reset(seed);obs=env.reset();history=collections.deque(maxlen=int(cfg.n_obs_steps))
    history.extend([obs]*int(cfg.n_obs_steps));steps=0;rewards=[];services=[];actions=[];trace=[]
    initial=env.get_state()['states'].tolist();started=time.perf_counter()
    while steps<int(cfg.task.env_runner.max_steps):
        port={k:np.stack([o[k] for o in history])[None].astype(np.float32) for k in cfg.shape_meta.obs}
        t=time.perf_counter();predicted=policy.predict(port);services.append(time.perf_counter()-t)
        assert np.isfinite(predicted).all()
        batch=transform_action(predicted,bool(cfg.task.abs_action))
        for a in batch:
            obs,reward,done,info=env.step(a);history.append(obs);steps+=1
            rewards.append(float(reward));actions.append(a.tolist());trace.append(description(env,steps))
            if reward>=1 or steps>=int(cfg.task.env_runner.max_steps):break
        if rewards[-1]>=1:break
        if steps%100<max(1,len(batch)):print(json.dumps({'seed':seed,'steps':steps,'max_reward':max(rewards)}),flush=True)
    result={'seed':seed,'policy':cfg.policy._target_,'checkpoint':out.parent.name,'max_reward':max(rewards),
        'C':max(rewards)>=1,'policy_control_steps':steps,'policy_service_seconds':services,
        'native_sim_seconds':steps*.05,'charged_seconds':steps*.05+sum(services),
        'wall_seconds':time.perf_counter()-started,'initial_state':initial,'actions':actions,'trace':trace,
        'timing_protocol':'Native synchronous benchmark for policy queries; charged cost includes measured inference. No added physical pause in this capability gate.'}
    out.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ('seed','C','max_reward','policy_control_steps','charged_seconds','wall_seconds')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',choices=('lstm_gmm','diffusion_policy_cnn'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seeds',type=int,nargs='+',default=list(range(91000,91005)))
    a=p.parse_args();folder=HERE/'development'/a.policy;folder.mkdir(parents=True,exist_ok=True)
    config_name='lstm-checkpoint-config.yaml' if a.policy=='lstm_gmm' else 'cnn-checkpoint-config.yaml'
    print('native renderer and CUDA policy execute in separate spawned processes',flush=True)
    env=make_env(OmegaConf.load(HERE/config_name))
    print('environment ready RSS KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    cfg=OmegaConf.load(HERE/config_name);policy=PolicyProcess(a.policy,a.device)
    print('policy ready RSS KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    for seed in a.seeds:
        out=folder/(str(seed)+'.json');assert not out.exists(),out
        run(policy,cfg,env,seed,a.device,out)
    policy.close()
