"""Resume the stateless checkpoint from its exact native action roll-in and queue.

Native absolute OSC commands are float32. Restore that dtype after JSON decoding;
float64 axis-angle conversion creates a different physical roll-in. Actual executed
prefix actions, physical states, RGB histories and RNG states are the controls.
"""
import functools,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from baseline import HERE,load_policy,transform_action
from vector_gate import AsyncVectorEnv
from controlled_transport import ControlledImageEnv,ControlledSteps,snapshot

class ExactDiffusionRollin(ControlledImageEnv):
    def reset(self):
        obs=super().reset();self.numerical_prefix_prediction_max_residual=0.
        return obs
    def step(self,action):
        index=len(self.actions)
        if self.prefix_actions is not None and index<self.pause_step+self.chunk_steps:
            expected=np.asarray(self.prefix_actions[index])
            self.numerical_prefix_prediction_max_residual=max(self.numerical_prefix_prediction_max_residual,float(np.max(abs(expected-action))))
            action=expected
        if index==self.pause_step:
            actual=snapshot(self.env)
            if actual!=self.expected_entry:
                (HERE/f'diffusion-prefix-mismatch-{self.seed_value}.json').write_text(json.dumps({'actual':actual,'expected':self.expected_entry},indent=2)+'\n')
                raise AssertionError(('physical prefix',self.seed_value))
        return super().step(action)

class ExactSteps(ControlledSteps):
    def configure(self,hold,wait,pause_step,prefix_actions,expected_entry,expected_initial):
        super().configure(hold,wait,pause_step,np.asarray(prefix_actions,dtype=np.float32))
        self.env.expected_entry=expected_entry;self.env.expected_initial=expected_initial
    def reset(self):
        obs=super().reset();assert self.env.initial_state==self.env.expected_initial,('initial state',self.env.seed_value)
        return obs

def factory(seed,cfg,render=True):
    return ExactSteps(ExactDiffusionRollin(seed,cfg,render),n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),max_episode_steps=700)

def main():
    for filename in ('formal-freeze.json','diffusion-continuation-freeze.json'):
        freeze=json.loads((HERE/filename).read_text())
        for name,digest in freeze['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    protocol=json.loads((HERE/'protocol.json').read_text());seeds=protocol['formal_seeds']
    out=HERE/'formal/diffusion_policy_cnn'
    baseline={s:json.loads((out/f'{s}-000-unpaused-00.json').read_text()) for s in seeds}
    assert len(list(out.glob('[0-9]*.json')))==20,'Resume only before any completed holding comparison'
    cfg=OmegaConf.load(HERE/'cnn-checkpoint-config.yaml')
    actors=AsyncVectorEnv([functools.partial(factory,s,cfg) for s in seeds],dummy_env_fn=functools.partial(factory,seeds[0],cfg,False))
    policy,cfg=load_policy('diffusion_policy_cnn','cuda:0')
    warm={k:torch.zeros((len(seeds),int(cfg.n_obs_steps),*v['shape']),device='cuda:0') for k,v in cfg.shape_meta.obs.items()}
    with torch.no_grad():policy.predict_action(warm)['action'].detach().cpu().numpy()
    conditions=[(h,w,b) for b in (32,272) for h in ('registered','measured_position') for w in (1,20)]
    np.random.default_rng(20261120).shuffle(conditions)
    summaries=[{k:r[k] for k in ('seed','method','hold_mode','wait_steps','pause_boundary','C','native_policy_steps','physical_wait_steps')} for r in baseline.values()]
    for hold,wait,boundary in conditions:
        torch.manual_seed(protocol['policy_query_seed']);np.random.seed(protocol['policy_query_seed']);random.seed(protocol['policy_query_seed']);policy.reset()
        actors.call_each('configure',args_list=[(hold,wait,boundary,baseline[s]['actions'],baseline[s]['boundary_states'][str(boundary)],baseline[s]['initial_state']) for s in seeds])
        actors.reset_async();obs=actors.reset_wait();query=[];step=0;started=time.perf_counter();port_hashes={};internal_states={}
        while step<700:
            if step in (32,272):port_hashes[str(step)]=[{k:hashlib.sha256(np.ascontiguousarray(v[i]).tobytes()).hexdigest() for k,v in obs.items()} for i in range(len(seeds))]
            t=time.perf_counter();port={k:torch.from_numpy(v).to('cuda:0') for k,v in obs.items()}
            with torch.no_grad():predicted=policy.predict_action(port)['action'].detach().cpu().numpy()
            query.append(time.perf_counter()-t);assert np.isfinite(predicted).all()
            if step in (32,272):
                internal_states[str(step)]={'cpu_rng':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    'cuda_rng':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
            if step==boundary:
                for i,seed in enumerate(seeds):
                    assert port_hashes[str(step)][i]==baseline[seed]['prefix_observation_sha256'][str(step)],('RGB prefix',seed)
                    assert internal_states[str(step)]==baseline[seed]['policy_boundary_state'][str(step)],('RNG prefix',seed)
            shape=predicted.shape;action=transform_action(predicted,True).reshape(*shape[:-1],14)
            obs,reward,done,info=actors.step(action);step+=8
            if step%100<8:print(json.dumps({'policy':'diffusion_policy_cnn','hold':hold,'wait':wait,'boundary':boundary,'steps':step,'completed':int((reward>=1).sum()),'paths':len(seeds),'wall_seconds':time.perf_counter()-started}),flush=True)
            if np.all(done):break
        names=('records','actions','initial_state','boundary_states','holding_input','hold_tracking','wait_records','prefix_action_max_residual','capture_seconds','numerical_prefix_prediction_max_residual')
        collected={name:actors.call('get_attr',name) for name in names}
        for i,seed in enumerate(seeds):
            row={k:v[i] for k,v in collected.items()};events=sorted(row['records']+row['wait_records'],key=lambda r:r['time'])
            row.update(seed=seed,method='diffusion_policy_cnn',hold_mode=hold,wait_steps=wait,pause_boundary=boundary,
                scenario='early' if boundary==32 else 'transport',policy_query_seconds=query,policy_query_seed=protocol['policy_query_seed'],batch_seeds=seeds,
                prefix_observation_sha256={k:v[i] for k,v in port_hashes.items()},policy_boundary_state=internal_states,
                C=any(r['C'] for r in events),max_reward=max(r['reward'] for r in row['records']),native_policy_steps=len(row['actions']),physical_wait_steps=len(row['wait_records']),
                simulated_seconds=float(events[-1]['time']-row['initial_state']['time']),query_seconds=float(sum(query)),observation_service_seconds=float(row['capture_seconds']),batch_wall_seconds=time.perf_counter()-started,
                timing_protocol=baseline[seed]['timing_protocol'],state_edits_after_reset=0,cohort='formal',
                prefix_protocol='Exact original native actions and committed chunk; stateless diffusion policy; physical/RGB/RNG states match the original unpaused trajectory. Unused repeated predictions are numerical diagnostics.')
            row['charged_seconds']=row['simulated_seconds']+row['query_seconds']+row['observation_service_seconds']
            assert row['native_policy_steps']==700 and len(row['wait_records'])==wait
            assert row['initial_state']==baseline[seed]['initial_state']
            assert row['holding_input']['entry']==baseline[seed]['boundary_states'][str(boundary)]
            filename=f'{seed}-{boundary:03d}-{hold}-{wait:02d}.json';assert not (out/filename).exists()
            (out/filename).write_text(json.dumps(row,separators=(',',':'))+'\n')
            summaries.append({k:row[k] for k in ('seed','method','hold_mode','wait_steps','pause_boundary','C','native_policy_steps','physical_wait_steps')})
        print(json.dumps({'condition_done':['diffusion_policy_cnn',hold,wait,boundary],'records_saved':len(summaries)}),flush=True)
    actors.close();(out/'completion.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
