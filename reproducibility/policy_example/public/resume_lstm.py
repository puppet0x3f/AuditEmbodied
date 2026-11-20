"""Bounded paired continuation of two released native Transport policies."""
import argparse,functools,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from baseline import HERE,load_policy,transform_action
from vector_gate import AsyncVectorEnv
from controlled_transport import factory,BOUNDARIES

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=('lstm_gmm','diffusion_policy_cnn'),required=True)
    parser.add_argument('--development',action='store_true');args=parser.parse_args()
    assert args.policy=='lstm_gmm' and not args.development
    protocol=json.loads((HERE/'protocol.json').read_text())
    continuation=json.loads((HERE/'lstm-continuation-freeze.json').read_text())
    for name,digest in continuation['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    if not args.development:
        freeze=json.loads((HERE/'formal-freeze.json').read_text())
        for name,digest in freeze['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    config=HERE/('lstm-checkpoint-config.yaml' if args.policy=='lstm_gmm' else 'cnn-checkpoint-config.yaml')
    cfg=OmegaConf.load(config);seeds=protocol['control_development_seeds'] if args.development else protocol['formal_seeds']
    horizon=40 if args.development else int(cfg.task.env_runner.max_steps)
    cfg.task.env_runner.max_steps=horizon
    out=HERE/('development-paired' if args.development else 'formal')/args.policy
    assert out.exists() and not (out/'completion.json').exists(),out
    actors=AsyncVectorEnv([functools.partial(factory,s,cfg) for s in seeds],
        dummy_env_fn=functools.partial(factory,seeds[0],cfg,False))
    policy,cfg_model=load_policy(args.policy,'cuda:0')
    warm={k:torch.zeros((len(seeds),int(cfg.n_obs_steps),*v['shape']),device='cuda:0') for k,v in cfg.shape_meta.obs.items()}
    with torch.no_grad():policy.predict_action(warm)['action'].detach().cpu().numpy()
    conditions=[]
    selected=[32] if args.development else list(BOUNDARIES)
    alternatives=[(hold,wait,boundary) for boundary in selected for hold in ('registered','measured_position') for wait in (1,20)]
    np.random.default_rng(20261120).shuffle(alternatives)
    existing=[json.loads(f.read_text()) for f in sorted(out.glob('[0-9]*.json'))]
    baseline={r['seed']:r for r in existing if r['hold_mode']=='unpaused'}
    assert len(baseline)==20
    summaries=[{k:r[k] for k in ('seed','method','hold_mode','wait_steps','pause_boundary','C','native_policy_steps','physical_wait_steps')} for r in existing]
    for hold,wait,boundary in alternatives:
        count=sum(r['hold_mode']==hold and r['wait_steps']==wait and r['pause_boundary']==boundary for r in existing)
        assert count in (0,20)
        if count==0:conditions.append((hold,wait,boundary))
    assert len(existing)+20*len(conditions)==180
    print(json.dumps({'preserved_existing':len(existing),'remaining_conditions':conditions}),flush=True)
    for hold,wait,boundary in conditions:
        torch.manual_seed(protocol['policy_query_seed']);np.random.seed(protocol['policy_query_seed']);random.seed(protocol['policy_query_seed']);policy.reset()
        actors.call_each('configure',args_list=[(hold,wait,boundary,baseline.get(s,{}).get('actions')) for s in seeds])
        actors.reset_async();obs=actors.reset_wait();query=[];step=0;started=time.perf_counter();port_hashes={};internal_states={}
        while step<horizon:
            if step in BOUNDARIES:
                port_hashes[str(step)]=[{k:hashlib.sha256(np.ascontiguousarray(v[i]).tobytes()).hexdigest() for k,v in obs.items()} for i in range(len(seeds))]
                if hold=='unpaused':np.savez_compressed(out/f'boundary-{step:03d}-policy-input.npz',**obs)
            started_query=time.perf_counter()
            port={k:torch.from_numpy(v).to('cuda:0') for k,v in obs.items()}
            with torch.no_grad():predicted=policy.predict_action(port)['action'].detach().cpu().numpy()
            query.append(time.perf_counter()-started_query);assert np.isfinite(predicted).all()
            if step in BOUNDARIES:
                internal={'cpu_rng':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                    'cuda_rng':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
                model=getattr(policy,'model',None)
                if hasattr(model,'_rnn_hidden_state'):
                    internal.update(counter=model._rnn_counter,
                        hidden=[hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in model._rnn_hidden_state])
                internal_states[str(step)]=internal
            shape=predicted.shape
            action=transform_action(predicted,bool(cfg.task.abs_action)).reshape(*shape[:-1],14) if cfg.task.abs_action else predicted
            obs,reward,done,info=actors.step(action);step+=int(cfg.n_action_steps)
            if step%100<int(cfg.n_action_steps):
                print(json.dumps({'policy':args.policy,'hold':hold,'wait':wait,'boundary':boundary,'steps':step,'completed':int((reward>=1).sum()),'paths':len(seeds),'wall_seconds':time.perf_counter()-started}),flush=True)
            if np.all(done):break
        collected={name:actors.call('get_attr',name) for name in ('records','actions','initial_state','boundary_states','holding_input','hold_tracking','wait_records','prefix_action_max_residual','capture_seconds')}
        for i,seed in enumerate(seeds):
            row={k:v[i] for k,v in collected.items()};events=sorted(row['records']+row['wait_records'],key=lambda r:r['time'])
            row.update(seed=seed,method=args.policy,hold_mode=hold,wait_steps=wait,pause_boundary=boundary,
                scenario='unpaused' if hold=='unpaused' else 'early' if boundary==32 else 'transport',
                policy_query_seconds=query,policy_query_seed=protocol['policy_query_seed'],batch_seeds=seeds,
                prefix_observation_sha256={k:v[i] for k,v in port_hashes.items()},
                policy_boundary_state=internal_states,
                C=any(r['C'] for r in events),max_reward=max(r['reward'] for r in row['records']),
                native_policy_steps=len(row['actions']),physical_wait_steps=len(row['wait_records']),
                simulated_seconds=float(events[-1]['time']-row['initial_state']['time']),
                query_seconds=float(sum(query)),observation_service_seconds=float(row['capture_seconds']),
                batch_wall_seconds=time.perf_counter()-started,
                timing_protocol='Native synchronous benchmark; a single additional physical wait occurs after a query and before its first action. Other measured query/acquisition times are charged, not replayed as extra physical waits.',
                state_edits_after_reset=0,cohort='control-development' if args.development else 'formal')
            row['charged_seconds']=row['simulated_seconds']+row['query_seconds']+row['observation_service_seconds']
            assert abs(row['simulated_seconds']-.05*(row['native_policy_steps']+row['physical_wait_steps']))<1e-7
            assert row['native_policy_steps']==horizon
            if hold=='unpaused':baseline[seed]=row
            else:
                reference=baseline[seed]
                assert row['initial_state']==reference['initial_state'],('initial',seed)
                assert row['holding_input']['entry']==reference['boundary_states'][str(boundary)],('prefix',seed,hold,wait,boundary)
                assert row['prefix_observation_sha256'][str(boundary)]==reference['prefix_observation_sha256'][str(boundary)],('RGB history',seed)
                assert row['policy_boundary_state'][str(boundary)]==reference['policy_boundary_state'][str(boundary)],('policy state',seed)
            filename=f'{seed}-{boundary:03d}-{hold}-{wait:02d}.json'
            assert not (out/filename).exists(),filename
            (out/filename).write_text(json.dumps(row,separators=(',',':'))+'\n')
            summaries.append({k:row[k] for k in ('seed','method','hold_mode','wait_steps','pause_boundary','C','native_policy_steps','physical_wait_steps')})
        print(json.dumps({'condition_done':[args.policy,hold,wait,boundary],'records_saved':len(summaries)}),flush=True)
    actors.close();(out/'completion.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps({'policy':args.policy,'total_records':len(summaries),'errors':0}),flush=True)

if __name__=='__main__':main()
