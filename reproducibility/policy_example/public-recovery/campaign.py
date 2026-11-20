import argparse,functools,hashlib,json,random,time,datetime
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from baseline import HERE,load_policy,transform_action
from vector_gate import AsyncVectorEnv
from recovery_transport import factory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('development','formal'),required=True)
    args=parser.parse_args();protocol=json.loads((HERE/'protocol.json').read_text())
    if args.phase=='formal':
        frozen=json.loads((HERE/'formal-freeze.json').read_text())
        for name,digest in frozen['source_sha256'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    cfg=OmegaConf.load(HERE/'cnn-checkpoint-config.yaml');seeds=protocol['development_seeds'] if args.phase=='development' else protocol['formal_seeds']
    out=HERE/args.phase;out.mkdir(exist_ok=False)
    actors=AsyncVectorEnv([functools.partial(factory,s,cfg) for s in seeds],dummy_env_fn=functools.partial(factory,seeds[0],cfg,False))
    policy,model_cfg=load_policy('diffusion_policy_cnn','cuda:0')
    warm={k:torch.zeros((len(seeds),int(cfg.n_obs_steps),*v['shape']),device='cuda:0') for k,v in cfg.shape_meta.obs.items()}
    with torch.no_grad():policy.predict_action(warm)
    conditions=[('fault','registered',1),('sham','registered',1)] if args.phase=='development' else [
        (fault,hold,wait) for fault in ('fault','sham') for hold in ('registered','measured_position') for wait in (1,20)]
    if args.phase=='formal':np.random.default_rng(20261123).shuffle(conditions)
    reference={};summaries=[]
    for fault,hold,wait in conditions:
        torch.manual_seed(20261123);np.random.seed(20261123);random.seed(20261123);policy.reset()
        actors.call_each('configure_recovery',args_list=[(fault,hold,wait) for _ in seeds])
        actors.reset_async();obs=actors.reset_wait();queries=[];started=time.perf_counter();postfault_ports={};q=0
        done=np.zeros(len(seeds),dtype=bool)
        while not np.all(done):
            triggered=actors.call('get_attr','event_triggered')
            for i,t in enumerate(triggered):
                if t and seeds[i] not in postfault_ports:
                    port={k:v[i] for k,v in obs.items()}
                    filename=f'{seeds[i]}-{fault}-{hold}-{wait:02d}-recovery-input.npz'
                    np.savez_compressed(out/filename,**port)
                    postfault_ports[seeds[i]]={'filename':filename,'sha256':{k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in port.items()},'query_index':q}
            port={k:torch.from_numpy(v).to('cuda:0') for k,v in obs.items()}
            tick=time.perf_counter()
            with torch.no_grad():prediction=policy.predict_action(port)['action'].detach().cpu().numpy()
            queries.append(time.perf_counter()-tick);assert np.isfinite(prediction).all()
            shape=prediction.shape;actions=transform_action(prediction,True).reshape(*shape[:-1],14)
            obs,reward,done,info=actors.step(actions);q+=1
            if q%12==0:print(json.dumps({'phase':args.phase,'condition':[fault,hold,wait],'queries':q,'native_completions':int((reward>=1).sum()),'triggered':sum(triggered),'wall_seconds':time.perf_counter()-started}),flush=True)
        fields=('records','actions','initial_state','holding_input','capture_seconds','event_triggered','fault_valid','fault_prefix','fault_end','event_active_step','discarded_chunk_actions','intervention_records')
        values={name:actors.call('get_attr',name) for name in fields}
        for i,seed in enumerate(seeds):
            r={k:v[i] for k,v in values.items()};events=sorted(r['records']+r['intervention_records'],key=lambda f:f['time'])
            native=any(f['C'] for f in events)
            r.update(seed=seed,method='diffusion_policy_cnn',scenario=fault,hold_mode=hold,wait_steps=wait,
                C_native=native,C=bool(native and r['event_triggered'] and r['fault_valid']),
                active_policy_steps=len(r['actions']),injected_steps=12 if r['event_triggered'] else 0,
                physical_wait_steps=wait if r['event_triggered'] else 0,policy_query_seconds=queries,
                query_seconds=float(sum(queries)),observation_service_seconds=float(r['capture_seconds']),
                simulated_seconds=float(events[-1]['time']-r['initial_state']['time']),
                recovery_policy_input=postfault_ports.get(seed),batch_seeds=seeds,cohort=args.phase,
                policy_query_seed=20261123,state_edits_after_reset=0,
                completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                batch_wall_seconds=time.perf_counter()-started,
                timing_protocol=protocol['timing'])
            r['charged_seconds']=r['simulated_seconds']+r['query_seconds']+r['observation_service_seconds']
            assert r['active_policy_steps']==700
            assert abs(r['simulated_seconds']-.05*(700+r['injected_steps']+r['physical_wait_steps']))<1e-7
            if seed in reference:
                base=reference[seed]
                assert r['initial_state']==base['initial_state'],('initial',seed)
                assert r['fault_prefix']==base['fault_prefix'],('event prefix',seed,fault,hold,wait)
                assert r['event_active_step']==base['event_active_step']
            else:reference[seed]=r
            path=out/f'{seed}-{fault}-{hold}-{wait:02d}.json'
            path.write_text(json.dumps(r,separators=(',',':'))+'\n')
            summaries.append({k:r[k] for k in ('seed','scenario','hold_mode','wait_steps','C','C_native','event_triggered','fault_valid','event_active_step')})
        print(json.dumps({'condition_done':[fault,hold,wait],'saved':len(summaries),'successes':sum(r['C'] for r in summaries[-len(seeds):])}),flush=True)
    actors.close();(HERE/(args.phase+'-completion.json')).write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps({'phase':args.phase,'episodes':len(summaries)}),flush=True)

if __name__=='__main__':main()
