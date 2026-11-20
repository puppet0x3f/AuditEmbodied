"""Fixed-budget training of an isolated, two-suffix outcome LoRA."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

from .duration_pilot import file_hash
from .mid_recovery_data import read
from .vlm_mid_recovery import KEYS,candidate_keys,forecast,stream_logits


def set_boundary(values,ids,labels,q,threshold):
    import torch
    from .vlm_duration_lora import admission_boundary_loss
    indices=[i for i,c in enumerate(labels) if c!='A']
    safe=[i for i,c in enumerate(labels) if c=='A']
    if safe:indices.append(safe[0])
    return torch.stack([admission_boundary_loss(values[i],ids,labels[i],q,threshold,1.) for i in indices]).mean()


def sampling(rows,steps,seed):
    rng=random.Random(seed);groups={}
    for i,row in enumerate(rows):
        mask=''.join('A' if row['labels'][k]=='A' else 'N' for k in candidate_keys(row))
        key=(row.get('decision_step',30),mask,row.get('plan',{}).get('repair_arm','shared'))
        groups.setdefault(key,{}).setdefault(row['seed'],[]).append(i)
    names=sorted(groups);schedule=[];queues={k:[] for k in names}
    examples={}
    windows=sorted({k[0] for k in names});window_counts=Counter()
    for t in range(steps):
        window=windows[t%len(windows)];choices=[k for k in names if k[0]==window]
        key=choices[window_counts[window]%len(choices)];window_counts[window]+=1
        if not queues[key]:queues[key]=rng.sample(sorted(groups[key]),len(groups[key]))
        seed_id=queues[key].pop();sample_key=(key,seed_id)
        if not examples.get(sample_key):examples[sample_key]=rng.sample(groups[key][seed_id],len(groups[key][seed_id]))
        schedule.append(examples[sample_key].pop())
    return schedule,dict(Counter(str(rows[i].get('decision_step',30))+':'+''.join(rows[i]['labels'][k] for k in candidate_keys(rows[i])) for i in schedule))


def sampling_correction(rows,schedule):
    """Uniform families per window / original sampler's marginal probability.

    Stratum frequencies are fixed by the unchanged schedule. Seeds and examples
    are uniformly shuffled within each stratum, so their marginal probabilities
    remain uniform even though successive draws are without replacement.
    """
    groups={};windows=Counter();keys=[]
    for i,row in enumerate(rows):
        mask=''.join('A' if row['labels'][k]=='A' else 'N' for k in candidate_keys(row))
        key=(row.get('decision_step',30),mask,row.get('plan',{}).get('repair_arm','shared'))
        keys.append(key);windows[key[0]]+=1
        groups.setdefault(key,{}).setdefault(row['seed'],[]).append(i)
    counts=Counter(keys[i] for i in schedule)
    if not schedule or set(counts)!=set(groups):
        raise ValueError('correction requires sampled support for every training stratum')
    records=[]
    for i,row in enumerate(rows):
        key=keys[i];seeds=groups[key];n_seed_rows=len(seeds[row['seed']])
        target=1/len(windows)/windows[key[0]]
        proposal=counts[key]/len(schedule)/len(seeds)/n_seed_rows
        records.append({'index':i,'window':key[0],'mask':key[1],'repair_arm':key[2],
            'seed':row['seed'],'target_probability':target,'sampling_probability':proposal,
            'weight':target/proposal})
    return records


def training_loss(ce,complete,process,pair,boundary,weight=None,correction_scope='all'):
    if weight is None:return ce+.5*complete+.5*process+.1*pair+.1*boundary
    if correction_scope=='classification':return weight*(ce+.5*complete+.5*process)+.1*pair+.1*boundary
    if correction_scope!='all':raise ValueError('unknown correction scope')
    return weight*(ce+.5*complete+.5*process+.1*boundary)+.1*pair


def metrics(rows):
    def selected(row):return {'continue':'local','R_global':'global','defer':'defer'}.get(row['action'])
    safe=[r for r in rows if 'A' in r['labels'].values()]
    needed=[r for r in rows if r['labels']['local']!='A' and any(v=='A' for k,v in r['labels'].items() if k!='local')]
    no_safe=[r for r in rows if 'A' not in r['labels'].values()]
    return {'families':len(rows),'candidates':sum(len(r['records']) for r in rows),
        'candidate_class_correct':sum(x['prediction']==r['labels'][x['key']] for r in rows for x in r['records']),
        'false_admissions':sum(x['accepted'] and r['labels'][x['key']]!='A' for r in rows for x in r['records']),
        'safe_families':len(safe),'safe_candidate_coverage':sum(any(x['accepted'] and r['labels'][x['key']]=='A' for x in r['records']) for r in safe),
        'safe_actual_choices':sum(selected(r) is not None and r['labels'][selected(r)]=='A' for r in rows),
        'model_admissions':sum(r['model_admitted'] for r in rows),
        'model_admitted_safe_choices':sum(r['model_admitted'] and r['labels'][selected(r)]=='A' for r in rows),
        'baseline_fallbacks':sum(r['baseline_fallback'] for r in rows),
        'needed_upgrades':len(needed),'needed_upgrades_selected':sum(selected(r) not in (None,'local') and r['labels'][selected(r)]=='A' for r in needed),
        'unnecessary_upgrades':sum(r['action'] in ('R_global','defer') and r['labels']['local']=='A' for r in rows),
        'unsafe_continues':sum(r['action']=='continue' and r['labels']['local']!='A' for r in rows),
        'no_safe_families':len(no_safe),'no_safe_rejected':sum(r['action']=='reject' for r in no_safe),
        'safe_families_rejected':sum(r['action']=='reject' for r in safe),
        'actions':dict(Counter(r['action'] for r in rows))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--steps',type=int,default=512);parser.add_argument('--train-only',action='store_true')
    parser.add_argument('--sampling-correction',action='store_true')
    parser.add_argument('--correction-scope',choices=('all','classification'),default='all')
    parser.add_argument('--admission-policy',choices=('legacy','consistent'),default='consistent')
    parser.add_argument('--init-adapter',default='outputs/vlm-duration-family-stream-20261109/set512/adapter')
    parser.add_argument('--model-path',default='outputs/models/Qwen3-VL-8B-Instruct')
    parser.add_argument('--working-point',default='outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json')
    args=parser.parse_args();output=Path(args.output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'adapter').exists():raise ValueError('output adapter exists')
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from .vlm_duration_family import checkpoint_stateless_parts
    from .vlm_duration_lora import conditional_completion_loss,conditional_process_loss,family_pair_loss
    torch.manual_seed(20261109);torch.cuda.set_device(0)
    point=json.loads(Path(args.working_point).read_text());threshold=point['threshold']
    train=read(Path(args.data)/'train.jsonl');schedule,counts=sampling(train,args.steps,20261109)
    correction=sampling_correction(train,schedule) if args.sampling_correction else None
    if correction:
        (output/'sampling-weights.json').write_text(json.dumps(correction,indent=2)+'\n')
    frozen=[Path(args.working_point).resolve()]
    for folder in (args.init_adapter,point['start_adapter'],point['outcome_adapter']):
        frozen.extend(Path(folder).resolve()/n for n in ('adapter_config.json','adapter_model.safetensors'))
    hashes={str(p):file_hash(p) for p in frozen}
    config={'steps':args.steps,'candidate_presentations':sum(len(candidate_keys(train[i])) for i in schedule),'train_only':args.train_only,
        'init_adapter':str(Path(args.init_adapter).resolve()),'threshold':threshold,'seed':20261109,
        'train_families':len(train),'sampling_label_pairs':counts,'schedule_indices':schedule,
        'frozen':hashes,'loss':('w * (CE + .5 completion + .5 process) + .1 pair + .1 set-boundary; margin1'
            if correction and args.correction_scope=='classification' else 'w * (CE + .5 completion + .5 process + .1 set-boundary) + .1 pair; margin1'
            if correction else 'CE + .5 completion + .5 process + .1 pair + .1 set-boundary; margin1'),
        'set_boundary':'reject each non-A; admit cheapest executed A; both A prefer local',
        'data_hash':file_hash(Path(args.data)/'train.jsonl'),
        'input':'fresh entry RGB + causal robot/wrench history at20/30, committed10 steps to30/40' if train[0].get('two_window') else 'h0 RGB, measured relevel entry, robot/command error at20; committed10 steps to30',
        'sampling':'balance window, A/non-A pair, repair role; cycle initialisation seeds inside each stratum',
        'sampling_correction':args.sampling_correction,
        'correction_scope':args.correction_scope,'admission_policy':args.admission_policy,
        'correction_target':'uniform physical family rows within equally weighted windows' if correction else None,
        'correction_proposal':'scheduled stratum frequency x uniform shuffled seed x uniform shuffled row within seed' if correction else None,
        'correction_applies_to':('CE, completion, process only; pair and set-boundary unchanged' if args.correction_scope=='classification'
            else 'CE, completion, process and set-boundary only; pair coefficient .1 unchanged') if correction else None,
        'dev_usage':'final checkpoint evaluation only; no tuning or selection',
        'inference_path':'same adapter-specific split prefix/tail in training, evaluation, serving'}
    (output/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    processor=AutoProcessor.from_pretrained(args.model_path,local_files_only=True)
    tokens=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDE']
    assert all(len(x)==1 for x in tokens);ids=[x[0] for x in tokens]
    started=time.perf_counter()
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model_path,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,device_map={'':0})
    model=PeftModel.from_pretrained(model,args.init_adapter,is_trainable=True)
    model.gradient_checkpointing_disable();model.enable_input_require_grads();checkpoint_stateless_parts(model)
    parameters=[p for n,p in model.named_parameters() if p.requires_grad]
    assert not any(p.requires_grad and 'visual' in n for n,p in model.named_parameters())
    optimizer=torch.optim.AdamW(parameters,lr=3e-5,weight_decay=.01)
    print(json.dumps({'loaded':True,'load_seconds':time.perf_counter()-started}),flush=True)
    model.train();training_start=time.perf_counter()
    with (output/'train-log.jsonl').open('w') as log:
        for step,index in enumerate(schedule,1):
            tick=time.perf_counter();row=train[index];keys=candidate_keys(row);labels=[row['labels'][k] for k in keys]
            optimizer.zero_grad(set_to_none=True);values=stream_logits(model,processor,row);v=[values[k] for k in keys]
            ce=torch.stack([F.cross_entropy(x.view(1,-1),torch.tensor([ids['ABCD'.index(c)]],device=x.device)) for x,c in zip(v,labels)]).mean()
            complete=torch.stack([conditional_completion_loss(x,ids,c) for x,c in zip(v,labels)]).mean()
            process=torch.stack([conditional_process_loss(x,ids,c) for x,c in zip(v,labels)]).mean()
            pair,_=family_pair_loss(v,ids,labels)
            boundary=set_boundary(v,ids,labels,row['start_valid_score'],threshold)
            weight=correction[index]['weight'] if correction else None
            loss=training_loss(ce,complete,process,pair,boundary,weight,args.correction_scope)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            record={'step':step,'index':index,'labels':''.join(labels),'loss':float(loss.detach()),
                'ce':float(ce.detach()),'process':float(process.detach()),'boundary':float(boundary.detach()),
                'pair':float(pair.detach()),'importance_weight':weight,
                'gradient_norm':float(norm),'seconds':time.perf_counter()-tick}
            log.write(json.dumps(record)+'\n');log.flush()
            if step<=4 or step%16==0:print(json.dumps(record),flush=True)
    model.save_pretrained(output/'adapter')
    training_seconds=time.perf_counter()-training_start
    model.eval();reports={}
    for split in (['train'] if args.train_only else ['train','development']):
        records=[]
        with torch.inference_mode():
            for row in (train if split=='train' else read(Path(args.data)/'development.jsonl')):
                decision=forecast(stream_logits(model,processor,row),ids,row['start_valid_score'],threshold,args.admission_policy)
                for r in decision['records']:r['prediction']='ABCD'[max(range(4),key=lambda i:r['conditional_probabilities'][i])]
                records.append({**{k:row[k] for k in ('seed','scenario','source_wait_steps','labels','decision_step','all_labels') if k in row},**decision})
        (output/f'{split}-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        reports[split]=metrics(records)
        if any('decision_step' in r for r in records):
            reports[split]['by_window']={str(step):metrics([r for r in records if r['decision_step']==step]) for step in (30,40)}
    assert all(file_hash(p)==h for p,h in hashes.items())
    report={'steps':args.steps,'training_seconds':training_seconds,'frozen_unchanged':True,
        'prefix_gradient_norm':getattr(model,'mid_prefix_gradient_norm',None),
        'max_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'evaluations':reports}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
