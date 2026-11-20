"""Frozen native-token duration policy, served across the GPU/physics runtimes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from unittest.mock import patch

from .l4_lift import LiftConfig
from .vlm_duration_data import plan
from .vlm_duration_lora import start_key,task_prompt_for


def routed_input(example,task,context,post_wait_mode='paired'):
    """Route by executed phase; the old and prerequisite prompts stay exact."""
    if task=='start':return 'start',[context['image']],task_prompt_for(example,task)
    if context.get('post_wait'):
        if post_wait_mode=='family_stream':raise ValueError('family_stream outcomes require family_prefill/family_tail RPC')
        from .vlm_duration_temporal import input_spec
        row={**example,'image':context['image']}
        if post_wait_mode in ('stage_early','stage_late','stage_joint'):
            for k in ('before_image','before_robot','after_prefix_image','after_prefix_robot','early_history','late_history'):
                row[k]=context[k]
        if post_wait_mode=='paired':
            row.update(before_image=context['before_image'],before_robot=context['before_robot'],wait_steps=context['wait_steps'])
        paths,prompt=input_spec(row,task,post_wait_mode)
        return 'post_wait',paths,prompt
    return 'default',[context['image']],task_prompt_for(example,task)


def candidate_bank():
    cfg=LiftConfig()
    candidates=[{"candidate":"continue","plan":plan({"candidate":"continue"}),"planned_recovery_seconds":0.}]
    for n in (30,40,60):
        for candidate in ("R_left","R_right"):
            p=plan({"candidate":candidate,"local_steps":{"local_relevel_steps":n}})
            candidates.append({"candidate":candidate,"plan":p,
                               "planned_recovery_seconds":(p['local_approach_steps']+p['local_close_steps']+n+p['final_settle_steps'])/20})
    global_steps=cfg.table_lower_steps+8+14+6+16+24+20+22+cfg.lift_steps+12
    candidates.append({"candidate":"R_global","plan":plan({"candidate":"R_global"}),
                       "planned_recovery_seconds":global_steps/20})
    return sorted(candidates,key=lambda r:(r['planned_recovery_seconds'],r['candidate']))


def choose_candidate(robot,threshold,score):
    """Select the first admitted plan in increasing nominal cost, without truth."""
    starts,records={},[]
    for candidate in candidate_bank():
        p=candidate['plan'];example={'robot':robot,'plan':p}
        q=1.
        if p['kind']=='local':
            role=p['repair_arm']
            if role not in starts:
                starts[role]=score(example,'start')[0]
            q=starts[role]
            if q<threshold:
                records.append({**candidate,'start_valid_score':q,'pruned_by_start_upper_bound':True})
                continue
        conditional=score(example,'outcome')
        value=q*conditional[0]
        records.append({**candidate,'start_valid_score':q,'conditional_probabilities':conditional,'joint_A_score':value})
        if value>=threshold:
            decision={'candidate':candidate['candidate'],'accepted':True,'threshold':threshold,
                      'plan':p,'planned_recovery_seconds':candidate['planned_recovery_seconds'],
                      'start_scores':starts,'scored_candidates':records}
            if p['kind']=='local':decision['local_relevel_steps']=p['local_relevel_steps']
            return decision
    return {'candidate':'abstain','accepted':False,'threshold':threshold,'start_scores':starts,
            'scored_candidates':records,'reason':'no_candidate_above_frozen_threshold'}


def choose_scope(robot,threshold,score):
    """Gate a shared local prefix by the unchanged start-prerequisite model."""
    starts={}
    for candidate in ('R_left','R_right'):
        example={'robot':robot,'plan':plan({'candidate':candidate})}
        starts[example['plan']['repair_arm']]=score(example,'start')
    eligible=[role for role,values in starts.items() if values[0]>=threshold]
    if len(eligible)==1:
        return {'candidate':'R_'+eligible[0],'parallel_local':True,'accepted':False,
                'threshold':threshold,'start_scores':{role:v[0] for role,v in starts.items()},
                'reason':'unique_start_prerequisite_passed; duration_not_yet_admitted'}
    def cached_score(example,task):
        return starts[example['plan']['repair_arm']] if task=='start' else score(example,task)
    return {**choose_candidate(robot,threshold,cached_score),'parallel_local':False}


def choose_local_duration(robot,candidate,threshold,score):
    """Evaluate the duration family from the refreshed pre-prefix observation."""
    if candidate not in ('R_left','R_right'):raise ValueError('local role required')
    p=plan({'candidate':candidate})
    q=score({'robot':robot,'plan':p},'start')[0]
    records=[]
    if q>=threshold:
        for item in candidate_bank():
            if item['candidate']!=candidate:continue
            conditional=score({'robot':robot,'plan':item['plan']},'outcome')
            value=q*conditional[0]
            records.append({**item,'start_valid_score':q,'conditional_probabilities':conditional,'joint_A_score':value})
            if value>=threshold:
                return {**item,'accepted':True,'threshold':threshold,'start_valid_score':q,
                        'local_relevel_steps':item['plan']['local_relevel_steps'],'scored_candidates':records}
    return {'candidate':candidate,'accepted':False,'threshold':threshold,'start_valid_score':q,
            'scored_candidates':records,'reason':'no_local_duration_above_frozen_threshold'}


def score_local_family(robot,candidate,threshold,score):
    """Offline assessment: score all three durations, without threshold pruning."""
    if candidate not in ('R_left','R_right'):raise ValueError('local role required')
    q=score({'robot':robot,'plan':plan({'candidate':candidate})},'start')[0]
    records=[]
    for item in candidate_bank():
        if item['candidate']!=candidate:continue
        conditional=score({'robot':robot,'plan':item['plan']},'outcome')
        probabilities=[q*v for v in conditional]+[1.-q]
        records.append({**item,'start_valid_score':q,'conditional_probabilities':conditional,
            'class_probabilities':probabilities,'prediction':'ABCDE'[max(range(5),key=lambda i:probabilities[i])],
            'joint_A_score':probabilities[0],'accepted':probabilities[0]>=threshold})
    return {'candidate':candidate,'start_valid_score':q,'threshold':threshold,'scored_candidates':records}


def serve(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from peft import PeftModel

    started=time.perf_counter()
    point=json.loads(Path(args.working_point).read_text())
    threshold=point['threshold']
    processor=AutoProcessor.from_pretrained(args.model_path,local_files_only=True)
    processor.tokenizer.padding_side='left'
    ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDE']
    if any(len(x)!=1 for x in ids):raise ValueError('answers must be single native tokens')
    ids=[x[0] for x in ids]
    torch.cuda.set_device(0)
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model_path,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True,device_map={'':0})
    print('Loading frozen default adapter',file=sys.stderr,flush=True)
    model=PeftModel.from_pretrained(model,point['outcome_adapter'],is_trainable=False)
    print('Loading frozen start adapter',file=sys.stderr,flush=True)
    model.load_adapter(point['start_adapter'],adapter_name='start',is_trainable=False)
    print('Loading post-prefix adapter',file=sys.stderr,flush=True)
    if args.post_wait_adapter:model.load_adapter(args.post_wait_adapter,adapter_name='post_wait',is_trainable=False)
    if args.family_control_adapter:model.load_adapter(args.family_control_adapter,adapter_name='family_control',is_trainable=False)
    if args.mid_adapter:model.load_adapter(args.mid_adapter,adapter_name='mid',is_trainable=False)
    model.set_adapter('default',inference_mode=True)
    model.config.use_cache=False
    model.eval()
    if any(p.requires_grad for p in model.parameters()):raise ValueError('frozen worker has trainable parameters')
    print('Frozen adapters ready; checking original reference',file=sys.stderr,flush=True)

    def load_images(path):
        with np.load(path) as archive:return [Image.fromarray(x.copy()) for x in archive['rgb']]

    start_cache={}
    def scorer(image_path,context=None):
        calls=[];context={'image':image_path,**(context or {})};cache={image_path:load_images(image_path)}
        visual_cache={}
        def score(example,task):
            tick=time.perf_counter()
            adapter,paths,prompt=routed_input(example,task,context,args.post_wait_mode)
            if adapter=='post_wait' and not args.post_wait_adapter:raise ValueError('post-wait adapter not loaded')
            if model.active_adapter!=adapter:model.set_adapter(adapter,inference_mode=True)
            images=[]
            for path in paths:
                if path not in cache:cache[path]=load_images(path)
                images.extend(cache[path])
            content=[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]
            text=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
            batch=processor(text=[text],images=images,return_tensors='pt').to('cuda:0')
            multimodal=model.get_base_model().model
            original_features=multimodal.get_image_features
            def shared_features(*args,**kwargs):
                key=tuple(paths)
                if key not in visual_cache:visual_cache[key]=original_features(*args,**kwargs)
                return visual_cache[key]
            sharing=context.get('op') in ('score_examples_shared','choose_examples_shared') and task=='outcome'
            feature_context=patch.object(multimodal,'get_image_features',shared_features) if sharing else nullcontext()
            with torch.inference_mode(),feature_context:
                values=model(**batch,logits_to_keep=1,use_cache=False).logits[0,-1].float()
                chosen=[ids[0],ids[4]] if task=='start' else ids[:4]
                probabilities=torch.softmax(values[chosen],0).cpu().tolist()
            calls.append({'task':task,'adapter':adapter,'images':len(images),'seconds':time.perf_counter()-tick})
            return probabilities
        def batch_outcomes(examples):
            tick=time.perf_counter();texts=[];all_images=[];adapters=set()
            for example in examples:
                adapter,paths,prompt=routed_input(example,'outcome',context,args.post_wait_mode)
                adapters.add(adapter);images=[]
                for path in paths:
                    if path not in cache:cache[path]=load_images(path)
                    images.extend(cache[path])
                content=[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]
                texts.append(processor.apply_chat_template([{'role':'user','content':content}],
                    tokenize=False,add_generation_prompt=True))
                all_images.extend(images)
            if len(adapters)!=1:raise ValueError('one adapter required per batch')
            adapter=adapters.pop()
            if adapter=='post_wait' and not args.post_wait_adapter:raise ValueError('post-wait adapter not loaded')
            if model.active_adapter!=adapter:model.set_adapter(adapter,inference_mode=True)
            batch=processor(text=texts,images=all_images,padding=True,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                values=model(**batch,logits_to_keep=1,use_cache=False).logits[:,-1].float()
                probabilities=torch.softmax(values[:,ids[:4]],-1).cpu().tolist()
            calls.append({'task':'outcome_batch','adapter':adapter,'examples':len(examples),
                'images':len(all_images),'seconds':time.perf_counter()-tick})
            return probabilities
        score.batch_outcomes=batch_outcomes
        return score,calls

    # Existing training observation only: warm both adapters and verify loading.
    training=[json.loads(s) for s in Path(args.reference_data).read_text().splitlines()]
    expected=[json.loads(s) for s in Path(args.reference_predictions).read_text().splitlines()]
    index=next(i for i,r in enumerate(training) if r['plan']['kind']=='local')
    example=training[index];score,_=scorer(example['image'])
    q=score(example,'start')[0];conditional=score(example,'outcome')
    error=max(abs(q-expected[index]['start_valid_score']),
              max(abs(a-b) for a,b in zip(conditional,expected[index]['conditional_probabilities'])))
    if error>1e-6:raise ValueError(f'frozen reference scores differ: {error}')
    for candidate in candidate_bank():
        score({'robot':example['robot'],'plan':candidate['plan']},'outcome')
    score({'robot':example['robot'],'plan':plan({'candidate':'R_right'})},'start')
    print(json.dumps({'ready':True,'setup_seconds':time.perf_counter()-started,'reference_max_error':error,
                      'gpu':torch.cuda.get_device_name(0),'threshold':threshold}),flush=True)
    family_context=None;mid_context=None
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        try:
            tick=time.perf_counter()
            operation=request.get('op','choose')
            if operation in ('mid_prefill','mid_tail','mid_fresh','mid_release'):
                from .vlm_mid_recovery import prefill,cached_logits,stream_logits,forecast
                if not args.mid_adapter:raise ValueError('mid adapter not loaded')
                if operation=='mid_release':
                    mid_context=None;decision={'released':True}
                else:
                    example=request['example']
                    if operation=='mid_prefill':
                        entry={'image':example['before_image'],'robot':example['before_robot'],'plan':example['plan']}
                        key=start_key(entry)
                        if key not in start_cache:
                            start_score,_=scorer(entry['image']);start_cache[key]=start_score(entry,'start')[0]
                        model.set_adapter('mid',inference_mode=True)
                        with torch.inference_mode():kv=prefill(model,processor,example)
                        mid_context={'kv':kv,'start_score':start_cache[key]}
                        decision={'prefilled':True,'prefix_tokens':kv['length'],'start_valid_score':start_cache[key]}
                    else:
                        if mid_context is None:raise ValueError('mid context not prefilled')
                        model.set_adapter('mid',inference_mode=True)
                        with torch.inference_mode():
                            values=(cached_logits(model,processor,example,mid_context['kv']) if operation=='mid_tail'
                                else stream_logits(model,processor,example))
                            decision=forecast(values,ids,mid_context['start_score'],threshold)
                        decision.update(attempt_id=request.get('attempt_id'),prerequisite_phase='h0')
                if operation!='mid_release':torch.cuda.synchronize()
                decision.update(model_pipeline_seconds=time.perf_counter()-tick,forward_calls=0 if operation=='mid_release' else 1)
                print(json.dumps(decision),flush=True);continue
            if operation in ('family_prefill','family_tail','family_full','family_release'):
                from .vlm_duration_family import entry_example,prefill,cached_logits,stream_logits,forecast
                if not args.post_wait_adapter or args.post_wait_mode!='family_stream':
                    raise ValueError('family readouts require the family_stream adapter/input mode')
                if operation=='family_release':
                    family_context=None;decision={'released':True}
                else:
                    example=request['example']
                    if operation=='family_prefill':
                        entry=entry_example(example);key=start_key(entry)
                        if key not in start_cache:
                            start_score,_=scorer(entry['image'])
                            start_cache[key]=start_score(entry,'start')[0]
                        adapter=request.get('adapter_name','post_wait')
                        if adapter not in ('post_wait','family_control') or adapter=='family_control' and not args.family_control_adapter:
                            raise ValueError('unknown loaded family adapter')
                        model.set_adapter(adapter,inference_mode=True)
                        with torch.inference_mode():kv=prefill(model,processor,example)
                        family_context={'kv':kv,'start_score':start_cache[key]}
                        decision={'prefilled':True,'prefix_tokens':kv['length'],'start_valid_score':start_cache[key],
                            'prerequisite_phase':'h0'}
                    else:
                        if family_context is None:raise ValueError('family context not prefilled')
                        model.set_adapter(family_context['kv']['adapter'],inference_mode=True)
                        with torch.inference_mode():
                            values=(cached_logits(model,processor,example,family_context['kv']) if operation=='family_tail'
                                else stream_logits(model,processor,example))
                            decision=forecast(values,ids,request['plans'],family_context['start_score'],threshold)
                        decision.update(attempt_id=request.get('attempt_id'),prerequisite_phase='h0')
                if operation!='family_release':torch.cuda.synchronize()
                decision.update(model_pipeline_seconds=time.perf_counter()-tick,
                    forward_calls=0 if operation=='family_release' else 1)
                print(json.dumps(decision),flush=True)
                continue
            score,calls=scorer(request['image'],request)
            if operation=='scope':decision=choose_scope(request['robot'],threshold,score)
            elif operation=='cache_start':
                example={'image':request['image'],'robot':request['robot'],'plan':request['plan']}
                key=start_key(example);hit=key in start_cache
                if not hit:start_cache[key]=score(example,'start')[0]
                decision={'start_valid_score':start_cache[key],'start_cache_hit':hit}
            elif operation=='local_durations':
                decision=choose_local_duration(request['robot'],request['candidate'],threshold,score)
            elif operation=='score_local_family':
                decision=score_local_family(request['robot'],request['candidate'],threshold,score)
            elif operation in ('score_examples','score_examples_batch','score_examples_shared','choose_examples_shared'):
                records=[];starts={}
                cache_hits=[]
                examples=[{'image':request['image'],'robot':request['robot'],'plan':p} for p in request['plans']]
                if operation in ('score_examples_batch','score_examples_shared','choose_examples_shared'):
                    for example in examples:
                        if example['plan']['kind']!='local':continue
                        role=example['plan']['repair_arm'];key=start_key(example)
                        if role in starts:continue
                        hit=key in start_cache;cache_hits.append(hit)
                        if not hit:start_cache[key]=score(example,'start')[0]
                        starts[role]=start_cache[key]
                    if operation=='score_examples_batch':outcomes=score.batch_outcomes(examples)
                for index,p in enumerate(request['plans']):
                    example={'robot':request['robot'],'plan':p};q=1.
                    if p['kind']=='local':
                        role=p['repair_arm']
                        if role not in starts:starts[role]=score(example,'start')[0]
                        q=starts[role]
                    conditional=outcomes[index] if operation=='score_examples_batch' else score(example,'outcome')
                    probabilities=[q*x for x in conditional]+[1.-q]
                    records.append({'plan':p,'start_valid_score':q,'conditional_probabilities':conditional,
                        'class_probabilities':probabilities,'prediction':'ABCDE'[max(range(5),key=lambda i:probabilities[i])],
                        'accepted':probabilities[0]>=threshold})
                    if operation=='choose_examples_shared' and probabilities[0]>=threshold:break
                decision={'records':records,'start_cache_hits':cache_hits}
            elif operation=='choose':decision=choose_candidate(request['robot'],threshold,score)
            else:raise ValueError(f'unknown operation: {operation}')
            if operation=='scope':
                for role,q in decision.get('start_scores',{}).items():
                    entry={'image':request['image'],'robot':request['robot'],'plan':plan({'candidate':'R_'+role})}
                    start_cache[start_key(entry)]=q
            decision.update(model_pipeline_seconds=time.perf_counter()-tick,forward_calls=len(calls),forward_timings=calls)
            print(json.dumps(decision),flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error':repr(error)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--working-point',required=True)
    parser.add_argument('--model-path',required=True)
    parser.add_argument('--post-wait-adapter')
    parser.add_argument('--family-control-adapter')
    parser.add_argument('--mid-adapter')
    parser.add_argument('--post-wait-mode',choices=('static','paired','stage_early','stage_late','stage_joint','family_stream'),default='paired')
    parser.add_argument('--reference-data',default='outputs/l4-duration-newstates-20261108/dataset/train.jsonl')
    parser.add_argument('--reference-predictions',default='outputs/vlm-duration-lora-riskreplay512-20261108/train-after-predictions.jsonl')
    serve(parser.parse_args())


if __name__=='__main__':main()
