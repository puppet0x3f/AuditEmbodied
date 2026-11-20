import json
import os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from auditembodied.vlm_duration_family import encode_prefix,encode_full,encode_tail,cached_logits

if os.environ.get('FAMILY_BF16_FULL_PRECISION')=='1':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
root=Path(__file__).resolve().parent;p=json.loads((root/'experiment-plan.json').read_text())
row=json.loads((root/'dataset/train.jsonl').read_text().splitlines()[0])
processor=AutoProcessor.from_pretrained(p['model_path'],local_files_only=True)
model=Qwen3VLForConditionalGeneration.from_pretrained(p['model_path'],dtype=torch.bfloat16,
    device_map={'':0},attn_implementation='sdpa',local_files_only=True)
model=PeftModel.from_pretrained(model,p['initial_outcome_adapter'],is_trainable=False)
model.set_adapter('default',inference_mode=True);model.eval();base=model.get_base_model().model
ids=[processor.tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCD']
with torch.inference_mode():
    prefix=encode_prefix(processor,row,model.device);full,positions=encode_full(processor,row,model.device)
    length=prefix['input_ids'].shape[1]
    ropefull,deltafull=base.get_rope_index(full['input_ids'],full['mm_token_type_ids'],full['image_grid_thw'],attention_mask=full['attention_mask'])
    ropepre,deltapre=base.get_rope_index(prefix['input_ids'],prefix['mm_token_type_ids'],prefix['image_grid_thw'],attention_mask=prefix['attention_mask'])
    reconstructed=torch.arange(length,full['input_ids'].shape[1],device=model.device).view(1,1,-1).expand(3,1,-1)+deltapre.view(1,1,1)
    print('ROPE',deltafull.tolist(),deltapre.tolist(),int((ropefull[:,:,:length]-ropepre).abs().max()),int((ropefull[:,:,length:]-reconstructed).abs().max()),flush=True)
    out=model(**full,logits_to_keep=torch.tensor(positions,device=model.device),use_cache=True)
    full_logits=out.logits[0].float();full_keys=[layer.keys[:,:,:length].clone() for layer in out.past_key_values.layers]
    del out
    pre=model(**prefix,logits_to_keep=1,use_cache=True)
    for i in (0,1,8,16,35):
        a=full_keys[i].float();b=pre.past_key_values.layers[i].keys.float()
        print('KV',i,float((a-b).abs().max()),float((a-b).abs().mean()),flush=True)
    del full_keys
    cache={'past_key_values':pre.past_key_values,'length':length,'attention_mask':prefix['attention_mask'],
        'rope_deltas':base.rope_deltas.clone(),'adapter':model.active_adapter,
        'prefix_text':__import__('auditembodied.vlm_duration_family',fromlist=['prefix_text']).prefix_text(row),
        'image':row['before_image']}
    values=cached_logits(model,processor,row,cache)
    for i,n in enumerate((60,40,30)):
        print('LOGITS',n,full_logits[i,ids].tolist(),values[n][ids].tolist(),flush=True)
    # Text-axis positions make causal sequence positions explicit, separately from visual RoPE.
    tail,q=encode_tail(processor,row,model.device)
    raw=torch.arange(length,length+tail['input_ids'].shape[1],device=model.device).view(1,1,-1)
    tail['position_ids']=torch.cat([raw,raw.expand(3,1,-1)+deltapre.view(1,1,1)],dim=0)
    tail['attention_mask']=torch.cat([prefix['attention_mask'],tail['attention_mask']],1)
    pre.past_key_values.crop(length)
    out=model(**tail,past_key_values=pre.past_key_values,use_cache=True,
        logits_to_keep=torch.tensor(q,device=model.device)).logits[0].float()
    for i,n in enumerate((60,40,30)):print('FOUR_AXIS',n,out[i,ids].tolist(),flush=True)
