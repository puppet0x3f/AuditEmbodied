"""Training-only cache parity and deadline probe; no accuracy claim or fitting."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration

from auditembodied.vlm_duration_family import cached_logits,encode_tail,stream_logits,prefill

parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parent))
parser.add_argument('--adapter');parser.add_argument('--output-name',default='cache-probe')
args=parser.parse_args();root=Path(args.root);p=json.loads((root/'experiment-plan.json').read_text())
rows=[json.loads(s) for s in (root/'dataset/train.jsonl').read_text().splitlines()][::3][:12]
processor=AutoProcessor.from_pretrained(p['model_path'],local_files_only=True)
model=Qwen3VLForConditionalGeneration.from_pretrained(p['model_path'],dtype=torch.bfloat16,
    device_map={'':0},attn_implementation='sdpa',local_files_only=True)
model=PeftModel.from_pretrained(model,args.adapter or p['initial_outcome_adapter'],is_trainable=False)
model.set_adapter('default',inference_mode=True);model.eval()
ids=[processor.tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCD']
records=[]
with torch.inference_mode():
    # Warm only an existing training observation.
    stream_logits(model,processor,rows[0]);cache=prefill(model,processor,rows[0]);cached_logits(model,processor,rows[0],cache)
    del cache
    for row in rows:
        torch.cuda.synchronize();tick=time.perf_counter();full=stream_logits(model,processor,row)
        torch.cuda.synchronize();full_s=time.perf_counter()-tick
        tick=time.perf_counter();cache=prefill(model,processor,row)
        torch.cuda.synchronize();prefix_s=time.perf_counter()-tick
        tail_times=[];errors=[];identical=[]
        for repetition in range(3):
            tick=time.perf_counter();cached=cached_logits(model,processor,row,cache)
            torch.cuda.synchronize();tail_times.append(time.perf_counter()-tick)
            for n in (30,40,60):
                a=torch.softmax(full[n][ids],0);b=torch.softmax(cached[n][ids],0)
                errors.append(float(torch.max(torch.abs(a-b))))
                identical.append(int(a.argmax())==int(b.argmax()))
            assert cache['past_key_values'].get_seq_length()==cache['length']
        record={'seed':row['seed'],'scenario':row['scenario'],'W_steps':row['wait_steps'],
            'prefix_tokens':cache['length'],'tail_tokens':encode_tail(processor,row)[0]['input_ids'].shape[1],
            'full_seconds':full_s,'prefill_seconds':prefix_s,'tail_seconds':tail_times,
            'max_probability_error':max(errors),'same_native_class':all(identical)}
        records.append(record);print(json.dumps(record),flush=True)
        del cache,full,cached
tail=[v for r in records for v in r['tail_seconds']]
report={'training_families':len(records),'tail_calls':len(tail),'tail_p50_seconds':float(np.median(tail)),
    'tail_p95_seconds':float(np.percentile(tail,95)),'tail_max_seconds':max(tail),
    'max_probability_error':max(r['max_probability_error'] for r in records),
    'reference':'fresh prefix-plus-tail, identical to training/evaluation path','same_native_classes':all(r['same_native_class'] for r in records),'GPU':torch.cuda.get_device_name(0),
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cases':records,
    'adapter':args.adapter or p['initial_outcome_adapter'],
    'scope':'Training observations only; cache equivalence and compute timing, not held-out classification performance or policy success.'}
(root/(args.output_name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)
