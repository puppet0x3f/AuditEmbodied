"""Freeze a neutral initialization and label-independent matched sampling."""
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file,save_file

from auditembodied.vlm_duration_temporal import family_updates

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
PARENT=REPO/'outputs/vlm-duration-family-stream-20261109'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not (ROOT/'protocol.json').exists(),'never overwrite a frozen study'
old=json.loads((PARENT/'experiment-plan.json').read_text())
dataset=ROOT/'dataset';dataset.mkdir()
for name in ('train.jsonl','development.jsonl'):shutil.copy2(PARENT/'dataset'/name,dataset/name)
train=[json.loads(x) for x in (dataset/'train.jsonl').read_text().splitlines()]
dev=[json.loads(x) for x in (dataset/'development.jsonl').read_text().splitlines()]
assert len(train)==168 and len(dev)==24
assert not {r['seed'] for r in train}&{r['seed'] for r in dev}
adapter=ROOT/'neutral-adapter';adapter.mkdir()
reference=PARENT/'set512/adapter'
config=json.loads((reference/'adapter_config.json').read_text())
state=load_file(reference/'adapter_model.safetensors',device='cpu')
torch.manual_seed(20261109)
neutral={}
for name,value in state.items():
    tensor=torch.empty(value.shape,dtype=torch.float32)
    if '.lora_A.' in name:torch.nn.init.kaiming_uniform_(tensor,a=math.sqrt(5))
    elif '.lora_B.' in name:torch.nn.init.zeros_(tensor)
    else:raise ValueError('unexpected trained parameter in neutral template: '+name)
    neutral[name]=tensor
save_file(neutral,str(adapter/'adapter_model.safetensors'))
(adapter/'adapter_config.json').write_text(json.dumps(config,indent=2)+'\n')
updates,design=family_updates(train,512,20261109,balanced_replay=True,stratify_safe_sets=False)
(ROOT/'sampling-plan.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in updates))
snapshot=ROOT/'frozen-source'
if not snapshot.exists():
    shutil.copytree(REPO/'auditembodied',snapshot/'auditembodied',ignore=shutil.ignore_patterns('__pycache__'))
point_path=REPO/'outputs/vlm-duration-lora-riskreplay512-20261108/candidate-working-point.json'
point=json.loads(point_path.read_text())
frozen=[dataset/'train.jsonl',dataset/'development.jsonl',adapter/'adapter_config.json',adapter/'adapter_model.safetensors',point_path]
for folder in (point['start_adapter'],point['outcome_adapter']):
    frozen.extend(Path(folder).resolve()/name for name in ('adapter_config.json','adapter_model.safetensors'))
protocol={'scope':'neutral-initialized matched supervision; train/dev first, then fresh physics; no test tuning',
    'parent':str(PARENT),'model_path':old['model_path'],'data_dir':str(dataset),
    'initial_outcome_adapter':str(adapter),'frozen_start_adapter':point['start_adapter'],
    'working_point':str(point_path),'scope_threshold':point['threshold'],
    'variants':['ce','terminal','joint'],'steps':512,'candidate_presentations':1024,
    'seed':20261109,'learning_rate':.00003,'neutral_initialization':{
        'process_supervision_presentations':0,'all_B_matrices_zero':True,
        'A_initialization':'fresh Kaiming uniform; seed20261109; no pretrained adapter values copied',
        'base_function_unchanged':True,'trainable_parameters':sum(x.numel() for x in neutral.values())},
    'sampling_design':design,'safety_stratification':False,
    'development_calibration':{'grid':[.5,.75,.9,.95,.99,.995,.999,point['threshold'],1.],
        'criterion':'maximize safe executed selections including frozen60 fallback; minimize complete source-matrix cost; then prefer higher threshold',
        'scores':{'ce':'q*P(A)','joint':'q*P(A)','terminal':'q*(P(A)+P(B))'},
        'scope_threshold_unchanged':True,'no_refit_after_calibration':True},
    'formal_target':{'independent_seeds':20,'scenarios':['left_loss','right_loss'],
        'methods':['fixed60','ce','terminal','joint'],'episodes':160},
    'training_gpu':'1','frozen_inputs':{str(p):sha(p) for p in frozen},
    'source_sha256':{str(p.relative_to(snapshot)):sha(p) for p in (snapshot/'auditembodied').glob('*.py')}}
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'prepared':True,'neutral_parameters':protocol['neutral_initialization']['trainable_parameters'],
    'candidate_presentations':design['presentations'],'all_B_zero':all(torch.count_nonzero(v)==0 for k,v in neutral.items() if '.lora_B.' in k)}))
