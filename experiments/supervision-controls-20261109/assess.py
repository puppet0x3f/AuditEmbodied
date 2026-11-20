"""Final-only matched supervision assessment and predeclared dev calibration."""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def matrix(protocol):
    source=Path(protocol['parent'])
    spec=importlib.util.spec_from_file_location('previous_matrix_assessment',
        source.parent/'vlm-duration-jointfamily-full512-20261109/assess.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parent=json.loads((source/'experiment-plan.json').read_text())
    physical=module.load_physics(parent)
    for row in read(Path(parent['source_new_collection'])/'trials.jsonl'):
        key=row['seed'],row['scenario'],row['post_prefix_wait_steps'],row['local_steps']['local_relevel_steps']
        label=('B' if row['protection_violation'] else 'A') if row['success'] else ('D' if row['protection_violation'] else 'C')
        physical[key]={'label':label,'post_fault_seconds':row['total_post_fault_seconds']}
    return physical


def selection(rows,tau,physical):
    groups=defaultdict(list)
    for row in rows:groups[row['seed'],row['scenario'],row['wait_steps']].append(row)
    cases=[]
    for key,group in groups.items():
        ordered=sorted(group,key=lambda x:x['plan']['local_relevel_steps'])
        assert len(ordered)==3
        for row in ordered:
            assert physical[(*key,row['plan']['local_relevel_steps'])]['label']==row['label']
        admitted=next((r for r in ordered if r['admission_score']>=tau),None)
        rejected=admitted is None and all(1-r['admission_score']>=tau for r in ordered)
        chosen=admitted if admitted is not None else None if rejected else ordered[-1]
        cases.append({'seed':key[0],'scenario':key[1],'wait_steps':key[2],
            'selected_steps':chosen['plan']['local_relevel_steps'] if chosen else None,
            'label':chosen['label'] if chosen else None,'model_admitted':admitted is not None,
            'baseline_fallback':admitted is None and not rejected,'explicit_rejection':rejected,
            'total_seconds':physical[(*key,chosen['plan']['local_relevel_steps'])]['post_fault_seconds'] if chosen else None})
    return {'families':len(cases),'safe_selected':sum(r['label']=='A' for r in cases),
        'unsafe_selected':sum(r['label'] is not None and r['label']!='A' for r in cases),
        'refusals':sum(r['explicit_rejection'] for r in cases),
        'model_admissions':sum(r['model_admitted'] for r in cases),
        'fallbacks':sum(r['baseline_fallback'] for r in cases),
        'mean_observed_selected_seconds':mean(r['total_seconds'] for r in cases if r['total_seconds'] is not None),
        'safe_candidates_admitted':sum(r['label']=='A' and r['admission_score']>=tau for r in rows),
        'unsafe_candidates_admitted':sum(r['label']!='A' and r['admission_score']>=tau for r in rows),
        'cases':cases}


def main():
    p=json.loads((ROOT/'protocol.json').read_text())
    physical=matrix(p);report={'scope':'same neutral outcome initialization, inputs, sampler and finite supervision budget',
        'training':{},'development':{},'calibration':{},'frozen_inputs_unchanged':True}
    for variant in p['variants']:
        folder=ROOT/(variant+'512')
        result=json.loads((folder/'result.json').read_text())
        config=json.loads((folder/'protocol.json').read_text())
        training=read(folder/'training.jsonl')
        assert len(training)==512 and sum(len(r['example_indices']) for r in training)==1024
        assert (folder/'sampling-plan.jsonl').read_bytes()==(ROOT/'sampling-plan.jsonl').read_bytes()
        assert config['init_adapter']==p['initial_outcome_adapter']
        assert result['data_unchanged'] and result['frozen_start_unchanged']
        if variant!='joint':
            assert all(r['pair_loss']==r['auxiliary_task_loss']==r['process_marginal_loss']==r['admission_boundary_loss']==0 for r in training)
        predictions=read(folder/'after-predictions.jsonl')
        grid=[]
        for tau in p['development_calibration']['grid']:
            m=selection(predictions,tau,physical);grid.append({'threshold':tau,**{k:v for k,v in m.items() if k!='cases'}})
        key=lambda r:(r['safe_selected'],-r['unsafe_selected'],-r['refusals'],
            -(r['mean_observed_selected_seconds'] or float('inf')),r['threshold'])
        chosen=max(grid,key=key)
        tau=chosen['threshold']
        report['calibration'][variant]={'threshold':tau,'grid':grid,
            'score_mode':'completion' if variant=='terminal' else 'safe',
            'rule':p['development_calibration']['criterion'],
            'adapter':str((folder/'adapter').resolve()),
            'adapter_sha256':{name:sha(folder/'adapter'/name) for name in ('adapter_config.json','adapter_model.safetensors')}}
        report['development'][variant]=selection(predictions,tau,physical)
        train=read(folder/'train-after-predictions.jsonl')
        report['training'][variant]={k:result[k] for k in ('completed_steps','training_seconds','peak_allocated_gpu_bytes','data_unchanged','frozen_start_unchanged')}
        report['training'][variant].update(candidate_presentations=1024,language_forward_calls=1024,
            selection=selection(train,tau,physical),first_prefix_gradient_norm=training[0]['first_prefix_gradient_norm'])
    assert all(sha(path)==digest for path,digest in p['frozen_inputs'].items())
    (ROOT/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'calibration.json').write_text(json.dumps(report['calibration'],indent=2)+'\n')
    print(json.dumps({'thresholds':{k:v['threshold'] for k,v in report['calibration'].items()},
        'development':{k:{x:v[x] for x in ('safe_selected','families','unsafe_candidates_admitted','model_admissions','fallbacks')}
            for k,v in report['development'].items()},'matched_sampling':True}),flush=True)


if __name__=='__main__':main()
