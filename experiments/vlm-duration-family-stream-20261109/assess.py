"""Final-only matched losses, state coverage and measured source-matrix costs."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=json.loads((ROOT/'experiment-plan.json').read_text())
    spec=importlib.util.spec_from_file_location('selection_assessment',
        ROOT.parent/'vlm-duration-jointfamily-full512-20261109/assess.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    physics=old.load_physics(p)
    for r in old.read(Path(p['source_new_collection'])/'trials.jsonl'):
        n=r['local_steps']['local_relevel_steps'];w=r['post_prefix_wait_steps']
        physics[r['seed'],r['scenario'],w,n]={
            'label':('B' if r['protection_violation'] else 'A') if r['success'] else ('D' if r['protection_violation'] else 'C'),
            'post_fault_seconds':r['total_post_fault_seconds'],'episode_simulated_seconds':r['episode_simulated_seconds'],
            'wait_seconds':r['inference_wait_seconds'],'skill_seconds':r['skill_seconds'],'final_hold_seconds':r['hold_seconds'],
            'decision_prefix_seconds':r['stage_capture']['h2']['state']['time']-r['prefix_state']['time']+w/20,
            'path_m':r['recovery_path_m']}
    paths={'previous':ROOT.parent/'vlm-duration-batch-intervention-20261109/mixed512',
        'set':ROOT/'set512','per_candidate':ROOT/'candidate512'}
    report={'threshold':p['threshold'],'variants':{},'training':{},'frozen_unchanged':True,
        'scope':'Final checkpoints only; original executed C0/C1/W0-or66 labels and costs retained. Model-only coverage and selection here; live fallback is reported separately. Matrix costs are not zero-wait live rollouts.'}
    for variant,directory in paths.items():
        train=old.read(directory/'train-after-predictions.jsonl');dev=old.read(directory/'after-predictions.jsonl')
        report['variants'][variant]={}
        for name,rows in [('old_train',[r for r in train if r['seed']<48]),
            ('new_train',[r for r in train if r['seed']>=48]),('all_train',train),('development',dev)]:
            report['variants'][variant][name]={}
            for wait in (None,0,66):
                group=rows if wait is None else [r for r in rows if r['wait_steps']==wait]
                report['variants'][variant][name]['all' if wait is None else f'W{wait}']=old.selection_metrics(group,p['threshold'],physics)
        report['variants'][variant]['fixed60_development']={f'W{w}':old.selection_metrics(
            [r for r in dev if r['wait_steps']==w],p['threshold'],physics,fixed_n=60) for w in (0,66)}
        if variant=='previous':continue
        result=json.loads((directory/'result.json').read_text());protocol=json.loads((directory/'protocol.json').read_text())
        training=old.read(directory/'training.jsonl')
        assert len(training)==result['completed_steps']==512
        assert sum(len(r['example_indices']) for r in training)==1024
        assert Counter(r['sample_kind'] for r in training)==Counter(duration_family=256,uniform_single=256)
        assert protocol['init_adapter']==p['initial_outcome_adapter'] and protocol['final_checkpoint_evaluation_only']
        assert protocol['sampling_design']['source_presentations']=={'old':512,'new':512}
        assert result['frozen_start_unchanged'] and result['data_unchanged']
        report['training'][variant]={k:result[k] for k in ('completed_steps','training_seconds','peak_allocated_gpu_bytes','frozen_start_unchanged','data_unchanged')}
        report['training'][variant]['prefix_gradient_norm']=training[0]['first_prefix_gradient_norm']
    a=old.read(ROOT/'set512/sampling-plan.jsonl');b=old.read(ROOT/'candidate512/sampling-plan.jsonl')
    assert a==b
    report['identical_sampling_plans']=True
    hashes=json.loads((ROOT/'frozen-inputs-before.json').read_text())
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    report['paired_matrix_costs']={}
    for w in ('W0','W66'):
        model=report['variants']['set']['development'][w]
        report['paired_matrix_costs'][w]=old.paired_costs(report['variants']['set']['fixed60_development'][w],model)
    (ROOT/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    keys=['safe_candidates','safe_admitted_candidates','false_admitted_candidates','recoverable_families',
        'safe_covered_families','safe_selected_families','no_safe_families','no_safe_families_rejected']
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['variant','split','wait']+keys);writer.writeheader()
        for variant,by_split in report['variants'].items():
            for split,by_wait in by_split.items():
                if split=='fixed60_development':continue
                for wait,m in by_wait.items():writer.writerow(dict(variant=variant,split=split,wait=wait,**{k:m[k] for k in keys}))
    print(json.dumps({v:{s:{w:{k:m[k] for k in keys} for w,m in ms.items()} for s,ms in ds.items()
        if s in ('old_train','new_train','development')} for v,ds in report['variants'].items()}))


if __name__=='__main__':main()
