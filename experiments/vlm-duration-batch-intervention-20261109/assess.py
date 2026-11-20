"""Final-only old/new/dev comparisons using actually executed candidate costs."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=json.loads((ROOT/'experiment-plan.json').read_text())
    spec=importlib.util.spec_from_file_location('previous_assessment',Path(p['previous_round'])/'assess.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    physics=old.load_physics(p)
    for r in old.read(ROOT/'collection/trials.jsonl'):
        n=r['local_steps']['local_relevel_steps'];w=r['post_prefix_wait_steps']
        key=r['seed'],r['scenario'],w,n
        physics[key]={'label':('B' if r['protection_violation'] else 'A') if r['success'] else ('D' if r['protection_violation'] else 'C'),
            'post_fault_seconds':r['total_post_fault_seconds'],'episode_simulated_seconds':r['episode_simulated_seconds'],
            'wait_seconds':r['inference_wait_seconds'],'skill_seconds':r['skill_seconds'],'final_hold_seconds':r['hold_seconds'],
            'decision_prefix_seconds':r['stage_capture']['h2']['state']['time']-r['prefix_state']['time']+w/20,
            'path_m':r['recovery_path_m']}
    train=old.read(ROOT/'mixed512/train-after-predictions.jsonl')
    dev=old.read(ROOT/'mixed512/after-predictions.jsonl')
    baseline_train=old.read(Path(p['previous_round'])/'full512/train-after-predictions.jsonl')
    baseline_dev=old.read(Path(p['previous_round'])/'full512/after-predictions.jsonl')
    metrics={}
    for name,rows in [('old_train_before',baseline_train),('old_train_after',[r for r in train if r['seed']<48]),
        ('new_train_after',[r for r in train if r['seed']>=48]),('all_train_after',train),
        ('dev_before',baseline_dev),('dev_after',dev)]:
        metrics[name]={'all':old.selection_metrics(rows,p['threshold'],physics)}
        for w in (0,66):
            metrics[name][f'W{w}']=old.selection_metrics([r for r in rows if r['wait_steps']==w],p['threshold'],physics)
    metrics['fixed60_dev_W66']=old.selection_metrics([r for r in dev if r['wait_steps']==66],p['threshold'],physics,fixed_n=60)
    training=old.read(ROOT/'mixed512/training.jsonl');result=json.loads((ROOT/'mixed512/result.json').read_text())
    protocol=json.loads((ROOT/'mixed512/protocol.json').read_text())
    assert len(training)==result['completed_steps']==512
    assert protocol['init_adapter']==p['initial_outcome_adapter'] and protocol['final_checkpoint_evaluation_only']
    assert protocol['sampling_design']['source_presentations']=={'old':512,'new':512}
    assert result['frozen_start_unchanged'] and result['data_unchanged']
    hashes=json.loads((ROOT/'frozen-inputs-before.json').read_text())
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
    report={'metrics':metrics,'training':result,'same_threshold':p['threshold'],'frozen_unchanged':True,
        'old_train_cost_change':old.paired_costs(metrics['old_train_before']['all'],metrics['old_train_after']['all']),
        'dev_cost_change':old.paired_costs(metrics['dev_before']['W66'],metrics['dev_after']['W66']),
        'scope':'Final512 only. Old72 and new96 train; fixed development24 unchanged. Matrix outcomes/costs are measured with supplied repair roles. Separate live report supplies autonomous held-out evidence. Refused matrix states have observed prefix cost, without invented terminal outcomes.'}
    (ROOT/'assessment.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['split','W','safe_candidates','safe_admitted_candidates','false_admitted_candidates','recoverable_families',
        'safe_covered_families','safe_selected_families','selected_families','no_safe_families','no_safe_families_rejected','mean_selected_post_fault_seconds']
    with (ROOT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for split,by_wait in metrics.items():
            if split=='fixed60_dev_W66':continue
            for w,m in by_wait.items():
                record={k:m[k] for k in columns[2:-1]}
                record.update(split=split,W=w,mean_selected_post_fault_seconds=m['costs']['mean_selected_post_fault_seconds'])
                writer.writerow(record)
    print(json.dumps({k:{w:{f:m[f] for f in ('safe_admitted_candidates','false_admitted_candidates',
        'safe_covered_families','safe_selected_families','no_safe_families_rejected')} for w,m in v.items()}
        for k,v in metrics.items() if k!='fixed60_dev_W66'}),flush=True)


if __name__=='__main__':main()
