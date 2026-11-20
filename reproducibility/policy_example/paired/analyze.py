"""Score shared-prefix executions without modifying their outcomes or thresholds."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'reproducibility'))
import controller_audit as controller
spec=importlib.util.spec_from_file_location('offline',ROOT/'reproducibility/recompute.py')
offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
sys.path.insert(0,str(HERE))
import geometry as cad
from visual_servo import estimate as landmark_estimate
import paired_holding_audit as audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--refit',action='store_true');a=p.parse_args()
    raw=[];rows=[];fits=landmark_fits=0
    cad.configure('pot')
    for path in sorted((HERE/a.phase).glob('*.json')):
        r=json.loads(path.read_text());raw.append(r)
        assert r['nominal_ready'] and r['fault_class_valid'] and not r.get('error'),path.name
        assert not r['blocked_calls'] and r['contact_anchor_count']==r['state_edit_count_after_initialization']==0,path.name
        scored,count=audit.score(r,path.parent/r['native_file'],offline.score_lift,controller,
            cad.fit if a.refit and r['method']!='visual_servo' else None)
        rows.append(scored);fits+=count
        if a.refit and r['method']=='visual_servo':
            for d in r['global_execution']['observations']:
                estimate=landmark_estimate(d['input']);landmark_fits+=1
                assert estimate['valid']==d['estimate']['valid']
                if estimate['valid']:
                    for key in ('center','targets','rotation','rotation_targets','landmarks','reference_landmarks'):
                        assert np.allclose(estimate[key],d['estimate'][key],atol=1e-9,rtol=0),key
    report=audit.summarize(rows)
    report.update(audit.check_pairing(raw),pose_fits_recomputed=fits,landmark_estimates_recomputed=landmark_fits)
    if a.phase=='paired':
        cad_effect=report['effects']['closed_loop']['position_main_wait']['native_Y']
        landmark_effect=report['effects']['visual_servo']['position_main_wait']['native_Y']
        difference=report['effects']['executor_position_wait']['native_Y']
        report['placement_trigger']=bool(cad_effect['ci95'][0]>0 and difference['ci95'][0]>0 and landmark_effect['ci95'][0]<=0)
    if a.phase=='placement':
        baseline=[json.loads(s) for s in (HERE/'paired-analysis/scored.jsonl').read_text().splitlines()]
        baseline=[r for r in baseline if r['method']=='closed_loop' and r['hold_mode'] in ('registered','measured_position')]
        report['matched_baseline_episodes']=len(baseline)
        report['effects']=audit.effects(rows+baseline)
    out=HERE/(a.phase+'-analysis');out.mkdir(exist_ok=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'scored.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    (out/('paired-placement.tex' if 'placement' in a.phase else 'paired-holding.tex')).write_text(audit.table(report))
    print(json.dumps({k:report[k] for k in ('episodes','native_samples','matched_prehold_groups','matched_posthold_method_pairs')}))
    print(json.dumps(report['effects'],indent=2));print('placement_trigger',report.get('placement_trigger'))


if __name__=='__main__':main()
