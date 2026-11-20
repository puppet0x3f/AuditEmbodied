"""Family-level offline decisions; physical branch costs are not neural timings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .mid_recovery_data import read


def family_metrics(rows):
    grouped={}
    for r in rows:grouped.setdefault((r['seed'],r['scenario'],r['source_wait_steps']),{})[r['decision_step']]=r
    methods={'single':[],'two':[]}
    for key,pair in grouped.items():
        first=pair[30];second=pair[40];truth=first['all_labels']
        def branch(action,step):
            return {'continue':'LOCAL60','R_global':f'GLOBAL{step}','defer':'GLOBAL40'}.get(action)
        b1=branch(first['action'],30)
        last=second if first['action'] in ('continue','defer') else first
        b2=branch(last['action'],last['decision_step'])
        for name,selected,decision in [('single',b1,first),('two',b2,last)]:
            methods[name].append({'seed':key[0],'scenario':key[1],'wait':key[2],'labels':truth,
                'selected':selected,'safe':selected is not None and truth[selected]=='A',
                'model_admitted':decision['model_admitted'],'fallback':decision['baseline_fallback']})
    reports={}
    for name,cases in methods.items():
        recoverable=[r for r in cases if 'A' in r['labels'].values()]
        needed=[r for r in cases if r['labels']['LOCAL60']!='A' and 'A' in r['labels'].values()]
        reports[name]={'physical_states':len(cases),'recoverable_states':len(recoverable),
            'safe_actual_choices':sum(r['safe'] for r in cases),'safe_state_coverage':sum(r['safe'] for r in recoverable),
            'model_admitted_safe_choices':sum(r['safe'] and r['model_admitted'] and not r['fallback'] for r in cases),
            'model_selected_unsafe_choices':sum(r['selected'] is not None and not r['safe'] and r['model_admitted'] and not r['fallback'] for r in cases),
            'needed_upgrades':len(needed),'needed_upgrade_misses':sum(not r['safe'] for r in needed),
            'unnecessary_upgrades':sum(r['selected'] in ('GLOBAL30','GLOBAL40') and r['labels']['LOCAL60']=='A' for r in cases),
            'no_safe_states':sum('A' not in r['labels'].values() for r in cases),
            'no_safe_rejected':sum('A' not in r['labels'].values() and r['selected'] is None for r in cases),
            'safe_states_rejected':sum('A' in r['labels'].values() and r['selected'] is None for r in cases),
            'fallback_choices':sum(r['fallback'] for r in cases),'branches':dict(Counter(r['selected'] for r in cases))}
    timing=[r for r in rows if r['decision_step']==30 and (r['labels']['global']=='A')!=(r['labels']['defer']=='A')]
    correct=0
    for r in timing:
        p={x['key']:x['class_probabilities'][0] for x in r['records']}
        correct+=(p['global']>p['defer'])==(r['labels']['global']=='A')
    return {'states':len(grouped),'policies':reports,'timing_safety_contrasts':len(timing),'timing_contrasts_ranked_correctly':correct}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);a=p.parse_args();root=Path(a.model_dir)
    report={split:family_metrics(read(root/f'{split}-predictions.jsonl')) for split in ('train','development')}
    (root/'family-analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
