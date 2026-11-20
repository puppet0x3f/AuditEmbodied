"""Reanalyse already frozen outcomes; no new physical executions or score changes."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'outputs/hold-components-vision-20261111'
sys.path.insert(0,str(ROOT/'reproducibility'));sys.path.insert(0,str(HERE))
from paired_holding_audit import diagnostics,transition


def main():
    result={'new_physical_executions':0,'cohorts':{}};output=[]
    for phase in ('mechanism','independent'):
        scored=[json.loads(x) for x in (OLD/(phase+'-analysis/scored.jsonl')).read_text().splitlines()]
        by_key={(r['seed'],r['scenario'],r['hold_mode'],r['wait_steps']):r for r in scored}
        for path in sorted((OLD/phase).glob('*.json')):
            raw=json.loads(path.read_text());key=(raw['seed'],raw['scenario'],raw['hold_mode'],raw['wait_steps'])
            r=by_key[key];r.update(diagnostics(raw,path.parent/raw['native_file']));output.append(r)
        method=scored[0]['method'];groups={};changes={}
        for hold in sorted({r['hold_mode'] for r in scored}):
            changes[hold+'_wait']=transition(scored,(method,hold,1),(method,hold,20))
            if hold!='registered':changes[hold+'_long_gain']=transition(scored,(method,'registered',20),(method,hold,20))
            rows=[r for r in scored if r['scenario']!='sham' and r['hold_mode']==hold and r['wait_steps']==20]
            groups[hold]={'episodes':len(rows)}
            for field in ('native_Y','total','hold_entry_position_offset_mm','hold_mean_position_error_mm',
                'wait_translation_mm','wait_end_speed_m_s','setdown_tilt_deg','first_close_dual','first_close_miss_mm',
                'preclose_rgb_target_miss_mm','preclose_observation_to_close_s'):
                values=[r[field] for r in rows if r.get(field) is not None]
                groups[hold][field]=float(np.mean(values)) if values else None
                groups[hold][field+'_n']=len(values)
        result['cohorts'][phase]={'method':method,'long_wait_diagnostics':groups,'paired_transitions':changes}
    (HERE/'prior-mechanism.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'prior-diagnostics.jsonl').open('x') as f:
        for r in output:f.write(json.dumps(r,separators=(',',':'))+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
