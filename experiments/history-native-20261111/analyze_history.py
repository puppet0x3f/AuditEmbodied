"""Native-frequency and all-phase checks of the ten unchanged old B paths."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def longest(flags):
    n=best=0
    for v in flags:n=n+1 if v else 0;best=max(n,best)
    return best

def score(case):
    row=json.loads((ROOT/'source-cases'/f"{case['name']}.json").read_text())
    meta=json.loads((ROOT/'native'/f"{case['name']}.json").read_text())
    assert meta['all_original_frames_match'] and meta['maximum_original_observation_error']==0 and not meta['error']
    with np.load(ROOT/'native'/f"{case['name']}.npz") as f:all_values=f['values'];stage_names=list(f['stages'])
    a=all_values[all_values[:,1]>=2];missing=~(a[:,3].astype(bool)|a[:,4].astype(bool)|a[:,5].astype(bool))
    assert len(a)==25*sum(f['phase'] in ('recovery','final_hold') for f in row['trace'])
    gap=longest(missing)*.002
    start=next(f['time']-.05 for f in row['trace'] if f['stage']=='global_hover')
    suffix=a[a[:,0]>start+1e-8]
    tail_gap=longest(~(suffix[:,3].astype(bool)|suffix[:,4].astype(bool)|suffix[:,5].astype(bool)))*.002
    phase=[]
    for offset in range(25):
        b=a[offset::25];flags=~(b[:,3].astype(bool)|b[:,4].astype(bool)|b[:,5].astype(bool))
        phase.append(bool(longest(flags)<=2 and not np.any(b[:,11]>.01)))
    assert not phase[24]
    native_ok=gap<=.100000001 and not np.any(a[:,11]>.01)
    force=a[missing,12]
    return {**case,'primary_label':'B','native_Y':bool(native_ok),
        'native_gap_ms':gap*1000,'native_suffix_gap_ms':tail_gap*1000,
        'native_history_only':bool(not native_ok and tail_gap<=.100000001 and not np.any(suffix[:,11]>.01)),
        'phase_Y':phase,'phase_Y_count':sum(phase),'native_recorded_samples':len(all_values),
        'native_post_fault_samples':len(a),'minimum_force_on_missing_N':float(force.min()) if len(force) else None,
        'zero_force_gap_ms':longest(missing&(a[:,12]<1e-6))*.002*1000,
        'primary_wait_seconds':row['inference_wait_seconds'],'primary_total_seconds':row['total_post_fault_seconds'],
        'all_observations_match':True,'maximum_observation_error':0.}

def main():
    manifest=json.loads((ROOT/'cases.json').read_text());rows=[score(c) for c in manifest['cases']]
    result={'diagnostic_replays':len(rows),'new_independent_policy_trials':0,'unique_fault_states':len({(r['seed'],r['scenario']) for r in rows}),
        'initialization_seeds':len({r['seed'] for r in rows}),
        'all_original_observations_match':True,'maximum_original_observation_error':0.,
        'primary_B_cases':len(rows),'native_Y_cases':sum(r['native_Y'] for r in rows),
        'native_history_only_cases':sum(r['native_history_only'] for r in rows),
        'native_samples':sum(r['native_recorded_samples'] for r in rows),'rows':rows}
    (ROOT/'history-report.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'history-cases.csv').open('w') as f:
        names=['seed','scenario','window','primary_gap_ms','native_gap_ms','native_suffix_gap_ms','native_Y','native_history_only','phase_Y_count','minimum_force_on_missing_N']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows({k:r[k] for k in names} for r in rows)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
