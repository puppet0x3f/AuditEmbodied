"""Append pilot measurements and ten diagnostic replays without rewriting old data."""
import gzip,importlib.util,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];DEST=REPO/'reproducibility'
PILOT=REPO/'outputs/cross-controller-20261110'

def native_projection(source,target):
    with np.load(source,allow_pickle=False) as raw:
        a=raw['values'];a=a[a[:,1]>=2]
    left=a[:,3].astype(bool);right=a[:,4].astype(bool);table=a[:,5].astype(bool)
    missing=~(left|right|table)
    # Unrounded native atoms and depths; force norms are needed only on gaps.
    np.savez_compressed(target,time=a[:,0],stage=a[:,2].astype(np.int32),left=left,right=right,table=table,
        interarm_penetration_m=a[:,11],force_measured=missing,
        object_contact_force_norm_N=np.where(missing,a[:,12],0.))

def main():
    spec=importlib.util.spec_from_file_location('export_core',DEST/'export_from_workspace.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    native_dir=DEST/'data/pilot-native';native_dir.mkdir(exist_ok=False)
    count=0
    with (PILOT/'trials.jsonl').open() as src,gzip.open(DEST/'data/controller_pilot.jsonl.gz','wt') as dst:
        for line in src:
            row=json.loads(line);small=core.project(row,'controller_pilot')
            path=Path(row['native_recording']['path'])
            if not path.is_absolute():path=REPO/path
            native_projection(path,native_dir/small['native_file'])
            dst.write(json.dumps(small,separators=(',',':'))+'\n');count+=1
    assert count==480
    manifest=json.loads((ROOT/'cases.json').read_text())
    history_dir=DEST/'data/history-native';history_dir.mkdir(exist_ok=False)
    for case in manifest['cases']:
        case['native_file']=case['name']+'.npz'
        native_projection(ROOT/'native'/case['native_file'],history_dir/case['native_file'])
    (DEST/'data/history-cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(ROOT/'history-report.json',DEST/'reference/history-native.json')
    shutil.copy2(PILOT/'report.json',DEST/'reference/pilot-results.json')
    cfg=json.loads((DEST/'data/config.json').read_text())
    cfg['episodes']['controller_pilot']=480
    cfg['source_files']['controller_pilot']='outputs/cross-controller-20261110/trials.jsonl'
    cfg['diagnostic_history_replays']={'cases':10,'new_independent_policy_trials':0,
        'source':'interface','manifest':'history-cases.json','scope':'Original ten61000-series retarget B paths; all20Hz observations reproduced exactly before projection.'}
    (DEST/'data/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'new_policy_executions':480,'total_executions':sum(cfg['episodes'].values()),
        'diagnostic_replays_not_added_to_denominator':10,'native_projected_files':490}),flush=True)

if __name__=='__main__':main()
