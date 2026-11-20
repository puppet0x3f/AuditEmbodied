import argparse
from collections import Counter
import json
from pathlib import Path
import re

import h5py
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--roots',nargs='+',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
proposed=set(range(50000,50020));seen=set();counts=Counter();errors=[];hits=[];recovered=[]
scalar=re.compile(r'"(?:seed|seed_start|random_seed|rng_seed)"\s*:\s*(-?\d+)')
arrays=re.compile(r'"(?:seeds|train_seeds|dev_seeds|test_seeds|initialized_seeds)"\s*:\s*\[([^\]]*)\]')

def from_text(text,path):
    values={int(v) for v in scalar.findall(text)}
    for item in arrays.findall(text):values.update(int(v) for v in re.findall(r'-?\d+',item))
    seen.update(values)
    if values & proposed:hits.append({'path':str(path),'seeds':sorted(values & proposed)})

for root in args.roots:
    root=Path(root)
    if not root.exists():continue
    for path in root.rglob('*'):
        if 'p1-load-transfer-20261109' in path.parts or not path.is_file():continue
        suffix=path.suffix.lower()
        if suffix not in ('.json','.jsonl','.h5','.hdf5','.npz'):continue
        try:
            if suffix in ('.json','.jsonl'):
                # Scan complete and interrupted records without loading large traces.
                with path.open(errors='replace') as stream:
                    for line in stream:from_text(line,path)
                # Pretty-printed seed arrays span lines; metadata JSON is small.
                if suffix=='.json':from_text(path.read_text(errors='replace'),path)
            elif suffix in ('.h5','.hdf5'):
                with h5py.File(path,'r') as f:
                    def visit(name,obj):
                        for key,value in obj.attrs.items():
                            if 'seed' in key.lower():
                                for v in np.asarray(value).ravel():
                                    try:seen.add(int(v))
                                    except (ValueError,TypeError):pass
                            if isinstance(value,(str,bytes)):from_text(str(value),path)
                        if isinstance(obj,h5py.Dataset) and name.rsplit('/',1)[-1] in ('seed','seeds'):
                            seen.update(int(v) for v in np.asarray(obj).ravel())
                    visit('',f);f.visititems(visit)
            else:
                with np.load(path,allow_pickle=False) as f:
                    for name in ('seed','seeds','train_seeds','dev_seeds','test_seeds'):
                        if name in f:seen.update(int(v) for v in np.asarray(f[name]).ravel())
            counts[suffix]+=1
        except Exception as e:
            match=re.match(r'^(\d+)-',path.name)
            if suffix=='.npz' and path.stat().st_size==0 and match:
                seed=int(match.group(1));seen.add(seed);counts['empty_npz_named_seed']+=1
                recovered.append({'path':str(path),'seed':seed,'basis':'zero-byte historical observation file; explicit seed in filename'})
            else:errors.append({'path':str(path),'error':repr(e)})
r={'proposed_test_seeds':sorted(proposed),'roots':args.roots,'files_scanned':dict(counts),
   'previously_seen_seed_count':len(seen),'previously_seen_seeds':sorted(seen),
   'overlap':sorted(proposed & seen),'matching_records':hits,'errors':errors,'recovered_empty_files':recovered,
   'fresh':not(proposed & seen) and not errors}
Path(args.output).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ('previously_seen_seeds','matching_records')}))
