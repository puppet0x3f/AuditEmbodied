"""Read seed metadata only; skip this reserved study directory."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('roots',nargs='+');args=parser.parse_args()
    wanted=set(range(40000,40020));hits=[];errors=[];counts={'json':0,'hdf5':0,'npz':0}
    def inspect(value,path,key=''):
        if 'seed' in key.lower():
            if isinstance(value,(list,tuple,np.ndarray)):
                for item in value:inspect(item,path,key)
            elif isinstance(value,dict):
                for k,v in value.items():inspect(v,path,k)
            else:
                try:
                    seed=int(value)
                    if seed in wanted:hits.append({'path':str(path),'key':key,'seed':seed})
                except (TypeError,ValueError,OverflowError):pass
        elif isinstance(value,dict):
            for k,v in value.items():
                if k not in ('trace','qpos','qvel','ctrl','act','weights','mean','scale','robot','evaluator'):
                    inspect(v,path,k)
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,(dict,list)):inspect(item,path)
        elif isinstance(value,(str,bytes)) and value[:1] in ('{','['):
            try:inspect(json.loads(value),path)
            except (ValueError,TypeError):pass
    for root in args.roots:
        for path in Path(root).rglob('*'):
            if 'handover-audit-20261109' in path.parts or not path.is_file():continue
            try:
                if path.suffix in ('.json','.jsonl'):
                    counts['json']+=1
                    with path.open() as stream:
                        if path.suffix=='.json':inspect(json.load(stream),path)
                        else:
                            for line in stream:
                                if line.strip():inspect(json.loads(line),path)
                elif path.suffix in ('.h5','.hdf5'):
                    counts['hdf5']+=1
                    with h5py.File(path,'r') as file:
                        def visit(name,item):
                            for key,value in item.attrs.items():inspect(value,path,str(key))
                            if isinstance(item,h5py.Dataset) and 'seed' in name.lower():inspect(item[()],path,name)
                        visit('',file);file.visititems(visit)
                elif path.suffix=='.npz':
                    counts['npz']+=1
                    with np.load(path,allow_pickle=False) as archive:
                        for key in archive.files:
                            if 'seed' in key.lower():inspect(archive[key],path,key)
            except Exception as error:errors.append({'path':str(path),'error':repr(error)})
    print(json.dumps({'roots':args.roots,'scanned':counts,'occurrences':hits,'used_overlap':sorted({r['seed'] for r in hits}),'errors':errors}))


if __name__=='__main__':main()
