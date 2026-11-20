"""Eight fixed known-state recordings; separate processes preserve source versions."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def run(case):
    command=[sys.executable,str(ROOT/'replay.py')]
    for key,value in case.items():command += ['--'+key,str(value)]
    name='-'.join(str(case[k]) for k in ('cohort','seed','scenario','method'))
    with (ROOT/f'{name}.log').open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps({'case':name,'exit_code':result.returncode}),flush=True)
    assert result.returncode==0,name


if __name__=='__main__':
    cases=json.loads((ROOT/'protocol.json').read_text())['frequency_diagnostic']['cases']
    with ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(run,cases))
