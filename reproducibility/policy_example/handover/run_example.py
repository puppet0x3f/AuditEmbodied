"""Execute a new handover holding condition with the frozen physical policies."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','LP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import campaign


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=91)
    p.add_argument('--method',choices=('rule','R3'),default='rule')
    p.add_argument('--hold',choices=('registered','measured_position','smooth_position'),default='smooth_position')
    p.add_argument('--wait-seconds',type=float,choices=(.05,1.),default=1.)
    p.add_argument('--scenario',choices=('nominal','object_sham','object_rotation'),default='object_rotation')
    a=p.parse_args();out=a.out.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
    n=round(a.wait_seconds/.05)
    campaign.one((a.seed,a.scenario,n,a.hold,a.method,str(out)))
    spec=importlib.util.spec_from_file_location('example_offline',HERE.parents[1]/'recompute.py')
    offline=importlib.util.module_from_spec(spec);spec.loader.exec_module(offline)
    spec=importlib.util.spec_from_file_location('handover_holding_audit',HERE.parents[1]/'handover_holding_audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    name=f'{a.seed}-{a.scenario}-{n:02d}-{a.hold}-{a.method}'
    row=json.loads((out/(name+'.json')).read_text())
    scored=audit.score(row,out/row['native_file'],offline)
    (out/'example-audit.json').write_text(json.dumps(scored,indent=2)+'\n')
    print(json.dumps({'C':scored['C'],'Y20':scored['Y'],'Y500':scored['native_Y'],
        'complete_seconds':scored['total'],'native_samples':scored['native_samples'],
        'scope':'new development example, excluded from all formal denominators'}))


if __name__=='__main__':main()
