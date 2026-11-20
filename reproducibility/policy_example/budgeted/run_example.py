"""Execute the separately frozen arrival-adaptive CAD policy."""
import argparse
from pathlib import Path
from campaign import one


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=94)
    p.add_argument('--scenario',choices=('sham','left_loss','right_loss'),default='sham')
    p.add_argument('--method',choices=('robust_cad',),default='robust_cad')
    p.add_argument('--hold',choices=('registered','measured_position'),default='measured_position')
    p.add_argument('--steps',type=int,choices=(1,20),default=20)
    a=p.parse_args();assert not a.out.exists(),'Use a new output directory'
    folder=a.out.resolve()/'execution'
    result=one((a.seed,a.scenario,a.steps,a.hold,a.method,str(folder)))
    assert not result['error'] and result['nominal_ready'] and result['fault_valid'],result
    print('Recorded complete physical execution and native guards in',folder)
