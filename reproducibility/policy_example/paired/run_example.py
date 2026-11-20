"""Execute either frozen visual policy under a selected holding component."""
import argparse
from pathlib import Path
from campaign import one


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=90)
    p.add_argument('--scenario',choices=('sham','left_loss','right_loss'),default='sham')
    p.add_argument('--method',choices=('closed_loop','visual_servo','fixed_touchdown'),default='visual_servo')
    p.add_argument('--hold',choices=('registered','measured_position','measured_rotation','measured_pose'),default='measured_position')
    p.add_argument('--steps',type=int,choices=(1,20),default=20)
    a=p.parse_args();assert not a.out.exists(),'Use a new output directory'
    folder=a.out.resolve()/'execution'
    result=one((a.seed,a.scenario,a.steps,a.hold,a.method,str(folder)))
    assert not result['error'] and result['nominal_ready'] and result['fault_valid'],result
    print('Recorded complete physical execution and native guards in',folder)
