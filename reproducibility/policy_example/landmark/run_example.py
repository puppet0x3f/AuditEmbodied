"""Run the frozen CAD-free visual controller in a new output directory."""
import argparse
from pathlib import Path
from vision_campaign import one


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=90)
    p.add_argument('--scenario',choices=('sham','left_loss','right_loss'),default='sham')
    p.add_argument('--hold',choices=('registered','measured_pose'),default='measured_pose')
    p.add_argument('--steps',type=int,choices=(1,20),default=1)
    a=p.parse_args();assert not a.out.exists(),'Use a new output directory'
    status=one((a.seed,a.scenario,a.steps,a.hold,str(a.out.resolve()/'independent')))
    assert not status['error'] and status['nominal_ready'] and status['fault_valid'],status
    print('Recorded complete execution and native guards in',a.out.resolve()/'independent')
