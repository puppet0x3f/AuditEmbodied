"""Execute a single physical example with explicit holding and supplied CAD."""
import argparse
from pathlib import Path
from campaign import one


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--geometry',choices=('pot','tray'),default='tray')
    p.add_argument('--hold',choices=('registered','measured_pose'),default='registered')
    p.add_argument('--steps',type=int,choices=(1,5,10,20),default=1)
    p.add_argument('--scenario',choices=('left_loss','right_loss','sham'),default='sham')
    p.add_argument('--method',choices=('retarget','closed_loop'),default='closed_loop')
    p.add_argument('--seed',type=int,default=70)
    a=p.parse_args();assert not a.out.exists(),'Use a new output directory'
    status=one((a.seed,a.scenario,a.method,a.steps,a.hold,a.geometry,str(a.out.resolve())))
    assert not status['error'] and status['nominal_ready'] and status['fault_valid'],status


if __name__=='__main__':main()
