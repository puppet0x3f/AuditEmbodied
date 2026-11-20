"""Counterbalanced four-method independent campaign; no test adaptation."""
from collections import Counter,defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from auditembodied.duration_aligned_loop import read_response
from auditembodied.duration_family_loop import family_trial
from auditembodied.l0 import _import_robosuite
from auditembodied.l4_lift import LiftConfig
from auditembodied.p0 import same_prefix
from dataclasses import asdict

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering',action='store_true')
    args=parser.parse_args()
    p=json.loads((ROOT/'protocol.json').read_text())
    calibration=json.loads((ROOT/'calibration.json').read_text())
    audit=json.loads((ROOT/'seed-audit.json').read_text())
    assert not audit['used_overlap']
    seeds=[60] if args.engineering else audit['requested_seeds']
    assert args.engineering or len(seeds)==20
    methods=['fixed60','ce','terminal','joint']
    episodes=len(seeds)*2*len(methods)
    output=ROOT/('engineering' if args.engineering else 'closed-loop')
    assert not output.exists(),'never overwrite observed physical episodes'
    (output/'observations').mkdir(parents=True)
    for method in methods:(output/method/'observations').mkdir(parents=True)
    frozen={**p['frozen_inputs'],str(ROOT/'calibration.json'):sha(ROOT/'calibration.json'),
        str(ROOT/'supervision_worker.py'):sha(ROOT/'supervision_worker.py'),
        str(ROOT/'campaign.py'):sha(ROOT/'campaign.py')}
    for variant,info in calibration.items():
        for name,digest in info['adapter_sha256'].items():
            path=Path(info['adapter'])/name;assert sha(path)==digest;frozen[str(path)]=digest
    schedule=[]
    for i,seed in enumerate(seeds):
        for j,scenario in enumerate(('left_loss','right_loss')):
            shift=(i+j)%4
            schedule.append({'seed':seed,'scenario':scenario,'order':methods[shift:]+methods[:shift]})
    protocol={'scope':'four frozen policies, fresh seeds, same scope head and physical timing; no test refit',
        'test_seeds':seeds,'methods':methods,'episodes':episodes,'schedule':schedule,'engineering_only':args.engineering,
        'scope_threshold':p['scope_threshold'],'outcome_calibration':calibration,
        'config':asdict(LiftConfig()),'lookahead_steps':10,'paced_recovery':True,
        'gpu':'1','seed_audit':audit,'frozen_inputs':frozen,
        'runtime_source_sha256':p['source_sha256'],
        'source_directory':str(ROOT/'frozen-source'),
        'cost':'measured acquisition, queueing, decision holds, controller overruns, all skills and final1s hold',
        'method_first_counts':dict(Counter(row['order'][0] for row in schedule))}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT/'frozen-source'))
    command=['/opt/miniconda3/bin/python',str(ROOT/'supervision_worker.py'),
        '--working-point',p['working_point'],'--model-path',p['model_path'],
        '--post-wait-adapter',calibration['joint']['adapter'],'--post-wait-mode','family_stream',
        '--family-control-adapter',calibration['ce']['adapter'],'--family-terminal-adapter',calibration['terminal']['adapter']]
    adapter_names={'joint':'post_wait','ce':'family_control','terminal':'family_terminal'}
    rows=[]
    with (output/'worker.log').open('x') as log:
        worker=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log,text=True,bufsize=1)
        try:
            line=worker.stdout.readline()
            if not line:raise RuntimeError('worker failed at startup; inspect worker.log')
            ready=json.loads(line);assert ready.get('ready'),ready
            warm=json.loads((ROOT/'dataset/train.jsonl').read_text().splitlines()[0])
            for method in ('ce','terminal','joint'):
                messages=[{'op':'family_prefill','example':warm,'adapter_name':adapter_names[method]},
                    {'op':'family_tail','example':warm,'attempt_id':0,
                     'plans':[{**warm['plan'],'local_relevel_steps':n} for n in (30,40,60)],
                     'score_mode':calibration[method]['score_mode'],'admission_threshold':calibration[method]['threshold']},
                    {'op':'family_release'}]
                for message in messages:
                    worker.stdin.write(json.dumps(message)+'\n');worker.stdin.flush();read_response(worker)
            _import_robosuite(True)
            with (output/'trials.jsonl').open('x') as stream:
                for state in schedule:
                    group=[]
                    for method in state['order']:
                        info=calibration.get(method,{})
                        row=family_trial(state['seed'],state['scenario'],output/method,worker,baseline=method=='fixed60',
                            adapter_name=adapter_names.get(method,'post_wait'),
                            score_mode=info.get('score_mode','safe'),admission_threshold=info.get('threshold'))
                        row.update(method=method,wall_seconds=row['episode_wall_seconds'])
                        group.append(row);rows.append(row)
                        stream.write(json.dumps(row)+'\n');stream.flush()
                        print(json.dumps({'completed':len(rows),'total':episodes,
                            **{k:row.get(k) for k in ('seed','scenario','method','success','protection_violation','total_post_fault_seconds','error')},
                            'duration':row.get('local_steps',{}).get('local_relevel_steps')}),flush=True)
                    assert same_prefix(group)
        finally:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                except BrokenPipeError:pass
                worker.wait(timeout=30)
    assert len(rows)==episodes
    assert all(sha(path)==digest for path,digest in frozen.items())
    assert all(not row.get('error') for row in rows)
    (output/'complete.json').write_text(json.dumps({'episodes':episodes,'frozen_unchanged':True,'worker_setup':ready},indent=2)+'\n')


if __name__=='__main__':main()
