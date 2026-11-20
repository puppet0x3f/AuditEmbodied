"""Attach the unchanged prerequisite model's h0 scores to physical samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .duration_aligned_loop import read_response
from .mid_recovery_data import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--working-point',required=True);p.add_argument('--model-path',required=True)
    a=p.parse_args();data=Path(a.data);cache={};reports={}
    with (data/'start-worker.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,'-m','auditembodied.vlm_duration_policy','--working-point',a.working_point,'--model-path',a.model_path],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        try:
            ready,_=read_response(worker);assert ready['ready'],ready
            threshold=ready.get('threshold',json.loads(Path(a.working_point).read_text())['threshold'])
            for split in ('train','development'):
                rows=read(data/f'{split}.jsonl')
                for i,row in enumerate(rows):
                    key=json.dumps([row['before_image'],row['before_robot'],row['plan']['repair_arm']],sort_keys=True)
                    if key not in cache:
                        worker.stdin.write(json.dumps({'op':'cache_start','image':row['before_image'],'robot':row['before_robot'],'plan':row['plan']})+'\n');worker.stdin.flush()
                        response,_=read_response(worker);cache[key]=response['start_valid_score']
                    row['start_valid_score']=cache[key]
                    if i%16==0:print(json.dumps({'split':split,'scored':i+1,'q0':row['start_valid_score']}),flush=True)
                (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                reports[split]={'window_examples':len(rows),'prerequisite_passed':sum(r['start_valid_score']>=threshold for r in rows)}
        finally:
            if worker.poll() is None:
                try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
                except BrokenPipeError:pass
                worker.wait(timeout=30)
            (data/'start-worker-exit.json').write_text(json.dumps({'exit_code':worker.poll()})+'\n')
    (data/'start-score-report.json').write_text(json.dumps({'unique_h0_queries':len(cache),'splits':reports},indent=2)+'\n')


if __name__=='__main__':main()
