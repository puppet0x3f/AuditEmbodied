from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied.handover_audit import trial


if __name__=='__main__':
    p=json.loads((ROOT/'protocol.json').read_text());out=ROOT/'engineering-holdpose'
    cases=[('nominal','visual','actual'),('object_sham','rule','actual'),
           ('object_rotation','proprio','actual'),('object_rotation','R3','actual')]
    cases += [('object_sham',method,window) for window in ['1000ms','2000ms'] for method in ['visual','R3']]
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(trial,60,q,m,w,out,p) for q,m,w in cases]
        for f in as_completed(futures):
            r=f.result();rows.append(r)
            with (out/'trials.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
            print(r['method'],r['scenario'],r['window'],r['audit'],flush=True)
    for window in ['1000ms','2000ms']:
        pair=[r for r in rows if r['scenario']=='object_sham' and r['window']==window]
        assert pair[0]['prefix_state']==pair[1]['prefix_state']
        assert pair[0]['decision_events'][0]['post_wait_state']==pair[1]['decision_events'][0]['post_wait_state']
    assert all(not r.get('error') and r['audit']['cost_identity'] for r in rows)
    print('Eight engineering records valid; fixed-window paired states exact.',flush=True)
