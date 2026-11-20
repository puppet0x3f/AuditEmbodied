"""One bounded feasibility batch, using previously used development seeds only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent
CASES=[
 ('v5-heavy-right',61,'right_loss','visual','density150','actual'),
 ('v5-heavy-sham',60,'sham','visual','density150','actual'),
 ('v5-native-left',61,'left_loss','visual','native','actual'),
 ('v5-native-sham',61,'sham','visual','native','actual'),
 ('v5-heavy-right-wait',61,'right_loss','visual','density150','1000ms'),
 ('v5-baseline-heavy-left',60,'left_loss','retarget','density150','actual')]
def run(case):
 tag,seed,scenario,method,load,window=case
 with (ROOT/(tag+'.log')).open('w') as stream:
  r=subprocess.run([sys.executable,str(ROOT/'campaign.py'),'trial','--seed',str(seed),
   '--scenario',scenario,'--method',method,'--load',load,'--window',window,'--tag',tag],stdout=stream,stderr=subprocess.STDOUT)
 assert r.returncode==0,(tag,r.returncode)
 row=json.loads((ROOT/'development'/tag/'trial.json').read_text())
 print(json.dumps({'tag':tag,'label':row['label'],'reason':row['failure_reason'],'error':row.get('error')}),flush=True)
 return tag
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,CASES))
