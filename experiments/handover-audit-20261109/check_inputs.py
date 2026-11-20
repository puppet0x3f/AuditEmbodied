"""Reproduce frozen predictions from the exact saved permitted inputs."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen-source'))
from auditembodied.visual_recovery import MultiSupportVerifier

models={m:MultiSupportVerifier.load(ROOT/'models'/name) for m,name in
        [('visual','visual-support.npz'),('proprio','support-robot.npz')]}
count=0;maximum=0.;ranges={m:[] for m in models}
for row in json.loads((ROOT/'episodes-without-traces.json').read_text()):
    method=row['method']
    if method not in models:continue
    model=models[method]
    with np.load(row['policy_input_file']) as saved:
        probabilities=model.predict_proba(saved['rgb'],saved['robot'])
    expected=np.asarray([event['evidence']['probabilities'] for event in row['decision_events']])
    error=float(np.max(np.abs(probabilities-expected)))
    maximum=max(maximum,error);count+=len(probabilities)
    assert np.allclose(probabilities,expected,rtol=0,atol=1e-12)
    assert all(event['evidence']['thresholds']==model.thresholds.tolist() for event in row['decision_events'])
    if row['scenario']!='nominal' and row['window']=='actual':ranges[method].append(probabilities[0].tolist())
result={'reconstructed_predictions':count,'maximum_absolute_probability_error':maximum,
        'all_original_thresholds_identical':True,
        'measured_fault_initial_probability_range':{m:{'minimum':np.min(v,axis=0).tolist(),
            'maximum':np.max(v,axis=0).tolist()} for m,v in ranges.items()},
        'interpretation':'Frozen visual chooses global on all40fault states in this CPU-rendered cohort; no cause is inferred from this observation alone.'}
(ROOT/'input-reconstruction.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
