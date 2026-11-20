"""Check the physical intervention and native identity before fresh-seed execution."""
import json
from pathlib import Path
import numpy as np
import campaign as c
from auditembodied.p0 import same_prefix

root=Path(__file__).resolve().parent
p=json.loads((root/'protocol.json').read_text());c.verify(p)
rows=[json.loads(x) for x in (root/'engineering.jsonl').open()]
assert len(rows)==24 and not any(r.get('error') for r in rows)
groups={}
for r in rows:groups.setdefault((r['load'],r['scenario']),[]).append(r)
assert all(same_prefix(g) for g in groups.values())
native=next(r['physical_parameters'] for r in rows if r['load']=='native')
shift=next(r['physical_parameters'] for r in rows if r['load']=='density150')
for key in ('body_mass_kg','body_inertia_kg_m2','geometry_density_kg_per_m3'):
    np.testing.assert_allclose(shift[key],np.asarray(native[key])*1.5,rtol=1e-12,atol=1e-14)
assert all(native[k]==shift[k] for k in native if k not in
    ('body_mass_kg','body_inertia_kg_m2','geometry_density_kg_per_m3','total_object_mass_kg','body_ipos'))
np.testing.assert_allclose(shift['body_ipos'],native['body_ipos'],rtol=0,atol=1e-14)
assert all(r['audit']['cost_identity'] for r in rows)
assert all(not r['blocked_calls'] and not r['state_edit_count_after_initialization'] and not r['contact_anchor_count'] for r in rows)
assert all(set(r['selection']['input'])=={'positions','fingers','reference_positions'} for r in rows)
gated=c.StartGatedMLP(json.loads((root/'direct-selected.json').read_text())['mlp'])
assert all(gated(r['selection']['input'])==r['selection']['decision'] for r in rows if r['method']=='gated_direct')
for load in c.LOADS:
 for scenario in ('left_loss','sham'):
  g=[r for r in rows if r['load']==load and r['scenario']==scenario and r['window']=='1000ms']
  assert all(r['post_wait_state']==g[0]['post_wait_state'] for r in g)
# The factor-one wrapper must reproduce the original physical runner, not just
# another execution of the new wrapper. Decisions and all recorded states/actions agree.
baseline=c.runner.run_trial(60,'left_loss',config=c.runner.LiftConfig(**p['test_config']),
    selector=c.ProprioRule(**p['rule']),account_selector_latency=True)
reference=next(r for r in rows if (r['load'],r['scenario'],r['method'],r['window'])==('native','left_loss','protected','actual'))
fields=('initialization_state','preclose_state','initialization_end_state','prefix_state','post_wait_state',
        'candidate','trace','success','protection_violation','protection','total_post_fault_seconds')
assert all(baseline[k]==reference[k] for k in fields)
c.save(root/'native-identity-check.json',{'seed':60,'fields':fields,'exactly_equal':True})
result={'mechanics_pass':True,'engineering_episodes':24,'development_seed':60,
    'native_mass_kg':native['total_object_mass_kg'],'shifted_mass_kg':shift['total_object_mass_kg'],
    'mass_and_inertia_scale':1.5,'friction_geometry_actuators_identical':True,
    'max_compiler_center_of_mass_roundoff_m':float(np.max(np.abs(np.asarray(shift['body_ipos'])-native['body_ipos']))),
    'native_wrapper_reproduces_original_runner':True,'paired_prefixes_exact':True,
    'fixed_wait_poststates_match':True,'frozen_mlp_decisions_reproduced':True,
    'models_thresholds_skills_unmodified':True,'test_outcomes_used':False,
    'initial_metadata_failure':'One initial saved row failed before physical initialization because density metadata is a list. Fixed recording type only; no physical parameter search.'}
c.save(root/'engineering-check.json',result)
print(json.dumps(result,indent=2))
