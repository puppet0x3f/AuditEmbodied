"""Evaluate the predeclared single-window gate, final checkpoint only."""
from pathlib import Path
import json

from auditembodied.two_window_analysis import family_metrics

root=Path(__file__).resolve().parent
old=root.parent/'vlm-two-window-rgbd-20261109/full1024'
new=root/'corrected1024'
read=lambda p:[json.loads(s) for s in p.open()]
current={split:family_metrics(read(new/f'{split}-predictions.jsonl')) for split in ('train','development')}
(new/'family-analysis.json').write_text(json.dumps(current,indent=2)+'\n')
prior=json.loads((old/'family-analysis.json').read_text())
report=json.loads((new/'report.json').read_text())
limits=json.loads((root/'protocol.json').read_text())['gates']
train=current['train']['policies']['single'];dev=current['development']['policies']['single']
checks={
 'train_candidate_false_admissions':report['evaluations']['train']['false_admissions']<=limits['train_candidate_false_admissions_max'],
 'train_single_model_selected_unsafe':train['model_selected_unsafe_choices']<=limits['train_single_model_selected_unsafe_max'],
 'train_single_safe_coverage':train['safe_state_coverage']>=limits['train_single_safe_coverage_min'],
 'train_single_necessary_upgrades':train['needed_upgrades']-train['needed_upgrade_misses']>=limits['train_single_necessary_upgrade_successes_min'],
 'train_single_no_safe_rejections':train['no_safe_rejected']>=limits['train_single_no_safe_rejections_min'],
 'development_single_safe_coverage':dev['safe_state_coverage']>=limits['development_single_safe_coverage_min'],
 'development_single_unnecessary_upgrades':dev['unnecessary_upgrades']<=limits['development_single_unnecessary_upgrades_max']}
gate={'passed':all(checks.values()),'checks':checks,'limits':limits,
 'before':prior,'after':current,
 'candidate_metrics_before':json.loads((old/'report.json').read_text())['evaluations'],
 'candidate_metrics_after':report['evaluations'],
 'development_risk_positives_available':False,
 'checkpoint_selection':'Final1024 only; no retuning if gate fails.',
 'training_no_safe_states_have_empirical_labels_only':True}
(root/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({'passed':gate['passed'],'checks':checks,
 'train_before':prior['train']['policies']['single'],'train_after':train,
 'development_before':prior['development']['policies']['single'],'development_after':dev},indent=2))
