"""Matched-state attribution of the frozen visual head's renderer sensitivity."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
STUDY=ROOT/'outputs/handover-audit-20261109'
sys.path.insert(0,str(STUDY/'frozen-source'))
from auditembodied.visual_recovery import MultiSupportVerifier,visual_features
from auditembodied.handover_audit import model_plan


def main():
    model=MultiSupportVerifier.load(STUDY/'models/visual-support.npz')
    p=OUT/'visual-probes';cases=[];images={}
    for seed in (60,61,500,40000):
        warm=1 if seed in (61,40000) else 0
        name=f'{seed}-warm{warm}'
        egl=np.load(p/(name+'-remote-egl.npz'));meta=json.loads((p/(name+'-remote-egl.json')).read_text())
        if seed==40000:
            cpu=np.load(STUDY/'observations/40000-object_sham-visual-actual.npz')
            rgb,robot=cpu['rgb'][0],cpu['robot'][0]
            with (STUDY/'seed-shards/40000.jsonl').open() as f:
                original=next(r for r in map(json.loads,f) if r['scenario']=='object_sham' and r['method']=='visual' and r['window']=='actual')
            equal=original['prefix_state']==meta['prefix_state']
        else:
            cpu=np.load(p/(name+'.npz'));rgb,robot=cpu['rgb'],cpu['robot']
            equal=all(np.array_equal(cpu[k],egl[k]) for k in ('qpos','qvel','ctrl','act','robot','cam_xpos','cam_xmat'))
            assert json.loads((p/(name+'.json')).read_text())['prefix_state']==meta['prefix_state']
        assert equal and np.array_equal(robot,egl['robot'])
        a=model.predict_proba(rgb[None],robot[None])[0]
        b=model.predict_proba(egl['rgb'][None],robot[None])[0]
        features=[visual_features(x[None],robot[None]) for x in (rgb,egl['rgb'])]
        contributions=[(x-model.mean)/model.scale*model.weights[:,0] for x in features]
        cases.append({'seed':seed,'split':'development' if seed<100 else 'known historical test' if seed==500 else 'known CPU audit test',
            'same_physics_and_robot_input':equal,'cpu_probabilities':a.tolist(),'egl_probabilities':b.tolist(),
            'cpu_plan':model_plan(a,model.thresholds),'egl_plan':model_plan(b,model.thresholds),
            'rgb_mean_absolute_difference_per_view':[float(np.abs(x.astype(float)-y).mean()) for x,y in zip(rgb,egl['rgb'])],
            'cpu_rgb_giver_logit':float(contributions[0][0,:-46].sum()),
            'egl_rgb_giver_logit':float(contributions[1][0,:-46].sum()),
            'identical_robot_giver_logit':float(contributions[0][0,-46:].sum()),
            'egl_renderer':meta['gl_renderer']})
        images[seed]=(rgb,egl['rgb'])
    old=np.load(ROOT/'outputs/l3-visual-test-20261107/test/inputs/500-object_sham-visual.npz')
    assert np.array_equal(old['rgb'][0],images[500][1])
    cold=np.load(p/'60-warm0.npz');warm=np.load(p/'60-warm1.npz')
    assert all(np.array_equal(cold[k],warm[k]) for k in cold.files)
    cold_egl=np.load(p/'60-warm0-remote-egl.npz');warm_egl=np.load(p/'60-warm1-remote-egl.npz')
    assert all(np.array_equal(cold_egl[k],warm_egl[k]) for k in cold_egl.files)
    hashes={name:hashlib.sha256((STUDY/'models'/name).read_bytes()).hexdigest() for name in ('visual-support.npz','support-robot.npz')}
    protocol=json.loads((STUDY/'protocol.json').read_text());assert hashes==protocol['model_hashes']
    result={'scope':'Four known prefix diagnostics, not new independent policy evaluations.',
        'cause':'Changing the rendering stack from NVIDIA EGL to CPU Mesa changes RGB features enough to cross frozen head thresholds. Physics, robot inputs and camera geometry are unchanged.',
        'matched_cases':cases,'original_seed500_egl_rgb_reproduced_bitwise':True,
        'warm_vs_cold_cpu_identical':True,'warm_vs_cold_egl_identical':True,
        'camera_order':['frontview','agentview','robot0_eye_in_hand','robot1_eye_in_hand'],
        'image_convention':'opengl on both stacks; no image flip or normalization change',
        'model_hashes_unchanged':hashes,'thresholds_unchanged':model.thresholds.tolist(),
        'reporting':'Keep the720CPUepisodes and all original scores. Identify them as rendering-stack transfer, not an isolated new-seed loss. EGL probes restore local choices, but provide no new closed-loop success or cost estimate.',
        'failed_setup_probes':['Local EGL unavailable: no enumerated EGL device.','Remote OSMesa missing rendering dependencies. Neither failure reached a physical prefix or enters policy denominators.']}
    (OUT/'visual-cause.json').write_text(json.dumps(result,indent=2)+'\n')
    # Actual observations, plus absolute RGB differences, with a common scale.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a,b=images[500];fig,axes=plt.subplots(3,4,figsize=(6.8,4.3))
    for i in range(4):
        axes[0,i].imshow(b[i]);axes[1,i].imshow(a[i])
        axes[2,i].imshow(np.abs(a[i].astype(float)-b[i]).mean(2),cmap='magma',vmin=0,vmax=32)
        axes[0,i].set_title(result['camera_order'][i],fontsize=8)
    for row,label in zip(axes,['NVIDIA EGL','CPU Mesa','Abs. difference']):
        row[0].set_ylabel(label,fontsize=9)
        for ax in row:ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout()
    for ext in ('pdf','png'):
        fig.savefig(OUT/('renderer-pair.'+ext),dpi=180,bbox_inches='tight')
        fig.savefig(ROOT/'paper/figures'/('renderer-pair.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'cases':cases,'reproduces_original_rgb':True,'warmup_excluded':True},indent=2))


if __name__=='__main__':main()
