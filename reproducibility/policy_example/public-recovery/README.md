# Released-policy recovery after explicit grasp loss

This uses the same separately installed Linux/CUDA, original Diffusion Policy source,
robosuite fork and MuJoCo 2.1 runtime as `../public/README.md`. The unchanged released
Diffusion Policy CNN performs native Transport. No training or scripted recovery fallback
is used. Only the CNN checkpoint is needed; exact download information is in the new
`formal-freeze.json` and `download_checkpoints.py`.

At the first three-step shared payload grasp, the receiver is physically opened for
12 control steps while the donor remains closed. An equal-duration sham closes both.
The requested holding reference and additional 1/20 steps then run physically. The
remaining stale action chunk is discarded, the latest two observations are refreshed,
and the frozen policy is queried. The 700 active policy action steps remain intact.
Fault validity is evaluator-only; invalid events retain their assigned denominators.

Follow the dependency and pinned source setup in `../public/README.md`, placing
`official-source/diffusion_policy-main` in this directory. From a fresh copy of
this directory:

```bash
python download_checkpoints.py
AE_PUBLIC_PYTHON=python AE_PUBLIC_GPU=0 bash run_recovery.sh --phase development
AE_PUBLIC_PYTHON=python AE_PUBLIC_GPU=0 bash run_recovery.sh --phase formal
```

Development uses ten separate seeds and fault/sham controls. Formal uses the frozen
20 new seeds and eight fault/reference/wait conditions: 160 new executions. The
common pre-event states and post-fault states are checked; existing outputs are never
overwritten. This physical rerun is optional and separate from NumPy-only scoring.

Native bin completion and valid-prefix recovery R are separate from lifting Y500.
Timing retains active control, fault/sham and waiting steps, actual batch query time,
and the legacy observable-fetch timer. Camera acquisition is not isolated by that
fetch timer. These native synchronous components are not complete deployment latency
and must not be compared directly with lifting's complete physical-service costs.
