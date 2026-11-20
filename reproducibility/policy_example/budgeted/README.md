# Arrival-adaptive CAD physical example

Use the same separate robosuite 1.5.2 / MuJoCo 3.3.7 runtime as the lifting examples.
This is a new controller version. Original CAD, models and scoring thresholds remain unchanged.
The frozen controller extends pre-closing feedback by at most two seconds, keeping the
original placement, perception, targets, wrists, gripper settings and two-attempt limit.

From this directory, with the simulator environment activated:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_example.py --out /tmp/ae-adaptive-example --seed 99 --scenario left_loss --hold measured_position --steps 20
```

The destination must be new. This executes a fresh continuous physical trial and records
native contacts plus passive joint diagnostics. It is distinct from NumPy-only recomputation
of the 240 existing formal outcomes. The common evaluation budget is 45 seconds including
terminal hold; nonpreemptible service overruns remain charged. Development failures and
controller selection are documented in the frozen protocol and supplement.
